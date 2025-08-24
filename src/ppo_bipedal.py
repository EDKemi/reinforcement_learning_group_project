import os, shutil, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from src.trick_wrapper import TrickWalker

# ---------- small util ----------
def _ensure_parent_dir(path: str):
    """Make sure the parent folder of a file/dir path exists."""
    parent = os.path.dirname(path.rstrip("/"))
    if parent:
        os.makedirs(parent, exist_ok=True)

# ---------- utils ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_t(x, device): return torch.as_tensor(x, dtype=torch.float32, device=device)

def make_env(mode="jump", render_mode=None, base_scale=0.6):
    base = gym.make("BipedalWalker-v3", render_mode=render_mode)
    return TrickWalker(base, mode=mode, base_scale=base_scale)

def make_env_hardcore(mode="jump", render_mode=None, base_scale=0.6):
    base = gym.make("BipedalWalkerHardcore-v3", render_mode=render_mode)
    return TrickWalker(base, mode=mode, base_scale=base_scale)

# ---------- running obs norm ----------
class RunningNorm:
    def __init__(self, shape, eps=1e-8, clip=5.0):
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var  = torch.ones(shape,  dtype=torch.float64)
        self.count = torch.tensor(1e-4, dtype=torch.float64)
        self.eps = eps; self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x64 = x.detach().to(torch.float64).reshape(-1, x.shape[-1])
        if x64.numel() == 0: return
        bmean = x64.mean(0)
        bvar  = x64.var(0, unbiased=False)
        bcnt  = torch.tensor(x64.shape[0], dtype=torch.float64)
        delta = bmean - self.mean
        tot   = self.count + bcnt
        new_mean = self.mean + delta * (bcnt / tot)
        m_a = self.var * self.count
        m_b = bvar * bcnt
        M2  = m_a + m_b + delta.pow(2) * (self.count * bcnt / tot)
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    def normalize(self, x):
        mean = self.mean.to(x.device, dtype=x.dtype)
        std  = torch.sqrt(self.var + self.eps).to(x.device, dtype=x.dtype)
        z = (x - mean) / std
        return torch.clamp(z, -self.clip, self.clip)

# ---------- model ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=24, act_dim=4, hidden=(64,64), activation=nn.Tanh):
        super().__init__()
        def mlp(in_dim, hidden, out_dim):
            layers, last = [], in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), activation()]
                last = h
            layers += [nn.Linear(last, out_dim)]
            return nn.Sequential(*layers)
        self.pi = mlp(obs_dim, hidden, act_dim)
        self.v  = mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        mu = self.pi(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def value(self, obs):
        return self.v(obs).squeeze(-1)

# ---------- buffer ----------
class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.obs  = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act  = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.logp = torch.zeros(size, dtype=torch.float32, device=device)
        self.rew  = torch.zeros(size, dtype=torch.float32, device=device)
        self.done = torch.zeros(size, dtype=torch.float32, device=device)
        self.val  = torch.zeros(size, dtype=torch.float32, device=device)
        self.adv  = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret  = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, o, a, lp, r, d, v):
        i = self.ptr
        self.obs[i]  = o
        self.act[i]  = a
        self.logp[i] = lp
        self.rew[i]  = torch.as_tensor(r, dtype=torch.float32, device=self.rew.device)
        self.done[i] = torch.as_tensor(d, dtype=torch.float32, device=self.done.device)
        self.val[i]  = v
        self.ptr += 1

    def full(self): return self.ptr == len(self.rew)

# ---------- gae ----------
def compute_gae(buf, v_last, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(buf.rew); last = 0.0; T = len(buf.rew)
    for t in reversed(range(T)):
        nnterm = 1.0 - buf.done[t]
        next_v = v_last if t == T-1 else buf.val[t+1]
        delta  = buf.rew[t] + gamma * nnterm * next_v - buf.val[t]
        last   = delta + gamma * lam * nnterm * last
        adv[t] = last
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    buf.adv = adv
    buf.ret = buf.adv + buf.val

# ---------- ppo ----------
def ppo_update(
    ac, optimizer, buf, epochs=10, minibatch=256, clip_eps=0.2,
    vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, value_clip=0.2,
    target_kl=0.03, std_clamp=(-1.9, -1.1)
):
    N = len(buf.rew); idx = np.arange(N)
    stats = {"pi_loss":0.0,"v_loss":0.0,"ent":0.0,"approx_kl":0.0,"clip_frac":0.0,"batches":0}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, N, minibatch):
            mb = idx[start:start+minibatch]
            obs = buf.obs[mb]; act = buf.act[mb]; old_logp = buf.logp[mb]
            adv = buf.adv[mb]; ret = buf.ret[mb]; val_old = buf.val[mb]

            dist = ac.dist(obs)
            logp = dist.log_prob(act).sum(-1)
            ratio = (logp - old_logp).exp()

            pg_loss1 = ratio * adv
            pg_loss2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            pi_loss = -torch.min(pg_loss1, pg_loss2).mean()

            v = ac.value(obs)
            v_clip = val_old + (v - val_old).clamp(-value_clip, value_clip)
            v_loss = 0.5 * torch.max((v - ret).pow(2), (v_clip - ret).pow(2)).mean()

            ent = dist.entropy().sum(-1).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                ac.log_std.clamp_(std_clamp[0], std_clamp[1])
                approx_kl = (old_logp - logp).mean().item()
                clip_frac = (torch.abs(ratio - 1.0) > clip_eps).float().mean().item()

            stats["pi_loss"] += pi_loss.item()
            stats["v_loss"]  += v_loss.item()
            stats["ent"]     += ent.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["batches"] += 1

        # early stop on large KL (average over batches)
        if stats["approx_kl"] / max(1, stats["batches"]) > target_kl:
            break

    for k in list(stats.keys()):
        if k != "batches": stats[k] /= max(1, stats["batches"])
    return stats

# ---------- collect ----------
def collect_steps(env, ac, device, T, obs_norm: RunningNorm):
    buf = RolloutBuffer(T, env.observation_space.shape[0], env.action_space.shape[0], device)
    o, _ = env.reset()
    o_t = to_t(o, device); obs_norm.update(o_t.view(1,-1).cpu()); o_n = obs_norm.normalize(o_t)

    ep_ret=0.0; ep_len=0
    episodic_returns, episodic_airtime, episodic_flips = [], [], []

    while not buf.full():
        with torch.no_grad():
            dist = ac.dist(o_n.unsqueeze(0))
            a = dist.sample()[0]
            logp = dist.log_prob(a).sum(-1)
            v = ac.value(o_n.unsqueeze(0))[0]
        a_env = a.clamp(-1,1).cpu().numpy()
        o2, r, term, trunc, info = env.step(a_env)
        d = float(term or trunc)

        o2_t = to_t(o2, device); obs_norm.update(o2_t.view(1,-1).cpu()); o2_n = obs_norm.normalize(o2_t)
        buf.add(o_n, a, logp, r, d, v)

        o_n = o2_n; ep_ret += r; ep_len += 1
        if d:
            episodic_returns.append(ep_ret)
            if "trick_metrics" in info:
                episodic_airtime.append(info["trick_metrics"].get("airtime_steps", 0))
                episodic_flips.append(info["trick_metrics"].get("flips_completed", 0))
            else:
                episodic_airtime.append(0); episodic_flips.append(0)
            ep_ret, ep_len = 0.0, 0
            o, _ = env.reset()
            o_t = to_t(o, device); obs_norm.update(o_t.view(1,-1).cpu()); o_n = obs_norm.normalize(o_t)

    with torch.no_grad():
        v_last = ac.value(o_n.unsqueeze(0))[0]
    if ep_len > 0:
        episodic_returns.append(ep_ret); episodic_airtime.append(0); episodic_flips.append(0)
    return buf, v_last, episodic_returns, episodic_airtime, episodic_flips

# ---------- eval & video ----------
def _act(ac, o, device, obs_norm, stochastic=False):
    o_t = to_t(o, device); o_n = obs_norm.normalize(o_t)
    with torch.no_grad():
        d = ac.dist(o_n.unsqueeze(0))
        a = d.sample()[0] if stochastic else d.mean[0]
        return a.clamp(-1,1).cpu().numpy()

def evaluate_policy(ac, obs_norm, env_id="BipedalWalker-v3", episodes=5, device="cpu", stochastic=False):
    env = gym.make(env_id, render_mode=None)
    rets=[]
    for _ in range(episodes):
        o,_ = env.reset(); done=False; ep_ret=0.0; steps=0
        while not done and steps<2000:
            a = _act(ac, o, device, obs_norm, stochastic=stochastic)
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc; ep_ret += r; steps+=1
        rets.append(ep_ret)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))

def record_video_policy(ac, obs_norm, out_dir="videos/after", env_id="BipedalWalker-v3",
                        device="cpu", max_steps=1600, stochastic=False):
    _ensure_parent_dir(out_dir)
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    env = RecordVideo(gym.make(env_id, render_mode="rgb_array"), out_dir)
    o,_ = env.reset(); done=False; steps=0
    while not done and steps<max_steps:
        a = _act(ac, o, device, obs_norm, stochastic=stochastic)
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc; steps+=1
    env.close()

def record_video_random(out_dir="videos/before", env_id="BipedalWalker-v3", max_steps=1600):
    _ensure_parent_dir(out_dir)
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    env = RecordVideo(gym.make(env_id, render_mode="rgb_array"), out_dir)
    o,_ = env.reset(); done=False; steps=0
    while not done and steps<max_steps:
        a = np.random.uniform(-1,1, size=env.action_space.shape[0])
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc; steps+=1
    env.close()

def record_video_random_hardcore(out_dir="videos/before_hc",
                                 mode="jump", seconds=20, fps=60):
    _ensure_parent_dir(out_dir)
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    env = make_env_hardcore(mode=mode, render_mode="rgb_array")
    env = RecordVideo(env, out_dir)
    o,_ = env.reset(); done=False; t=0; max_steps=seconds*fps
    while not done and t<max_steps:
        a = np.random.uniform(-1,1, size=env.action_space.shape[0])
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc; t+=1
    env.close()

def record_video_policy_hardcore(ac, obs_norm, out_dir="videos/after_hc",
                                 mode="somersault", device="cpu", seconds=20, fps=60, stochastic=False):
    _ensure_parent_dir(out_dir)
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    env = make_env_hardcore(mode=mode, render_mode="rgb_array")
    env = RecordVideo(env, out_dir)
    o,_ = env.reset(); done=False; t=0; max_steps=seconds*fps
    while not done and t<max_steps:
        a = _act(ac, o, device, obs_norm, stochastic=stochastic)
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc; t+=1
    env.close()

# ---------- training ----------
def train(seed=42, total_updates=180, steps_per_update=4096,
          gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5,
          pi_lr=3e-4, v_lr=None, max_grad_norm=0.5, log_every=1,
          save_path="results/ppo_actorcritic.pt"):
    """
    Curriculum:
      1-79   : Normal terrain, mode='jump' (gait & hop)
      80-119 : Normal terrain, mode='somersault' (learn flips safely)
      120-139: Hardcore BOOTCAMP, mode='jump' (gap clearance without flips)
      140+   : Hardcore, mode='somersault' (combine gaps + flips)
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(mode="jump", render_mode=None, base_scale=0.85)  # generous shaping to start
    phase = "tricks-normal"  # will change over time

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ac = ActorCritic(obs_dim, act_dim).to(device)

    if v_lr is None: v_lr = pi_lr
    with torch.no_grad():
        ac.log_std.fill_(math.log(0.35))  # initial exploration

    optimizer = optim.Adam([
        {"params": ac.pi.parameters(), "lr": pi_lr},
        {"params": ac.v.parameters(),  "lr": v_lr},
        {"params": [ac.log_std],       "lr": 1e-4},
    ])

    obs_norm = RunningNorm(obs_dim)

    print(f"Device: {device} | seed={seed}")

    # pre-videos
    record_video_random(out_dir="videos/before")
    record_video_random_hardcore(out_dir="videos/before_hc")

    # entropy warmups (restart on phase changes)
    ent_warmup_updates = 20; ent_start, ent_end = 0.025, 0.0; warmup_anchor = 1

    history = {"update":[], "mean_ep_ret":[], "std_ep_ret":[],
               "pi_loss":[], "v_loss":[], "ent":[], "approx_kl":[], "clip_frac":[],
               "mean_airtime_steps":[], "mean_flips":[]}

    for upd in range(1, total_updates+1):
        # ---- phase switches ----
        if upd == 80:
            env.mode = "somersault"     # flips on normal terrain
            env.base_scale = 0.75
            phase = "tricks-normal-flips"
            ent_warmup_updates = 12; ent_start, ent_end = 0.03, 0.0; warmup_anchor = upd

        if upd == 120:
            # HARDCORE BOOTCAMP: jump only to learn gap timing/clearance
            env.close()
            env = make_env_hardcore(mode="jump", render_mode=None, base_scale=0.80)
            phase = "hc-bootcamp"
            ent_warmup_updates = 15; ent_start, ent_end = 0.035, 0.0; warmup_anchor = upd

        if upd == 140:
            # combine flips with hardcore
            env.mode = "somersault"
            env.base_scale = 0.65
            phase = "hc-flips"
            ent_warmup_updates = 10; ent_start, ent_end = 0.03, 0.0; warmup_anchor = upd

        if upd == 170:
            # taper shaping to avoid dependency
            env.base_scale = 0.55

        # per-phase std clamp
        if phase == "tricks-normal":
            std_clamp = (-1.9, -1.1)
        elif phase == "tricks-normal-flips":
            std_clamp = (-1.8, -0.7)
        elif phase == "hc-bootcamp":
            std_clamp = (-1.9, -0.9)
        else:  # "hc-flips"
            std_clamp = (-2.1, -1.0)

        # steps per update (slightly larger on Hardcore)
        eff_T = steps_per_update
        if phase in ("hc-bootcamp", "hc-flips"):
            eff_T = max(4096, int(steps_per_update * 1.25))

        # collect rollout
        buf, v_last, ep_returns, ep_air, ep_flips = collect_steps(env, ac, device, T=eff_T, obs_norm=obs_norm)
        compute_gae(buf, v_last, gamma, lam)

        # entropy schedule
        decay = max(0.0, 1.0 - (upd - warmup_anchor) / max(1, ent_warmup_updates))
        ent_coef_cur = ent_end + (ent_start - ent_end) * decay

        stats = ppo_update(
            ac, optimizer, buf, epochs=10, minibatch=256,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef_cur,
            max_grad_norm=max_grad_norm, value_clip=0.2,
            target_kl=0.03, std_clamp=std_clamp
        )

        # logs
        mret = float(np.mean(ep_returns)) if len(ep_returns) else float("nan")
        sret = float(np.std(ep_returns)) if len(ep_returns) else float("nan")
        history["update"].append(upd); history["mean_ep_ret"].append(mret); history["std_ep_ret"].append(sret)
        history["mean_airtime_steps"].append(float(np.mean(ep_air)) if ep_air else np.nan)
        history["mean_flips"].append(float(np.mean(ep_flips)) if ep_flips else 0.0)
        for k in ("pi_loss","v_loss","ent","approx_kl","clip_frac"): history[k].append(stats[k])

        if upd % log_every == 0:
            air = (np.mean(ep_air) if ep_air else 0.0); flips = (np.mean(ep_flips) if ep_flips else 0.0)
            print(f"upd {upd:03d} | ret {mret:7.2f} ± {sret:6.2f} | "
                  f"KL {stats['approx_kl']:.4f} | clip {stats['clip_frac']:.2f} | "
                  f"ent {stats['ent']:.2f} | air {air:5.1f} | flips {flips:4.2f} | phase {phase}")

    # save model
    _ensure_parent_dir(save_path)
    torch.save(ac.state_dict(), save_path)

    # videos: stochastic on hardcore to showcase flips; deterministic on normal
    record_video_policy(ac, obs_norm, out_dir="videos/after", device=str(device), stochastic=False)
    record_video_policy_hardcore(ac, obs_norm, out_dir="videos/after_hc",
                                 mode="somersault", device=str(device), stochastic=True)

    env.close()
    return ac, history
