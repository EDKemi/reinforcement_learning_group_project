import numpy as np
import gymnasium as gym

class TrickWalker(gym.Wrapper):
    """
    Reward shaping for BipedalWalker (normal & Hardcore).
    - mode="jump": encourage obstacle-aware takeoffs.
    - mode="somersault": add rotation reward while airborne + landing quality.

    Observation notes (Gymnasium BipedalWalker):
      obs[0]  hull angle (rad)
      obs[1]  hull angular velocity
      obs[2]  vx (horizontal vel)
      obs[3]  vy (vertical vel)
      ...
      The last 10 entries are lidar-like terrain scans in [0, 1] (near=small).
      Contact flags vary by build; we detect robustly.
    """
    def __init__(self, env, mode="jump", base_scale=0.6,
                 min_fwd_speed=0.2,
                 airtime_scale=0.002,
                 rot_scale=0.0015,
                 landing_base=0.25,
                 landing_spin_penalty=0.05,
                 takeoff_bonus=0.4,
                 stall_penalty=-0.003,
                 stall_speed=0.15,
                 stall_window=12,
                 obstacle_dist_thresh=0.60):
        super().__init__(env)
        assert mode in {"jump", "somersault"}
        self.mode = mode
        self.base_scale = float(base_scale)
        self.min_fwd_speed = float(min_fwd_speed)
        self.airtime_scale = float(airtime_scale)
        self.rot_scale = float(rot_scale)
        self.landing_base = float(landing_base)
        self.landing_spin_penalty = float(landing_spin_penalty)
        self.takeoff_bonus = float(takeoff_bonus)
        self.stall_penalty = float(stall_penalty)
        self.stall_speed = float(stall_speed)
        self.stall_window = int(stall_window)
        self.obstacle_dist_thresh = float(obstacle_dist_thresh)
        self._reset_counters()

    # --- lifecycle ---
    def _reset_counters(self):
        self.prev_hull_angle = 0.0
        self.rotation_accum = 0.0
        self.airtime_steps = 0
        self._episode_air_steps = 0
        self.flips_completed = 0
        self.landing_bonus_armed = False
        self.prev_both_off = False
        self._ground_slow_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_counters()
        self.prev_hull_angle = float(obs[0])
        return obs, info

    # --- helpers ---
    @staticmethod
    def _wrap_diff(dtheta):
        while dtheta >  np.pi: dtheta -= 2*np.pi
        while dtheta < -np.pi: dtheta += 2*np.pi
        return dtheta

    def _contacts(self, obs):
        """Return (left_contact, right_contact) robustly for common builds."""
        n = len(obs)
        # Try classic Gym layout first: indices 8 and 13
        if n >= 14:
            lc = bool(obs[8]  > 0.5) if abs(obs[8])  <= 1.0 else (obs[8]  > 0.0)
            rc = bool(obs[13] > 0.5) if abs(obs[13]) <= 1.0 else (obs[13] > 0.0)
            return lc, rc
        # Fallback older/alt: 13/14
        if n >= 15:
            return bool(obs[13] > 0.5), bool(obs[14] > 0.5)
        return True, True

    def _obstacle_ahead(self, obs):
        """Heuristic: small lidar value in front means obstacle close."""
        if len(obs) >= 10:
            lidar = np.array(obs[-10:], dtype=float)  # last 10 entries
            # look at the first few forward rays
            front_min = np.min(lidar[:3])
            return front_min < self.obstacle_dist_thresh
        return False

    # --- shaping ---
    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)

        hull_angle = float(obs[0])
        ang_vel    = float(obs[1])
        vx         = float(obs[2])
        vy         = float(obs[3])

        left_c, right_c = self._contacts(obs)
        both_off = (not left_c) and (not right_c)
        obstacle_close = self._obstacle_ahead(obs)
        # tiny forward-velocity shaping so it prefers moving to standing still
        shaped = 0.002 * max(vx, 0.0)

        # unwrap angle
        self.rotation_accum += self._wrap_diff(hull_angle - self.prev_hull_angle)
        self.prev_hull_angle = hull_angle

        # 0) anti-stall on ground: slow and grounded for several steps
        shaped = 0.0
        if (left_c or right_c):
            if abs(vx) < self.stall_speed:
                self._ground_slow_steps += 1
                if self._ground_slow_steps >= self.stall_window:
                    shaped += self.stall_penalty
            else:
                self._ground_slow_steps = 0

        # detect takeoff edge (ground -> air)
        takeoff = (not self.prev_both_off) and both_off
        self.prev_both_off = both_off

        # 1) takeoff bonus if obstacle ahead and moving forward
        if takeoff and obstacle_close and vx >= self.min_fwd_speed:
            shaped += self.takeoff_bonus
            self.landing_bonus_armed = True

        # 2) airtime reward only when moving forward and obstacle close
        if both_off and obstacle_close and vx >= self.min_fwd_speed:
            self.airtime_steps += 1
            self._episode_air_steps += 1
            shaped += self.airtime_scale * self.airtime_steps

        # 3) rotation while airborne (somersault mode)
        if self.mode == "somersault" and both_off and obstacle_close and vx >= self.min_fwd_speed:
            shaped += self.rot_scale * ang_vel
            full_turn = 2*np.pi
            if abs(self.rotation_accum) >= full_turn:
                shaped += 1.0
                self.flips_completed += 1
                self.rotation_accum = np.sign(self.rotation_accum) * (abs(self.rotation_accum) - full_turn)

        # 4) landing quality right after airtime
        if self.landing_bonus_armed and (left_c or right_c):
            shaped += (self.landing_base - self.landing_spin_penalty * abs(ang_vel))
            self.airtime_steps = 0
            self.landing_bonus_armed = False

        # blend with original reward
        total_r = self.base_scale * base_r + shaped

        if terminated or truncated:
            info = dict(info)
            info["trick_metrics"] = {
                "airtime_steps": int(self._episode_air_steps),
                "flips_completed": int(self.flips_completed)
            }
            # reset per-episode
            self._episode_air_steps = 0
            self.airtime_steps = 0
            self.flips_completed = 0
            self.rotation_accum = 0.0
            self._ground_slow_steps = 0
            self.prev_both_off = False

        return obs, total_r, terminated, truncated, info
