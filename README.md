# Bipedal Walker Reinforcement Learning Project
**Module:** CM500336_2024/5_M11_VP – Reinforcement Learning
**Team Members:** Toby Chitty, Arka Dasgupta, Eloho Kemi, Jason Lutz

The goal is to train a bipedal robot to walk across a procedurally generated terrain with obstacles using **function approximation-based RL algorithms**.  
We compare performance before and after training and evaluate against baseline agents.

---

## Objectives
- Implement a **continuous control RL algorithm** suitable for the Bipedal Walker task.
- Compare trained agent performance with baseline agents.
- Analyse training results, including learning curves and qualitative behaviour.
- Produce report, video presentation, and demonstration videos as per assessment requirements.

## Running SAC training
### Activate the project env
Pyenv local bipedal_walker_rl     # or: source .venv/bin/activate

### Install req
pip install -r requirements.txt

### Train and plot results
- Python run_all.py (check comments in run_all.py first)
- Entropy can be updated in train_sac_tf.py line 47
- Eval saved in "results/{environment}/eval_log.csv"
- Plot results using results.ipynb
- Videos saved in "videos/{env}". Make sure "do_record_after" is set to true in run_all.py