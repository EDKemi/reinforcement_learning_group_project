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

## Agents

### TD3
1. py -3.11 -m venv .venv 
2. .\.venv\Scripts\Activate.ps1   
3. pip install -r requirements.txt
4. Run python -m agents.td3.train