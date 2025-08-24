# RL PPO Bipedal Walker (portable)

A packaged version of the Colab notebook, with the same PPO + reward shaping logic,
but without any Colab-specific paths or magics. Runs locally, saves results and videos
under `results/` and `videos/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt