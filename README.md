# MADDPG: Multi-Agent Deep Deterministic Policy Gradient in PettingZoo

From-scratch PyTorch implementation of **MADDPG** (Multi-Agent DDPG) trained on PettingZoo's `simple_adversary_v3` environment (continuous actions).  

The environment features:
- 2 "good" agents (green) cooperating to reach landmarks
- 1 adversarial agent (red) trying to push them away

It's a great setup for learning centralized training with decentralized execution in multi-agent continuous control — cooperation + competition at the same time.

I created this project to build on my single-agent DDPG experience from my summer 2023 Research Assistant role at Dalhousie University, where I implemented DDPG in PyTorch + OpenAI Gym to optimize configurations for containerized distributed IoT messaging systems (Mininet + Kafka simulation). This multi-agent version helped me understand shared critics, credit assignment across agents, and stability in competitive continuous spaces — skills directly applicable to multi-agent robotics coordination (e.g., multiple dexterous hands/fingers or humanoid team behaviors at places like Sanctuary AI).

## Features
- Per-agent actor + centralized critic (sees all states & actions)
- Target networks with soft parameter updates (τ)
- Clamped Gaussian exploration noise
- Custom multi-agent replay buffer (separate actor memories + global state)
- Model checkpointing (save best based on 100-episode rolling average)
- Clean handling of PettingZoo parallel dict API → flattened tensors

## Tech Stack
- PyTorch (networks, Adam optimizer, GPU support if available)
- PettingZoo `[mpe]` (simple_adversary_v3, continuous actions)
- NumPy (buffer & data handling)

## Installation
```bash
# Recommended: create a virtual environment first
python -m venv maddpg_env
source maddpg_env/bin/activate    # Linux/Mac
# or maddpg_env\Scripts\activate   # Windows

pip install torch pettingzoo[mpe] numpy
```
## How to run
1. Train the agents
```python
python main.py
```
**Default Training Specs:**
* **Episodes:** 50,000 (`N_GAMES`)
* **Batch Size:** 1024
* **Learning Rate:** 0.1
* **Architecture:** MLP ($128 \to 128$ hidden layers)
* **Checkpoints:** Saved automatically whenever the 100-episode rolling average improves.


- Watch Trained Agents (Rendering)
To visualize the agents' behavior in real-time using render_mode="human":

- Open main.py.

- Set ```evaluate = True.```

#### Run the script
```python main.py```

  - This is useful after training to visually check if agents have learned to reach landmarks despite the adversary.



- Load & Evaluate a Saved Model
The script is configured to load saved models from the ./maddpg/simple directory by default. Ensure your checkpoints are in this folder before running with evaluate = True.

## 🏗 Project Structure

```text
.
├── agent.py      # Actor, Critic, and soft update logic for individual agents
├── networks.py   # MLP network definitions for Actor and Critic
├── buffer.py     # MultiAgentReplayBuffer: handles per-agent & global memories
├── maddpg.py     # MADDPG orchestrator: manages agents & learning steps
└── main.py       # Env creation, training loop, and action selection

