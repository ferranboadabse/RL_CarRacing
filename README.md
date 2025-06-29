# README.md

This repository contains the final project from  Ferran Boadà, Julian Romero, and Simon Vellin in the context of our Reinforcement Learning course.

## Overview
Using the existing CarRacing-v3 environment from Gymnasium, we propose a comparative study of two value-based RL methods - Tabular Q-Learning and Deep Q-Learning (DQN) - to evaluate their performance on discrete action spaces: do nothing, steer right, steer left, accelerate, brake.

## RL Methods

### Tabular Q-Learning  
Discretizes each 96×96×3 frame to an 8×8 grid (RGB or grayscale) with integer bins (32 or 64 levels) and maintains a Q-table over these discrete states and five actions. The agent uses ε-greedy exploration (decaying from 1.0 to 0.1) and updates via the Bellman equation over 1 000–100 000 episodes.

### Deep Q-Learning (DQN) – CNN-Based Approach  
Uses Stable-Baselines3’s DQN with a CNN policy to approximate Q(s,a) from raw pixels. Inputs are four stacked frames (VecFrameStack) from four parallel CarRacing-v3 environments (SubprocVecEnv), trained for 200 000 timesteps with a replay buffer (size 50 000), learning rate 3×10⁻⁴, target-network updates every 1 000 steps, and ε-greedy decay (1.0→0.05 over 20 % of training).

## Repo Structure

```text
.
├── dqn_learning  
│   ├── training.py             # DQN training script (Stable-Baselines3)  
│  
├── tabular_q_learning
│   ├── agents.py               # tabular q-Learning agent (discretization, ε-greedy)  
│   ├── experiments.py          # batch-run different configs (RGB vs grayscale, bin sizes)  
│   └── results_analysis.ipynb  # notebook aggregating plotting metrics and commenting results
│   ├── training.py             # training loop with AsyncVectorEnv  
│   └── results                 # svg files
│   └── utils.py                # helpers to read JSON results, extract video frames, plot grids 
```

## Environment Installation
1. **Clone the Repository**  
   `git clone <repository_url>`

2. **Set Up Environment**  
   - Install Python 3.9 or higher.
   - Create a virtual environment: `python -m venv rl_env`

3. **Install Dependencies**  
   Run the following to install required packages:
   ```bash
   pip install gymnasium==0.28.1 stable-baselines3==2.3.2 imageio numpy==1.23.5 matplotlib pandas cv2 tqdm


## Demo

<video width="640" controls>
  <source src="./demo.mp4" type="video/mp4">
</video>