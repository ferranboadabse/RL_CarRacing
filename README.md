<h1 align="center">🏁 Car Racing 🏎️ — Reinforcement Learning Project 🏎️</h1>
<h2 align="center">Final Project – Autonomous Driving with RL</h2>

<p align="center">
  <b>Ferran Boada Bergadà, Julian Romero & Simon Vellin</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python">
  <img src="https://img.shields.io/badge/stable--baselines3-2.2.1-orange?logo=python">
  <img src="https://img.shields.io/badge/gymnasium-0.29.1-green?logo=openai">
  <img src="https://img.shields.io/badge/numpy-1.26.4-blue?logo=numpy">
  <img src="https://img.shields.io/badge/matplotlib-3.8.4-yellow?logo=matplotlib">
  <img src="https://img.shields.io/badge/opencv-4.9.0-critical?logo=opencv">
</p>

---

## 🚀 Overview

This project applies **Reinforcement Learning (RL)** techniques to a simulated autonomous car driving task using OpenAI’s **CarRacing-v3** environment.  
We implement and compare two value-based RL approaches:

- **Tabular Q-Learning**  
- **Deep Q-Learning (DQN)**

Both models operate on a **discrete action space**:  
**`[do nothing, steer left, steer right, accelerate, brake]`**

---

## 🧠 RL Methods

### 📊 Tabular Q-Learning – State-Action Approach

We discretize each `96×96×3` frame into an `8×8` grid, converting to grayscale or RGB with 32/64 bin quantization. A Q-table is updated using the Bellman equation and ε-greedy policy (ε from 1.0 → 0.1), across 1,000–100,000 episodes.

#### 🖼️ State Representation (Discretization)

**Grayscale Discretization**  
<img src="tabular_q_learning/results/discretization_bin_32_use_rgb_False.svg" alt="Grayscale Discretization"/>

**RGB Discretization**  
<img src="tabular_q_learning/results/discretization_bin_32_use_rgb_True.svg" alt="RGB Discretization"/>

#### 📈 Model Performance
<img src="tabular_q_learning/results/q_learning_performance_metrics.svg" alt="Tabular Q-learning Performance"/>

#### 🏎️ Agent in Action
![Tabular QL Agent](tabular_q_learning/results/videos/tabular_carracing.gif)

---

### 🤖 Deep Q-Learning (DQN) – CNN-Based Approach

We use **Stable-Baselines3’s DQN** with a convolutional neural network (CNN) policy that maps raw pixel observations to Q-values.  
The agent is trained using:
- **4-frame stacking** (`VecFrameStack`)
- **Parallel environments** (`SubprocVecEnv`)
- **Replay buffer** of 50,000
- **Learning rate**: 3×10⁻⁴  
- **ε-greedy decay** from 1.0 → 0.05 over the first 20% of training  
- **Target network updates** every 1,000 steps  
- **Training steps**: 200,000

#### 🧠 CNN Feature Maps (First Layer Output)

Raw input vs first CNN features:

<img src="dqn_learning/results/illustration_cnn_raw_input.png" alt="Raw Input to CNN" width="300"/>
<img src="dqn_learning/results/illustration_cnn_features_input_layer.png" alt="CNN Layer Output" width="300"/>

#### 📈 Model Performance
<img src="dqn_learning/results/run_2025-06-29_18-35/mean_rewards_plot.svg" alt="DQN Performance" width="600"/>

#### 🏎️ Agent in Action
![DQN-CNN Agent Performance](dqn_learning/results/run_2025-06-29_18-35/dqn_carracing.gif)

---

## 📁 Repository Structure

```text
├── dqn_learning  
│   ├── training.py               # DQN training script using SB3  
│  
├── tabular_q_learning
│   ├── agents.py                 # Q-learning agent and discretization  
│   ├── experiments.py            # Experiment config runner  
│   ├── results_analysis.ipynb    # Jupyter for metrics & plots  
│   ├── training.py               # Training loop  
│   ├── results                   # Images, SVGs, and GIFs  
│   └── utils.py                  # Plotting, JSON utils, etc.
```
---
## Environment Setup

We use [uv](https://docs.astral.sh/uv/) for environment management. To run the 

```bash
# Install uv for Python environment management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and sync the environment from pyproject.toml
uv sync

# Activate environment
source .venv/bin/activate

#Run scripts
python dqn_learning/training.py
python tabular_q_learning/training.py
```