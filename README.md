# README.md

This repo contains the code for the final project of our Reinforcement Learning course.

## Overview
Using the existing CarRacing-v3 environment from Gymnasium, we propose a comparative study of two value-based RL methods - Tabular Q-Learning and Deep Q-Learning (DQN) - with experiments to evaluate their performance on discrete action spaces (i.e. we use a discrete action space of 5 unit: do nothing, steer right, steer left, accelerate, brake).

## RL Methods


### Tabular Q-Learning - Discrete Approach
Uses a Q-table to store action values for discrete states (state-action pair), trained over multiple episodes with epsilon-greedy exploration, transforming high-dimensional observations (e.g., 8x8 images) into manageable states (rounding line lengths to integers).

### Deep Q-Learning (DQN) - Continuous Approach
Employs a CNN to approximate Q-values from raw pixel inputs, trained over timesteps with a replay buffer. Uses 4 stacked frames and parallel environments, with hyperparameters tuned for stability and performance.

## Repo Structure
- **DQN Learning Training (`training.py`)**  
Implements DQN using the Stable-Baselines3 library to train an agent on CarRacing-v3. The agent processes 4 stacked frames with a CNN policy, trained over 200,000 timesteps across 4 parallel environments.
  
Key features include:
  - Hyperparameters (learning rate 3e-4, buffer size 50,000)
  - Video generation of the trained agent's performance

- **Tabular Q-Learning (`training.py`, `agents.py`, `experiments.py`, 'results_analysis.ipynb')**  
Implements a Tabular Q-Learning agent with discretized observations.

The agent handles CarRacing-v3 using:
  - Downsampled 8x8 images, optionally in RGB or grayscale, with bin sizes 32 or 64.
  - Epsilon-greedy exploration with decay, trained over 1,000 to 100,000 episodes.
  - Multiple configurations tested, saving agents and episode rewards in `./results/`.
  - Video recordings of agent performance in `./results/videos/`.

- **Utilities (`utils.py`)**  
  Provides helper functions:
  - Reads JSON files from `./results/` to aggregate experiment data.
  - Extracts frames from videos with timestamps, saved in `./results/frames_output/`.
  - Plots a grid of video frames for analysis, saved as `./results/frames_output.png`.

## Environment Installation
1. **Clone the Repository**  
   `git clone <repository_url>`

2. **Set Up Environment**  
   - Install Python 3.9 or higher.
   - Create a virtual environment: `python -m venv rl_env`
   - Activate it: `source rl_env/bin/activate` (Linux/Mac) or `rl_env\Scripts\activate` (Windows).

3. **Install Dependencies**  
   Run the following to install required packages:
   ```bash
   pip install gymnasium==0.28.1 stable-baselines3==2.3.2 imageio numpy==1.23.5 matplotlib pandas cv2 tqdm


## Demo

<video width="640" controls>
  <source src="/Users/simonvellin/Downloads/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>