#experiments.py
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from agents import CarRacingAgent

results_dir = "results"
video_dir = os.path.join(results_dir, "videos")
os.makedirs(video_dir, exist_ok=True)

np.random.seed(42)

def train_agent(agent_name, n_episodes=5000, use_rgb=True, bin_size=64):
    num_envs = 4

    def make_env():
        def _init():
            env = gym.make(
                "CarRacing-v3",
                render_mode="state_pixels",
                lap_complete_percent=0.95,
                domain_randomize=False,
                continuous=False
            )
            env.reset(seed=42)
            return env
        return _init

    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    agent = CarRacingAgent(
        num_actions=envs.single_action_space.n,
        learning_rate=0.01,
        initial_epsilon=1.0,
        epsilon_decay=1.0 / (n_episodes / 2),
        final_epsilon=0.1,
        use_rgb=use_rgb,
        bin_size=bin_size
    )

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    episode_rewards = []
    obs, _ = envs.reset()
    episode_reward_tracker = np.zeros(num_envs)

    for _ in tqdm(range(n_episodes)):
        actions = agent.get_action(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        for i in range(num_envs):
            agent.update(obs[i], actions[i], rewards[i], terminations[i], next_obs[i])
            episode_reward_tracker[i] += rewards[i]

            if terminations[i] or truncations[i]:
                episode_rewards.append(episode_reward_tracker[i])
                episode_reward_tracker[i] = 0

        obs = next_obs
        agent.decay_epsilon()

    envs.close()
    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    minutes_taken = (datetime.strptime(end_time, "%Y%m%d_%H%M%S") - datetime.strptime(start_time, "%Y%m%d_%H%M%S")).seconds / 60
    print(f"Training completed in {minutes_taken:.2f} minutes.")

    # Save Agent
    agent.save(os.path.join(results_dir, f"trained_agent_{agent_name}.pkl"))
    #save episode rewards as a json with the number of episodes and agent in a dict
    res = {"agent_name": agent_name, 
           "training_time": round(minutes_taken, 2),
           "action_space": "discrete",
           "use_rgb": use_rgb,
            "bin_size": bin_size,
           "n_episodes": len(episode_rewards), 
           "n_runs": n_episodes,
           "episode_rewards": episode_rewards
           }
    with open(os.path.join(results_dir, f"episode_rewards_{agent_name}.json"), 'w') as f:
        json.dump(res, f)

    return agent

def record_agent(agent, num_episodes=5, agent_name='default_agent'):
    best_reward = None
    for ep in range(num_episodes):
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=False
        )
        obs, _ = env.reset()
        done = False
        frames = []
        total_reward = 0

        while not done:
            frames.append(env.render())
            state = agent.discretize_obs(obs)
            action = int(np.argmax(agent.q_values[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        env.close()
        best_reward = max(best_reward, total_reward) if best_reward is not None else total_reward
        if total_reward < best_reward:
            continue
        # Save Video as MP4
        height, width, _ = frames[0].shape
        out_path = os.path.join(video_dir, f"{agent_name}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved video: {out_path}")