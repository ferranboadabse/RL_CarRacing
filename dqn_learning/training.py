import os
import sys
import json
import imageio
import numpy as np
import gymnasium as gym
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (DummyVecEnv, 
                                              VecFrameStack, 
                                              SubprocVecEnv, 
                                              VecTransposeImage
)

def make_env():
    def _init():
        env = gym.make(
            "CarRacing-v3",
            render_mode=None,
            continuous=False
        )
        env = Monitor(env)  # Log episode rewards with Monitor
        env.reset(seed=42)
        return env
    return _init


def train(experiment_start, n_timesteps=50_000):
    num_envs = 4

    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    # Despite is done automatically, enforce to avoid errors in compatibility with DQN CNN policy
    envs = VecFrameStack(envs, n_stack=4)  # Stack frames for DQN
    envs = VecTransposeImage(envs) 

    # Ensure evaluation environment is the same as training to avoid errors
    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./results/run_{experiment_start}/best_model/',
        log_path=f'./results/run_{experiment_start}/results/',
        eval_freq=2_500, # Evaluation frequency
        n_eval_episodes=3, # More episodes smooth out evaluation reward estimates
        deterministic=True,
        render=False,
        verbose=1,
    )
    parameters = {
        "learning_rate": 3e-4, # How fast the model updates its Q-values. Smaller = slower but more stable learning.
        "buffer_size": 50_000, # Size of replay buffer. Stores past experiences to sample from during training.
        "batch_size": 128, # Number of experiences sampled from replay buffer per update. Larger batch = more stable updates.
        "learning_starts": 10_000, # Number of timesteps before learning begins. Helps the replay buffer fill with experiences first.
        "gamma": 0.99, # Discount factor. Determines how much future rewards are considered. Closer to 1 = long-term focused.
        "exploration_initial_eps": 1.0, # Start of ε-greedy exploration. Agent takes completely random actions initially.
        "exploration_final_eps": 0.05, # Final ε value. Even late in training, there's a 5% chance of random actions (exploration).
        "exploration_fraction": 0.1, # Fraction of total training steps over which ε linearly decays from 1.0 to 0.05.
        "target_update_interval": 1_000, # How often (in steps) the target network is updated. More frequent updates can stabilize learning.
        "train_freq": 4, # How often the model trains (every 4 environment steps).
        "gradient_steps": 1, # Number of gradient steps taken per training iteration. Usually 1 for DQN.
        "n_steps": n_timesteps, # Total number of training steps.
    }
    #save training parameters in json file
    with open(f"./results/run_{experiment_start}/training_parameters.json", "w") as f:
        json.dump(parameters, f, indent=4)

    model = DQN(
        "CnnPolicy",
        envs,
        learning_rate=parameters["learning_rate"],
        buffer_size=parameters["buffer_size"],
        learning_starts=parameters["learning_starts"],
        batch_size=parameters["batch_size"],
        gamma=parameters["gamma"],
        exploration_initial_eps=parameters["exploration_initial_eps"],
        exploration_final_eps=parameters["exploration_final_eps"],
        exploration_fraction=parameters["exploration_fraction"],
        target_update_interval=parameters["target_update_interval"],
        train_freq=parameters["train_freq"],
        gradient_steps=parameters["gradient_steps"],
        verbose=2,
        tensorboard_log=f"./results/run_{experiment_start}/tensorboard/",

    )

    model.learn(total_timesteps=n_timesteps, callback=eval_callback)
    model.save(f"results/run_{experiment_start}/sb3_car_racing_dqn")

    envs.close()
    eval_env.close()
    return


def plot_results(experiment_start):
    # Path to evaluation logs (must match EvalCallback's log_path)
    log_path=f'./results/run_{experiment_start}/results/evaluations.npz'
    try:
        data = np.load(log_path)
        timesteps_logged = data["timesteps"]
        mean_rewards = data["results"].mean(axis=1)

        plt.plot(timesteps_logged, mean_rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Evaluation Reward")
        plt.title("DQN Evaluation Rewards Over Time")
        plt.grid()
        plt.savefig(f"./results/run_{experiment_start}/mean_rewards_plot.svg", format='svg')
        plt.close()
        # plt.show()
    except Exception as e:
        print("Plotting failed:", e)


def make_video(experiment_start):
    video_path = f"results/run_{experiment_start}/dqn_carracing.mp4"
    best_model_path = f"./results/run_{experiment_start}/best_model/best_model.zip"

    print("Loading best model for video generation...")
    model = DQN.load(best_model_path)
    print("Model loaded successfully.")

    eval_env = DummyVecEnv([lambda: gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=False
    )])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    obs = eval_env.reset()
    done = False
    frames = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        frame = eval_env.render()
        frames.append(frame)

    eval_env.close()
    imageio.mimsave(video_path, [np.array(f) for f in frames], fps=30)
    return

if __name__ == "__main__":

    experiment_start = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"Experiment started at {experiment_start}")

    PRINT_IN_LOG = True # Set to True to print logs in a file, False to print in console
    if PRINT_IN_LOG:
        os.makedirs(f"./results/run_{experiment_start}/", exist_ok=True)
        log_file = open(f"./results/run_{experiment_start}/trainig.log", "w")
        sys.stdout = log_file
    sys.stderr = log_file
    train(experiment_start, n_timesteps=200_000)
    plot_results(experiment_start)
    make_video(experiment_start)
    print("Training and evaluation completed successfully.")