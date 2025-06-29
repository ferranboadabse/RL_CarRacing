import os
import sys
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
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = DQN(
        "CnnPolicy",
        envs,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=128,
        learning_starts=5_000,
        gamma=0.99,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
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
    video_path=f"results/run_{experiment_start}/dqn_carracing.mp4"
    best_model_path=f"./results/run_{experiment_start}/best_model/best_model",

    print("Loading best model from video generation")
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
    #make experiment start as yyy-mm-dd_HH-MM
    experiment_start = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"Experiment started at {experiment_start}")

    PRINT_IN_LOG = True # Set to True to print logs in a file, False to print in console
    if PRINT_IN_LOG:
        os.makedirs(f"./results/run_{experiment_start}/", exist_ok=True)
        log_file = open(f"./results/run_{experiment_start}/trainig.log", "w")
        sys.stdout = log_file
    sys.stderr = log_file
    train(experiment_start, n_timesteps=50_000)
    plot_results(experiment_start)
    make_video(experiment_start)
    print("Training and evaluation completed successfully.")