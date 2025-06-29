#agents.py
import gymnasium as gym
from collections import defaultdict
import numpy as np
import cv2
import pickle


class CarRacingAgent:
    def __init__(
        self,
        num_actions: int,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        use_rgb: bool = False,
        bin_size: int = 64,
        discount_factor: float = 0.95,
    ):
        """Q-Learning agent with discretized observations."""
        self.num_actions = num_actions
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.use_rgb = use_rgb
        self.bin_size = bin_size

    def discretize_obs(self, obs) -> tuple:
        """Downsample and discretize observation."""
        small_obs = cv2.resize(obs, (8, 8), interpolation=cv2.INTER_AREA)
        if self.use_rgb:
            discretized_r = (small_obs[:, :, 0] // self.bin_size).astype(int)
            discretized_g = (small_obs[:, :, 1] // self.bin_size).astype(int)
            discretized_b = (small_obs[:, :, 2] // self.bin_size).astype(int)
            discretized = np.stack([discretized_r, discretized_g, discretized_b], axis=-1).flatten()
        else:
            gray = np.mean(small_obs, axis=2)
            discretized = (gray // self.bin_size).astype(int).flatten()

        return tuple(discretized)

    def get_action(self, obs_batch) -> np.ndarray:
        """Epsilon-greedy action selection for batched observations."""
        actions = []
        for obs in obs_batch:
            state = self.discretize_obs(obs)
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.num_actions)
            else:
                action = int(np.argmax(self.q_values[state]))
            actions.append(action)
        return np.array(actions)

    def update(self, obs, action, reward, terminated, next_obs):
        """Q-learning update for a single transition."""
        state = self.discretize_obs(obs)
        next_state = self.discretize_obs(next_obs)

        future_q = (not terminated) * np.max(self.q_values[next_state])
        target = reward + self.discount_factor * future_q

        td_error = target - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_error
        self.training_error.append(td_error)

    def decay_epsilon(self):
        """Reduce exploration rate."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_values), f)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            q_values_dict = pickle.load(f)
            self.q_values = defaultdict(lambda: np.zeros(self.num_actions), q_values_dict)
