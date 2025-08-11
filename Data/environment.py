# 2_environment.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class FinancePortfolioEnv(gym.Env):
    def __init__(self, df, initial_investment=10000):
        super().__init__()
        self.df = df.values
        self.n_assets = 6
        self.n_days = self.df.shape[0]
        self.n_features = self.df.shape[1]
        self.initial_investment = initial_investment

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features + self.n_assets,), dtype=np.float32)

        self.reset()

    def _get_observation(self):
        obs = self.df[self.current_step]
        return np.concatenate([obs, self.weights]).astype(np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_investment
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= np.sum(action) if np.sum(action) != 0 else 1.0

        prev_prices = self.df[self.current_step][:self.n_assets]
        self.current_step += 1
        done = self.current_step >= self.n_days - 1
        new_prices = self.df[self.current_step][:self.n_assets]
        price_ratios = new_prices / prev_prices

        portfolio_return = np.dot(price_ratios, self.weights)
        old_value = self.portfolio_value
        self.portfolio_value *= portfolio_return

        transaction_cost = 0.001 * np.sum(np.abs(action - self.weights)) * self.portfolio_value
        self.portfolio_value -= transaction_cost

        self.weights = action
        log_return = np.log(self.portfolio_value / old_value)
        reward = log_return - 0.001 * np.std(price_ratios)

        info = {
            'portfolio_value': self.portfolio_value,
            'reward': reward,
            'transaction_cost': transaction_cost
        }

        return self._get_observation(), reward, done, False, info
