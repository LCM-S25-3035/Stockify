import yfinance as yf
import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Finance stocks and ETFs
stocks = ['JPM', 'BAC', 'GS', 'MS']
etfs = ['XLF', 'ZEB.TO']
all_symbols = stocks + etfs

# Download adjusted close prices
data = yf.download(all_symbols, start="2015-01-01", end="2025-01-01")['Close']
data = data.dropna()

# Normalize
df_yf_normalized = data / data.iloc[0]
df_yf_normalized.columns = [f"{col}_Stock" if col in stocks else f"{col}_ETF" for col in df_yf_normalized.columns]

df_yf_normalized

# Custom Gym Environment for Portfolio Allocation
class FinancePortfolioEnv(gym.Env):
    def __init__(self, price_data):
        super(FinancePortfolioEnv, self).__init__()
        self.price_data = price_data.values
        self.n_assets = self.price_data.shape[1]
        self.n_days = self.price_data.shape[0]
        self.current_step = 0

        # Action space: allocation weights (continuous values from 0 to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Observation space: normalized prices for each asset at current step
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Track portfolio value
        self.initial_investment = 10000
        self.portfolio_value = self.initial_investment
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)

    def reset(self, seed=None):
        super().reset(seed=seed) # Add this line to call the parent reset with seed
        self.current_step = 0
        self.portfolio_value = self.initial_investment
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)
        return self.price_data[self.current_step].astype(np.float32), {} # Return observation and info dictionary

    def step(self, action):
        # Normalize action to sum to 1 (valid allocation)
        action = np.clip(action, 0, 1)
        action /= np.sum(action)

        # Price change between steps
        prev_prices = self.price_data[self.current_step]
        self.current_step += 1
        done = self.current_step >= self.n_days - 1
        new_prices = self.price_data[self.current_step]

        # Compute portfolio return
        price_ratios = new_prices / prev_prices
        portfolio_return = np.dot(price_ratios, self.weights)
        old_value = self.portfolio_value # Store old value before updating
        self.portfolio_value *= portfolio_return
        self.weights = action

        # Reward is the increase in portfolio value
        # reward = self.portfolio_value - self.initial_investment
        reward = (self.portfolio_value - old_value) / old_value
        info = {
            'portfolio_value': self.portfolio_value,
            'prices': new_prices.tolist(), # Include prices in info
            'weights': self.weights.tolist(), # Include weights in info
            'reward': reward # Include reward in info
        }

        return new_prices.astype(np.float32), reward, done, False, info # Return observation, reward, terminated, truncated, info

def train_ddpg_model(env, timesteps=10000, sigma=0.1, verbose=0):
    """Train a DDPG model on a given environment."""
    n_actions = env.action_space.shape[0]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=noise, verbose=verbose)
    model.learn(total_timesteps=timesteps)
    return model
    
def save_model(model, path="ddpg_portfolio_model.zip"):
    """Save the trained DDPG model to disk."""
    model.save(path)
    print(f"Model saved to: {path}")

def evaluate_model(env, model, steps=10):
    """Evaluate the trained model and return a list of logs."""
    obs, info = env.reset()
    log_data = []

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        log_data.append({
            "Time Step": step,
            "Prices": info["prices"],
            "Allocation (Action)": info["weights"],
            "Portfolio Value": f"${info['portfolio_value']:.2f}",
            # "Reward": f"{info['reward'] * 100:+.2f}%"
            "Reward": f"{reward * 100:+.2f}%"
        })

        if terminated or truncated:
            break

    return log_data

def show_results_table(log_data):
    """Prints a tabulated table of the evaluation results."""
    from tabulate import tabulate
    print(tabulate(log_data, headers="keys", tablefmt="grid"))

def run_portfolio_pipeline(df_normalized, train_steps=10000, eval_steps=10):
    env = FinancePortfolioEnv(df_normalized)
    check_env(env, warn=True)
    model = train_ddpg_model(env, timesteps=train_steps)
    logs = evaluate_model(env, model, steps=eval_steps)
    show_results_table(logs)

#Final execution
run_portfolio_pipeline(df_yf_normalized, train_steps=10000, eval_steps=10)

# Plot final performance
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label='DDPG Portfolio Value')
plt.axhline(env.initial_investment, color='r', linestyle='--', label='Initial Investment')
plt.title("Portfolio Value Over Time (DDPG Evaluation)")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()