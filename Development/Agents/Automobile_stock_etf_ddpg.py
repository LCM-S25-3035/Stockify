import yfinance as yf
import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from tabulate import tabulate

# -------------------- Download Financial Data --------------------
# Use reliable stock/ETF tickers
stocks = ['MGA', 'MSFT', 'CVX', 'TSLA']
etfs = ['VUG', 'MGK']
all_symbols = stocks + etfs

# Download adjusted close prices
print("Downloading data...")
data = yf.download(all_symbols, start="2015-01-01", end="2025-01-01")['Close']
data = data.dropna()

if data.empty:
    raise ValueError("Downloaded data is empty. Check ticker symbols or internet connection.")

# Normalize prices
df_yf_normalized = data / data.iloc[0]
df_yf_normalized.columns = [f"{col}_Stock" if col in stocks else f"{col}_ETF" for col in df_yf_normalized.columns]

# -------------------- Custom Gym Environment --------------------
class FinancePortfolioEnv(gym.Env):
    def __init__(self, price_data):
        super(FinancePortfolioEnv, self).__init__()
        self.price_data = price_data.values
        self.n_assets = self.price_data.shape[1]
        self.n_days = self.price_data.shape[0]
        self.current_step = 0

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.n_assets,), dtype=np.float32)

        self.initial_investment = 10000
        self.portfolio_value = self.initial_investment
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_investment
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)
        return self.price_data[self.current_step].astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.array([1.0 / self.n_assets] * self.n_assets)
        else:
            action /= np.sum(action)

        prev_prices = self.price_data[self.current_step]

        if self.current_step >= self.n_days - 1:
            done = True
            new_prices = prev_prices
            reward = 0
        else:
            self.current_step += 1
            new_prices = self.price_data[self.current_step]
            price_ratios = new_prices / prev_prices
            portfolio_return = np.dot(price_ratios, self.weights)
            old_value = self.portfolio_value
            self.portfolio_value *= portfolio_return
            self.weights = action
            reward = (self.portfolio_value - old_value) / old_value
            done = self.current_step >= self.n_days - 1

        info = {
            'portfolio_value': self.portfolio_value,
            'prices': new_prices.tolist(),
            'weights': self.weights.tolist(),
            'reward': reward
        }

        return new_prices.astype(np.float32), reward, done, False, info

# -------------------- DDPG Training --------------------
def train_ddpg_model(env, timesteps=10000, sigma=0.1, verbose=0):
    n_actions = env.action_space.shape[0]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=noise, verbose=verbose)
    model.learn(total_timesteps=timesteps)
    return model

def save_model(model, path="ddpg_portfolio_model.zip"):
    model.save(path)
    print(f"Model saved to: {path}")

# -------------------- Evaluation --------------------
def evaluate_model(env, model, steps=10):
    obs, info = env.reset()
    log_data = []
    portfolio_values = [env.portfolio_value]

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        log_data.append({
            "Time Step": step,
            "Prices": info["prices"],
            "Allocation (Action)": info["weights"],
            "Portfolio Value": f"${info['portfolio_value']:.2f}",
            "Reward": f"{reward * 100:+.2f}%"
        })

        portfolio_values.append(info["portfolio_value"])

        if terminated or truncated:
            break

    return log_data, portfolio_values

def show_results_table(log_data):
    print(tabulate(log_data, headers="keys", tablefmt="grid"))

# -------------------- Full Pipeline --------------------
def run_portfolio_pipeline(df_normalized, train_steps=10000, eval_steps=10):
    env = FinancePortfolioEnv(df_normalized)
    check_env(env, warn=True)
    model = train_ddpg_model(env, timesteps=train_steps)
    logs, portfolio_values = evaluate_model(env, model, steps=eval_steps)
    show_results_table(logs)
    return env, portfolio_values

# -------------------- Execute --------------------
env, portfolio_values = run_portfolio_pipeline(df_yf_normalized, train_steps=10000, eval_steps=10)

# -------------------- Plot Performance --------------------
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label='DDPG Portfolio Value', marker='o')
plt.axhline(env.initial_investment, color='r', linestyle='--', label='Initial Investment')
plt.title("Portfolio Value Over Time (DDPG Evaluation)")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
