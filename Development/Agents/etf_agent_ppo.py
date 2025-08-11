import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Download ETF data
df = yf.download('SPY', start='2015-01-01', end='2022-01-01')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# Add RSI and MACD indicators
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(df['Close'])
df['MACD'] = macd.macd_diff()
df.dropna(inplace=True)

# Custom trading environment
class ETFTradingEnv(gym.Env):
    def __init__(self, df):
        super(ETFTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = 10000
        self.shares_held = 0
        self.portfolio_value = 10000
        self.current_step = 0
        self.history = []
        return self._next_observation()

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['RSI'], row['MACD']], dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        if action == 1:  # Buy
            shares = self.balance // current_price
            self.balance -= shares * current_price
            self.shares_held += shares
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        current_value = self.balance + self.shares_held * current_price
        reward = current_value - self.portfolio_value
        self.portfolio_value = current_value
        self.history.append(current_value)

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Value: {self.portfolio_value:.2f}")

# Train PPO agent
env = ETFTradingEnv(df)
check_env(env)
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000)

# Evaluate agent
obs = env.reset()
for _ in range(env.n_steps - 1):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break

# Backtest result
plt.figure(figsize=(12, 6))
plt.plot(env.history, label='Portfolio Value')
plt.axhline(10000, color='r', linestyle='--', label='Initial Balance')
plt.title('ETF Trading with PPO Agent (SPY)')
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
