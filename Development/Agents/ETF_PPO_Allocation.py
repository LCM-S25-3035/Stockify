#pip install ta
#pip install stable-baselines3
#pip install gymnasium

# Import necessary libraries
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from ta.momentum import RSIIndicator
from ta.trend import MACD
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Load ETF dataset
file_path = "Cleaned_combined_global_ETFs.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])


# Define a custom multi-ETF trading environment with technical indicators
class HybridETFPortfolioEnv(gym.Env):
    def __init__(self, df, window_size=30, initial_cash=1e6):
        super(HybridETFPortfolioEnv, self).__init__()

        self.df = df.copy()
        self.window_size = window_size  # Number of past days used in observation
        self.initial_cash = initial_cash  # Starting money
        self.tickers = self.df['Ticker'].unique()  # Unique ETF tickers
        self.n_assets = len(self.tickers)  # Number of ETFs

        self._prepare_indicators()  # Add RSI and MACD to the dataset

        # Action space: agent provides portfolio weights for each ETF (must sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Observation space: sequence of [Close, RSI, MACD] for each ETF over the window
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.n_assets * 3),
            dtype=np.float32
        )

        self.reset()

    def _prepare_indicators(self):
        # Add RSI and MACD for each ETF using TA-Lib
        self.df['RSI'] = self.df.groupby('Ticker')['Close'].transform(lambda x: RSIIndicator(x, window=14).rsi())
        self.df['MACD'] = self.df.groupby('Ticker')['Close'].transform(lambda x: MACD(x).macd_diff())
        self.df.dropna(inplace=True)  # Drop any rows with missing indicator values

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        self.current_step = self.window_size  # Skip initial window
        self.portfolio_value = self.initial_cash  # Reset to starting cash
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)  # Equal allocation
        self.history = []  # Store portfolio value over time
        return self._get_observation(), {}  # Return observation and info dict

    def _get_observation(self):
        # Collect data for the past `window_size` days
        window_dates = sorted(self.df['Date'].unique())[self.current_step - self.window_size:self.current_step]
        window_df = self.df[self.df['Date'].isin(window_dates)]

        window = []
        for date in window_dates:
            row = window_df[window_df['Date'] == date].set_index('Ticker')
            obs_row = []
            for t in self.tickers:
                # Fill missing tickers with zeroes
                if t in row.index:
                    obs_row.extend([
                        row.loc[t, 'Close'],
                        row.loc[t, 'RSI'],
                        row.loc[t, 'MACD']
                    ])
                else:
                    obs_row.extend([0, 0, 0])
            window.append(obs_row)

        return np.array(window, dtype=np.float32)

    def step(self, action):
        # Normalize weights so they sum to 1
        action = np.clip(action, 0, 1)
        action = action / np.sum(action)

        # Get current and previous dates for price comparison
        current_date = sorted(self.df['Date'].unique())[self.current_step]
        prev_date = sorted(self.df['Date'].unique())[self.current_step - 1]

        # Get today's and yesterday's close prices
        prices_today = self.df[self.df['Date'] == current_date].set_index('Ticker')['Close']
        prices_yesterday = self.df[self.df['Date'] == prev_date].set_index('Ticker')['Close']

        # Calculate daily returns
        returns = (prices_today / prices_yesterday - 1).reindex(self.tickers).fillna(0).values

        # Calculate weighted portfolio return
        portfolio_return = np.dot(action, returns)

        # Reward shaping: penalize volatility and concentration
        volatility_penalty = np.std(action * returns)
        div_penalty = np.sum((action - 1 / self.n_assets) ** 2)

        # Final reward = return - penalties
        reward = portfolio_return - 0.5 * volatility_penalty - 0.1 * div_penalty

        # Update total portfolio value
        self.portfolio_value *= (1 + portfolio_return)

        # Move to next time step
        self.current_step += 1

        # End episode if last date reached
        terminated = self.current_step >= len(self.df['Date'].unique()) - 1
        truncated = False

        # Log current value
        self.history.append(self.portfolio_value)

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        # Display current portfolio value
        print(f"Step: {self.current_step}, Portfolio Value: ${self.portfolio_value:,.2f}")

# Initialize and train PPO agent
env = HybridETFPortfolioEnv(df)
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate the trained agent
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(env.history, label="Portfolio Value")
plt.axhline(y=env.initial_cash, color='r', linestyle='--', label="Initial Cash")
plt.title("Hybrid ETF Portfolio Value Over Time (PPO Agent)")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()