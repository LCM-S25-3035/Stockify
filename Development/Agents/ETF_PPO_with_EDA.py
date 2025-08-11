
# Install dependencies if not already installed
# !pip install ta stable-baselines3 gymnasium seaborn matplotlib

# Import libraries
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gymnasium import spaces
from ta.momentum import RSIIndicator
from ta.trend import MACD
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Load ETF dataset
file_path = "Cleaned_combined_global_ETFs.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])

# -----------------------------------------------
# 🧪 EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------

print("Shape of the dataset:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nNumber of unique ETFs (Tickers):", df['Ticker'].nunique())
print("Tickers:", df['Ticker'].unique())

print("\nDate range:", df['Date'].min(), "to", df['Date'].max())

# Plot number of records per ETF
plt.figure(figsize=(12, 4))
df['Ticker'].value_counts().plot(kind='bar', color='teal')
plt.title('Number of Records per ETF')
plt.xlabel('Ticker')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot close price trends
plt.figure(figsize=(14, 6))
for ticker in df['Ticker'].unique():
    subset = df[df['Ticker'] == ticker]
    plt.plot(subset['Date'], subset['Close'], label=ticker, alpha=0.7)
plt.title('ETF Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# Correlation Heatmap
pivot_close = df.pivot(index='Date', columns='Ticker', values='Close')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_close.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation between ETF Closing Prices")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# ♟️ CUSTOM GYM ENVIRONMENT FOR RL
# -----------------------------------------------

class HybridETFPortfolioEnv(gym.Env):
    def __init__(self, df, window_size=30, initial_cash=1e6):
        super(HybridETFPortfolioEnv, self).__init__()

        self.df = df.copy()
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.tickers = self.df['Ticker'].unique()
        self.n_assets = len(self.tickers)

        self._prepare_indicators()

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.n_assets * 3),
            dtype=np.float32
        )

        self.reset()

    def _prepare_indicators(self):
        self.df['RSI'] = self.df.groupby('Ticker')['Close'].transform(lambda x: RSIIndicator(x, window=14).rsi())
        self.df['MACD'] = self.df.groupby('Ticker')['Close'].transform(lambda x: MACD(x).macd_diff())
        self.df.dropna(inplace=True)

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.portfolio_value = self.initial_cash
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)
        self.history = []
        return self._get_observation(), {}

    def _get_observation(self):
        window_dates = sorted(self.df['Date'].unique())[self.current_step - self.window_size:self.current_step]
        window_df = self.df[self.df['Date'].isin(window_dates)]
        window = []

        for date in window_dates:
            row = window_df[window_df['Date'] == date].set_index('Ticker')
            obs_row = []
            for t in self.tickers:
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
        action = np.clip(action, 0, 1)
        action = action / np.sum(action)

        current_date = sorted(self.df['Date'].unique())[self.current_step]
        prev_date = sorted(self.df['Date'].unique())[self.current_step - 1]

        prices_today = self.df[self.df['Date'] == current_date].set_index('Ticker')['Close']
        prices_yesterday = self.df[self.df['Date'] == prev_date].set_index('Ticker')['Close']

        returns = (prices_today / prices_yesterday - 1).reindex(self.tickers).fillna(0).values
        portfolio_return = np.dot(action, returns)

        volatility_penalty = np.std(action * returns)
        div_penalty = np.sum((action - 1 / self.n_assets) ** 2)

        reward = portfolio_return - 0.5 * volatility_penalty - 0.1 * div_penalty
        self.portfolio_value *= (1 + portfolio_return)

        self.current_step += 1
        terminated = self.current_step >= len(self.df['Date'].unique()) - 1
        truncated = False

        self.history.append(self.portfolio_value)

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Portfolio Value: ${self.portfolio_value:,.2f}")

# -----------------------------------------------
# 🤖 PPO AGENT TRAINING AND EVALUATION
# -----------------------------------------------

env = HybridETFPortfolioEnv(df)
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

# -----------------------------------------------
# 📈 PERFORMANCE PLOTTING
# -----------------------------------------------

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
