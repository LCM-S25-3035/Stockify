import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# =======================
# Load and Prepare Data
# =======================
df = pd.read_csv("combined_prices.csv")
df['date'] = pd.to_datetime(df['date'])
tech_symbols = ['AAPL', 'MSFT', 'GOOGL','AMZN', 'TSLA', 'NVDA']
df = df[df['symbol'].isin(tech_symbols)]
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

df['daily_return'] = df.groupby('symbol')['close'].pct_change()
df['volatility_10d'] = df.groupby('symbol')['daily_return'].rolling(10).std().reset_index(0, drop=True)
df = df.dropna(subset=['daily_return', 'volatility_10d'])

features = ['close', 'daily_return', 'volatility_10d']
wide_df = df.pivot(index='date', columns='symbol', values=features)
wide_df.columns = [f"{feat}_{sym}" for feat, sym in wide_df.columns]
wide_df = wide_df.dropna().reset_index()

# =======================
# Environment Definition
# =======================
class MultiStockTradingEnvContinuous(gym.Env):
    def __init__(self, df, symbols, initial_cash=10000):
        super().__init__()
        self.df = df
        self.symbols = symbols
        self.num_stocks = len(symbols)
        self.initial_cash = initial_cash
        self.current_step = 0
        
        # Actions are unbounded; will be softmaxed into weights
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.num_stocks,), dtype=np.float32)
        
        # Observations: daily_return, volatility, portfolio allocation, cash ratio
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.num_stocks * 3 + 1,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = {sym: 0.0 for sym in self.symbols}
        self.total_asset = self.initial_cash
        self.prev_asset = self.initial_cash
        return self._next_observation()
    
    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = []
        for sym in self.symbols:
            obs.append(row[f'daily_return_{sym}'])
            obs.append(row[f'volatility_10d_{sym}'])
        
        # Portfolio allocation and cash ratio
        stock_values = [self.shares_held[sym] * row[f'close_{sym}'] for sym in self.symbols]
        total_stock_value = sum(stock_values)
        self.total_asset = self.cash + total_stock_value

        for val in stock_values:
            obs.append(val / self.total_asset)
        
        obs.append(self.cash / self.total_asset)
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        row = self.df.iloc[self.current_step]
        prices = {sym: row[f'close_{sym}'] for sym in self.symbols}

        # Convert actions to portfolio weights using softmax
        weights = np.exp(actions) / np.sum(np.exp(actions))
        portfolio_value = self.cash + sum(self.shares_held[sym] * prices[sym] for sym in self.symbols)

        # Rebalance portfolio based on weights
        for i, sym in enumerate(self.symbols):
            desired_value = portfolio_value * weights[i]
            current_value = self.shares_held[sym] * prices[sym]
            diff = desired_value - current_value
            shares_diff = diff // prices[sym]

            if shares_diff > 0:
                cost = shares_diff * prices[sym]
                if self.cash >= cost:
                    self.shares_held[sym] += shares_diff
                    self.cash -= cost
            elif shares_diff < 0:
                shares_to_sell = min(-shares_diff, self.shares_held[sym])
                self.shares_held[sym] -= shares_to_sell
                self.cash += shares_to_sell * prices[sym]
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Reward: relative return since last step
        self.total_asset = self.cash + sum(self.shares_held[sym] * prices[sym] for sym in self.symbols)
        reward = (self.total_asset - self.prev_asset) / self.prev_asset
        self.prev_asset = self.total_asset

        # Optionally add risk penalty
        # risk_penalty = np.std([row[f'daily_return_{sym}'] for sym in self.symbols])
        # reward -= 0.1 * risk_penalty
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}
    
    def render(self):
        print(f"\nStep {self.current_step} | Total Value: ${self.total_asset:.2f} | Cash: ${self.cash:.2f}")
        for sym in self.symbols:
            print(f"{sym}: {self.shares_held[sym]:.2f} shares")

# =======================
# Train DDPG Agent
# =======================
symbols = tech_symbols
env = MultiStockTradingEnvContinuous(wide_df, symbols)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000)
model.save("ddpg_multi_stock_trading")

# =======================
# Test Trained Agent
# =======================
env = MultiStockTradingEnvContinuous(wide_df, symbols)
obs = env.reset()

for _ in range(len(wide_df) - 1):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

print(f"\n✅ Final Portfolio Value: ${env.total_asset:.2f}")
