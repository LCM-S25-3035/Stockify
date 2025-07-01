import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Load and Prepare Multi-Stock Data 
df = pd.read_csv("combined_prices.csv")
df['date'] = pd.to_datetime(df['date'])
tech_symbols = ['AAPL', 'MSFT', 'GOOGL']
df = df[df['symbol'].isin(tech_symbols)]
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

df['daily_return'] = df.groupby('symbol')['close'].pct_change()
df['volatility_10d'] = df.groupby('symbol')['daily_return'].rolling(10).std().reset_index(0, drop=True)
df = df.dropna(subset=['daily_return', 'volatility_10d'])

features = ['close', 'daily_return', 'volatility_10d']
wide_df = df.pivot(index='date', columns='symbol', values=features)
wide_df.columns = [f"{feat}_{sym}" for feat, sym in wide_df.columns]
wide_df = wide_df.dropna().reset_index()

#Environment for Continuous Actions 
class MultiStockTradingEnvContinuous(gym.Env):
    def __init__(self, df, symbols, initial_cash=10000):
        super().__init__()
        self.df = df
        self.symbols = symbols
        self.num_stocks = len(symbols)
        self.current_step = 0
        self.initial_cash = initial_cash
        
        # Continuous actions between -1 and 1 per stock
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)
        
        # Observations: daily_return and volatility for each stock
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks * 2,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = {sym: 0.0 for sym in self.symbols}
        self.total_asset = self.cash
        return self._next_observation()
    
    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = []
        for sym in self.symbols:
            obs.append(row[f'daily_return_{sym}'])
            obs.append(row[f'volatility_10d_{sym}'])
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        row = self.df.iloc[self.current_step]
        prices = {sym: row[f'close_{sym}'] for sym in self.symbols}
        
        # Actions are continuous in [-1, 1], scale buy/sell amount
        for i, sym in enumerate(self.symbols):
            action = float(actions[i])
            price = prices[sym]
            max_shares_to_buy = self.cash // price
            # Buy action
            if action > 0:
                shares_to_buy = min(action * 10, max_shares_to_buy)  # scale buy volume
                self.shares_held[sym] += shares_to_buy
                self.cash -= shares_to_buy * price
            # Sell action
            elif action < 0:
                shares_to_sell = min(-action * 10, self.shares_held[sym])  # scale sell volume
                self.shares_held[sym] -= shares_to_sell
                self.cash += shares_to_sell * price
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        portfolio = sum(self.shares_held[sym] * prices[sym] for sym in self.symbols)
        self.total_asset = self.cash + portfolio
        reward = self.total_asset - self.initial_cash
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}
    
    def render(self):
        print(f"\nStep {self.current_step} | Total Value: ${self.total_asset:.2f} | Cash: ${self.cash:.2f}")
        for sym in self.symbols:
            print(f"{sym}: {self.shares_held[sym]:.2f} shares")

#Create env and train DDPG agent 
symbols = tech_symbols
env = MultiStockTradingEnvContinuous(wide_df, symbols)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000)
model.save("ddpg_multi_stock_trading")

# Test trained agent 
env = MultiStockTradingEnvContinuous(wide_df, symbols)
obs = env.reset()

for _ in range(len(wide_df) - 1):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

print(f"\n✅ Final Portfolio Value: ${env.total_asset:.2f}")
