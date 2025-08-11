import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import ta

# =======================
# Load and Prepare Price Data
# =======================

df = pd.read_csv("combined_prices.csv")
df['date'] = pd.to_datetime(df['date'])
tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
df = df[df['symbol'].isin(tech_symbols)]
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

df['daily_return'] = df.groupby('symbol')['close'].pct_change()
df['volatility_10d'] = df.groupby('symbol')['daily_return'].rolling(10).std().reset_index(0, drop=True)
df = df.dropna(subset=['daily_return', 'volatility_10d'])

features = ['close', 'daily_return', 'volatility_10d']
wide_df = df.pivot(index='date', columns='symbol', values=features)
wide_df.columns = [f"{feat}_{sym}" for feat, sym in wide_df.columns]
wide_df = wide_df.dropna().reset_index()  # Keep 'date' as a column for alignment

# =======================
# Compute Micro Technical Indicators
# =======================

def compute_micro_indicators(df, symbols):
    features = []
    for sym in symbols:
        close = df[f'close_{sym}']
        high = close  # simplified proxy
        low = close   # simplified proxy
        volume = pd.Series(np.ones(len(df)))  # placeholder if no volume

        rsi = ta.momentum.RSIIndicator(close).rsi().fillna(50)
        macd_diff = ta.trend.MACD(close).macd_diff().fillna(0)
        bb_width = ta.volatility.BollingerBands(close).bollinger_wband().fillna(0)
        stoch_k = ta.momentum.StochasticOscillator(high, low, close).stoch().fillna(0)
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().fillna(0)

        asset_features = pd.concat([rsi, macd_diff, bb_width, stoch_k, obv], axis=1)
        asset_features.columns = [f"{sym}_rsi", f"{sym}_macd_diff", f"{sym}_bb_width", f"{sym}_stoch_k", f"{sym}_obv"]
        features.append(asset_features)
    micro_df = pd.concat(features, axis=1).reset_index(drop=True)
    return micro_df.values

micro_indicators = compute_micro_indicators(wide_df, tech_symbols)
micro_scaler = MinMaxScaler().fit(micro_indicators)
micro_indicators_scaled = micro_scaler.transform(micro_indicators)

# =======================
# Load and Prepare Macro Data
# =======================

macro_df = pd.read_csv("macro_data.csv", parse_dates=['DATE'])
macro_df.set_index('DATE', inplace=True)
macro_df.fillna(method='ffill', inplace=True)

selected_macro_columns = [
    "VIX Market Volatility",
    "Federal Funds Rate",
    "10-Year Treasury Yield",
    "Unemployment Rate",
    "CPI All Items",
    "Recession Indicator"
]
macro_df = macro_df[selected_macro_columns]

# Align macro data dates with price dates (wide_df['date'])
macro_df = macro_df.reindex(pd.to_datetime(wide_df['date']), method='ffill')

macro_scaler = MinMaxScaler()
macro_indicators = macro_scaler.fit_transform(macro_df)

# =======================
# Download VIX for Regime Classification
# =======================

start_date = wide_df['date'].min().strftime('%Y-%m-%d')
end_date = wide_df['date'].max().strftime('%Y-%m-%d')

vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
vix = vix.reindex(pd.to_datetime(wide_df['date']), method='ffill').dropna()

vix_scaler = MinMaxScaler()
vix_norm = vix_scaler.fit_transform(vix.values.reshape(-1,1)).flatten()

def classify_regime(vix_norm):
    bullish_thresh = 0.33
    bearish_thresh = 0.66
    regimes = []
    for val in vix_norm:
        if val <= bullish_thresh:
            regimes.append(0)
        elif val <= bearish_thresh:
            regimes.append(1)
        else:
            regimes.append(2)
    return np.array(regimes)

regime_classes = classify_regime(vix_norm)

# =======================
# Align lengths of all data arrays
# =======================

min_len = min(len(wide_df), len(micro_indicators_scaled), len(macro_indicators), len(regime_classes))
wide_df = wide_df.iloc[:min_len].reset_index(drop=True)
micro_indicators_scaled = micro_indicators_scaled[:min_len]
macro_indicators = macro_indicators[:min_len]
regime_classes = regime_classes[:min_len]

# =======================
# Define Custom Trading Environment
# =======================

class MultiStockTradingEnvContinuous(gym.Env):
    def __init__(self, df, micro_indicators, macro_indicators, regimes, symbols, initial_cash=10000, transaction_fee=0.001, risk_penalty_coef=0.1):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.regimes = regimes
        self.symbols = symbols
        self.num_stocks = len(symbols)
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.risk_penalty_coef = risk_penalty_coef
        
        self.num_regimes = 3
        
        self.current_step = 0
        self.cash = initial_cash
        self.shares_held = {sym: 0.0 for sym in self.symbols}
        self.total_asset = initial_cash
        self.prev_asset = initial_cash
        
        obs_dim = self.num_stocks * (2 + 5) + self.num_stocks + 1 + self.num_regimes + self.macro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.num_stocks,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = {sym: 0.0 for sym in self.symbols}
        self.total_asset = self.initial_cash
        self.prev_asset = self.initial_cash
        return self._next_observation()
    
    def _next_observation(self):
        obs = []
        row = self.df.iloc[self.current_step]
        
        # Base features: daily_return, volatility_10d
        for sym in self.symbols:
            obs.append(row[f'daily_return_{sym}'])
            obs.append(row[f'volatility_10d_{sym}'])
        
        # Micro indicators
        obs.extend(self.micro_indicators[self.current_step])
        
        # Portfolio allocation (shares value / total asset)
        prices = {sym: row[f'close_{sym}'] for sym in self.symbols}
        stock_values = [self.shares_held[sym] * prices[sym] for sym in self.symbols]
        total_stock_value = sum(stock_values)
        self.total_asset = self.cash + total_stock_value
        
        for val in stock_values:
            obs.append(val / self.total_asset if self.total_asset > 0 else 0)
        
        obs.append(self.cash / self.total_asset if self.total_asset > 0 else 0)
        
        # Market regime one-hot
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regimes[self.current_step]] = 1
        obs.extend(regime_onehot)
        
        # Macro indicators
        obs.extend(self.macro_indicators[self.current_step])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        row = self.df.iloc[self.current_step]
        prices = {sym: row[f'close_{sym}'] for sym in self.symbols}
        
        weights = np.exp(actions) / np.sum(np.exp(actions))
        
        portfolio_value = self.cash + sum(self.shares_held[sym] * prices[sym] for sym in self.symbols)
        
        transaction_cost = 0
        for i, sym in enumerate(self.symbols):
            desired_value = portfolio_value * weights[i]
            current_value = self.shares_held[sym] * prices[sym]
            trade_value = abs(desired_value - current_value)
            transaction_cost += trade_value * self.transaction_fee
        
        self.cash -= transaction_cost
        
        for i, sym in enumerate(self.symbols):
            desired_value = portfolio_value * weights[i]
            current_value = self.shares_held[sym] * prices[sym]
            diff_value = desired_value - current_value
            shares_diff = int(diff_value // prices[sym])
            
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
        
        new_total_asset = self.cash + sum(self.shares_held[sym] * prices[sym] for sym in self.symbols)
        
        portfolio_return = (new_total_asset - self.prev_asset) / self.prev_asset
        
        returns = np.array([row[f'daily_return_{sym}'] for sym in self.symbols])
        risk_penalty = np.std(returns)
        
        reward = portfolio_return - self.risk_penalty_coef * risk_penalty - (transaction_cost / new_total_asset if new_total_asset > 0 else 0)
        
        self.prev_asset = new_total_asset
        self.total_asset = new_total_asset
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}
    
    def render(self):
        print(f"\nStep {self.current_step} | Total Portfolio Value: ${self.total_asset:.2f} | Cash: ${self.cash:.2f}")
        for sym in self.symbols:
            print(f"{sym}: {self.shares_held[sym]:.2f} shares")

# =======================
# Train the DDPG Agent
# =======================

env = MultiStockTradingEnvContinuous(wide_df, micro_indicators_scaled, macro_indicators, regime_classes, tech_symbols)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000)
model.save("ddpg_multi_stock_trading_with_macro")

# =======================
# Test the trained agent
# =======================

env = MultiStockTradingEnvContinuous(wide_df, micro_indicators_scaled, macro_indicators, regime_classes, tech_symbols)
obs = env.reset()

for _ in range(len(wide_df) - 1):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break

print(f"\n✅ Final Portfolio Value: ${env.total_asset:.2f}")
