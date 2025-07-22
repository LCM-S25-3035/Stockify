import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import joblib
import json
from gym.utils import seeding

# List of crypto tickers
crypto_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD", "LTC-USD", "DOT-USD", "AVAX-USD"]
start_date = '2018-01-01'
end_date = '2025-05-20'

# Compute micro indicators (same as stock logic)
def compute_micro_indicators(df):
    features = []
    for col in df.columns:
        close = df[col]
        high = df[col]
        low = df[col]

        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=df[col]).on_balance_volume().fillna(0).values
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx().fillna(0).values
        psar = ta.trend.PSARIndicator(high=high, low=low, close=close).psar().bfill().values
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low)
        ichimoku_diff = ichimoku.ichimoku_conversion_line().bfill().values - ichimoku.ichimoku_base_line().bfill().values
        rsi = ta.momentum.RSIIndicator(close=close).rsi().fillna(50).values
        macd_diff = ta.trend.MACD(close=close).macd_diff().fillna(0).values
        williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r().fillna(-50).values
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().bfill().values
        bb_width = ta.volatility.BollingerBands(close=close).bollinger_wband().fillna(0).values
        stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch().fillna(0).values

        asset_features = np.vstack([rsi, macd_diff, bb_width, stoch_k, obv, adx, psar, ichimoku_diff, williams_r, atr]).T
        features.append(asset_features)
    return np.hstack(features)

# Simplified "regime": based on BTC normalized volatility (can replace later)
def classify_regime(btc_norm):
    bullish_thresh = 0.33
    bearish_thresh = 0.66
    regimes = []
    for val in btc_norm:
        if val <= bullish_thresh:
            regimes.append(0)
        elif val <= bearish_thresh:
            regimes.append(1)
        else:
            regimes.append(2)
    return np.array(regimes)

class CryptoPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, prices, regime_class, micro_indicators, initial_amount=10000, risk_appetite=0.2, transaction_fee=0.001):
        super(CryptoPortfolioEnv, self).__init__()
        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee

        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        self.current_step = 0
        self.current_holdings = np.zeros(self.n_assets)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.asset_weights = np.ones(self.n_assets) / self.n_assets
        self.current_holdings = (self.portfolio_value * self.asset_weights) / self.prices[self.current_step]
        return self._next_observation()

    def _next_observation(self):
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1

        obs = np.concatenate([
            self.prices[self.current_step] / self.prices[0],
            [self.portfolio_value / self.initial_amount],
            [self.risk_appetite],
            regime_onehot,
            self.micro_indicators[self.current_step]
        ])
        return obs

    def _calculate_transaction_cost(self, new_weights):
        target_dollars = self.portfolio_value * new_weights
        current_dollars = self.current_holdings * self.prices[self.current_step]
        trades = target_dollars - current_dollars
        return np.sum(np.abs(trades)) * self.transaction_fee, target_dollars

    def step(self, action):
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)
        new_weights = action / (np.sum(action) + 1e-8)
        transaction_cost, target_dollars = self._calculate_transaction_cost(new_weights)
        self.portfolio_value -= transaction_cost
        new_weights = target_dollars / self.portfolio_value if self.portfolio_value > 0 else np.zeros(self.n_assets)
        self.current_holdings = (self.portfolio_value * new_weights) / self.prices[self.current_step]
        self.asset_weights = new_weights

        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        returns = (next_prices - current_prices) / current_prices

        portfolio_return = np.dot(new_weights, returns)
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        risk_penalty = np.std(returns)
        reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty - (transaction_cost / self.portfolio_value)

        self.current_step += 1
        self.portfolio_value = new_portfolio_value
        done = self.current_step >= len(self.prices) - 2
        return self._next_observation(), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_covariance_and_return(self, window_size=20):
        if self.current_step < window_size:
            window_start = 0
        else:
            window_start = self.current_step - window_size

        window_prices = self.prices[window_start:self.current_step + 1]
        returns = np.diff(window_prices, axis=0) / window_prices[:-1]
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))
        return mean_returns, cov_matrix


# === MAIN ===
if __name__ == "__main__":
    print("Downloading crypto data...")
    data = yf.download(crypto_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    adj_close_data = pd.DataFrame()
    for ticker in crypto_tickers:
        adj_close_data[ticker] = data[ticker]['Close']
    
    print("Using BTC for volatility regime estimation...")
    btc = adj_close_data["BTC-USD"]
    scaler = MinMaxScaler()
    btc_norm = scaler.fit_transform(btc.values.reshape(-1, 1)).flatten()
    regime_classes = classify_regime(btc_norm)

    print("Computing indicators...")
    micro_indicators = compute_micro_indicators(adj_close_data)
    indicator_scaler = MinMaxScaler().fit(micro_indicators)
    micro_indicators = indicator_scaler.transform(micro_indicators)

    # Replace NaNs and infs in micro indicators
    micro_indicators = np.nan_to_num(micro_indicators)

    # Clean price data
    adj_close_data = adj_close_data.replace([np.inf, -np.inf], np.nan).dropna()
    prices = adj_close_data.values

    #prices = adj_close_data.values

    print("Creating environment...")
    env = CryptoPortfolioEnv(prices, regime_classes, micro_indicators, 10000, 0.5, 0.001)
    env.seed(42)


    print("Training PPO model...")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01)
    
    model.learn(total_timesteps=1500000, progress_bar=True)

    mu, cov = env.get_covariance_and_return()
    print("Mean expected returns:", mu)
    print("Covariance matrix shape:", cov.shape)

    print("Saving model and scalers...")
    model.save("ppo_crypto_model")
    joblib.dump(scaler, "btc_scaler.pkl")
    joblib.dump(indicator_scaler, "indicator_scaler_crypto.pkl")
    with open("crypto_tickers.json", "w") as f:
        json.dump(crypto_tickers, f)
