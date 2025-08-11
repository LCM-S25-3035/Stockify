#Importing libraries
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import joblib
import json
from gym.utils import seeding


#List of ETF tickers to include in portfolio
etf_tickers = ["SPY", "QQQ", "VTI", "DIA", "IWM", "EFA", "EEM", "VNQ", "LQD", "BND"]

#Data range
start_date = '2018-01-01'
end_date = '2025-05-20'

#Selected macroeconomic indicators for environment features
selected_macro_columns = [
    "VIX Market Volatility",
    "Federal Funds Rate",
    "10-Year Treasury Yield",
    "Unemployment Rate",
    "CPI All Items",
    "Recession Indicator"
]

#Function to calculate micro indicator
def compute_micro_indicators(df):
    """
    Computes multiple TA features for each asset:
    RSI, MACD, Bollinger Band width, Stochastic K, OBV, ADX,
    Parabolic SAR, Ichimoku diff, Williams %R, ATR.
    Returns concatenated feature array for all tickers.
    """
    features = []
    for col in df.columns:
        close = df[col]
        high = df[col]
        low = df[col]

        #Volume-based
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=df[col]).on_balance_volume().fillna(0).values
        #Trend indicators
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx().fillna(0).values
        psar = ta.trend.PSARIndicator(high=high, low=low, close=close).psar().bfill().values
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low)
        ichimoku_diff = ichimoku.ichimoku_conversion_line().bfill().values - ichimoku.ichimoku_base_line().bfill().values
        #Momentum indicators
        rsi = ta.momentum.RSIIndicator(close=close).rsi().fillna(50).values
        macd_diff = ta.trend.MACD(close=close).macd_diff().fillna(0).values
        williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r().fillna(-50).values
        #Volatility indicators
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().bfill().values
        bb_width = ta.volatility.BollingerBands(close=close).bollinger_wband().fillna(0).values
        #Oscillator
        stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch().fillna(0).values

        #Stacking all features for this asset
        asset_features = np.vstack([
            rsi, macd_diff, bb_width, stoch_k,
            obv, adx, psar, ichimoku_diff,
            williams_r, atr
        ]).T
        features.append(asset_features)

    #Combining features from all assets
    return np.hstack(features)

#Function to classify market regimes
def classify_regime(spy_norm):
    """
    Classify market regimes into 3 categories based on normalized SPY values:
    0 - Bullish, 1 - Neutral, 2 - Bearish
    """
    bullish_thresh = 0.33
    bearish_thresh = 0.66
    regimes = []
    for val in spy_norm:
        if val <= bullish_thresh:
            regimes.append(0)
        elif val <= bearish_thresh:
            regimes.append(1)
        else:
            regimes.append(2)
    return np.array(regimes)

#Custome gym environment for ETF portfolio allocation
class ETFPortfolioEnv(gym.Env):
    """
    Custom Gym environment to simulate ETF portfolio allocation using RL.
    State includes:
      - Normalized prices
      - Portfolio value
      - Risk appetite
      - Regime one-hot encoding
      - Technical (micro) indicators
      - Macroeconomic indicators
    Action: allocation weights for each ETF.
    Reward: adjusted portfolio return with risk and transaction cost penalties.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, prices, regime_class, micro_indicators, macro_indicators,
                 initial_amount=10000, risk_appetite=0.2, transaction_fee=0.001):
        super(ETFPortfolioEnv, self).__init__()

        #Core data
        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee

        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        self.current_step = 0
        self.current_holdings = np.zeros(self.n_assets)

        #Action space: continuous weights [0,1] for each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        #Observation space size
        obs_shape = (
            self.n_assets + 2 + self.num_regimes +
            micro_indicators.shape[1] +
            macro_indicators.shape[1]
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        """Reset portfolio to initial state."""
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.asset_weights = np.ones(self.n_assets) / self.n_assets
        self.current_holdings = (self.portfolio_value * self.asset_weights) / self.prices[self.current_step]
        return self._next_observation()

    def _next_observation(self):
        """Construct the observation vector for the current step."""
        #One-hot encode market regime
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1

        obs = np.concatenate([
            self.prices[self.current_step] / self.prices[0],  #Normalized prices
            [self.portfolio_value / self.initial_amount],    #Portfolio value ratio
            [self.risk_appetite],                            #Risk appetite
            regime_onehot,                                   #Market regime
            self.micro_indicators[self.current_step],        #Technical indicators
            self.macro_indicators[self.current_step]         #Macro indicators
        ])
        return obs

    def _calculate_transaction_cost(self, new_weights):
        """Calculate transaction costs when changing portfolio weights."""
        target_dollars = self.portfolio_value * new_weights
        current_dollars = self.current_holdings * self.prices[self.current_step]
        trades = target_dollars - current_dollars
        return np.sum(np.abs(trades)) * self.transaction_fee, target_dollars

    def step(self, action):
        """Execute one time step in the environment."""
        #Ensuring weights are valid
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)
        new_weights = action / (np.sum(action) + 1e-8)

        #Transaction cost and portfolio reallocation
        transaction_cost, target_dollars = self._calculate_transaction_cost(new_weights)
        self.portfolio_value -= transaction_cost
        new_weights = target_dollars / self.portfolio_value if self.portfolio_value > 0 else np.zeros(self.n_assets)
        self.current_holdings = (self.portfolio_value * new_weights) / self.prices[self.current_step]
        self.asset_weights = new_weights

        #Calculating returns
        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        returns = (next_prices - current_prices) / current_prices
        portfolio_return = np.dot(new_weights, returns)

        #Updating portfolio value
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

        #Reward = return - risk penalty - transaction cost penalty
        risk_penalty = np.std(returns)
        reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty - (transaction_cost / self.portfolio_value)

        self.current_step += 1
        self.portfolio_value = new_portfolio_value
        done = self.current_step >= len(self.prices) - 2

        return self._next_observation(), reward, done, {}

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_covariance_and_return(self, window_size=20):
        """Get mean returns and covariance matrix for recent window."""
        window_start = max(0, self.current_step - window_size)
        window_prices = self.prices[window_start:self.current_step + 1]
        returns = np.diff(window_prices, axis=0) / window_prices[:-1]
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))
        return mean_returns, cov_matrix

#Main execution block
if __name__ == "__main__":
    #Downloading historical ETF price data
    print("Downloading ETF data...")
    data = yf.download(etf_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    adj_close_data = pd.DataFrame()
    for ticker in etf_tickers:
        adj_close_data[ticker] = data[ticker]['Close']

    #Market regime classification using SPY normalized prices
    print("Using SPY for volatility regime estimation...")
    spy = adj_close_data["SPY"]
    scaler = MinMaxScaler()
    spy_norm = scaler.fit_transform(spy.values.reshape(-1, 1)).flatten()
    regime_classes = classify_regime(spy_norm)

    #Computing technical (micro) indicators
    print("Computing indicators...")
    micro_indicators = compute_micro_indicators(adj_close_data)
    indicator_scaler = MinMaxScaler().fit(micro_indicators)
    micro_indicators = indicator_scaler.transform(micro_indicators)
    micro_indicators = np.nan_to_num(micro_indicators)

    #Loading and process macroeconomic data
    print("Processing macroeconomic data...")
    macro_df = pd.read_csv('macroeconomic_data_2010_2024.csv', parse_dates=['Date'])
    macro_df.set_index('Date', inplace=True)
    macro_df = macro_df.reindex(adj_close_data.index, method='ffill')  # Align with price dates
    macro_df = macro_df[selected_macro_columns]

    macro_scaler = MinMaxScaler().fit(macro_df)
    macro_indicators = macro_scaler.transform(macro_df)

    #Preparing price array
    adj_close_data = adj_close_data.replace([np.inf, -np.inf], np.nan).dropna()
    prices = adj_close_data.values

    #Saving preprocessed data for later use
    print("Saving preprocessed arrays for meta agent...")
    np.save("prices_etf.npy", prices)
    np.save("regime_etf.npy", regime_classes)
    np.save("micro_indicators_etf.npy", micro_indicators)
    np.save("macro_indicators_etf.npy", macro_indicators)

    #Creating and seeding environment
    print("Creating environment...")
    env = ETFPortfolioEnv(prices, regime_classes, micro_indicators, macro_indicators, 10000, 0.5, 0.001)
    env.seed(42)

    #Training PPO model
    print("Training PPO model...")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01)
    model.learn(total_timesteps=500000, progress_bar=True)

    #Calculing mean returns and covariance matrix
    mu, cov = env.get_covariance_and_return()
    print("Mean expected returns:", mu)
    print("Covariance matrix shape:", cov.shape)

    #Saving model and scalers
    print("Saving model and scalers...")
    model.save("ppo_etf_model")
    joblib.dump(scaler, "spy_scaler.pkl")
    joblib.dump(indicator_scaler, "indicator_scaler_etf.pkl")
    joblib.dump(macro_scaler, "macro_scaler_etf.pkl")
    with open("etf_tickers.json", "w") as f:
        json.dump(etf_tickers, f)
