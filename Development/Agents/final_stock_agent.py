#Importing necessary libraries
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


#List of stock tickers to analyze in the portfolio
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "UNH"
]

#Defining the start and end dates for historical data download
start_date = '2015-01-01'
end_date = '2025-05-20'


# Select only the top macro features for PPO
selected_macro_columns = [
    "VIX Market Volatility",
    "Federal Funds Rate",
    "10-Year Treasury Yield",
    "Unemployment Rate",
    "CPI All Items",
    "Recession Indicator"
]


#Function to classify market regime based on normalized VIX values
def classify_regime(vix_norm):
    bullish_thresh = 0.33    #Threshold below which market is bullish
    bearish_thresh = 0.66    #Threshold above which market is bearish
    regimes_class = []
    for val in vix_norm:
        if val <= bullish_thresh:
            regimes_class.append(0)  #Bullish regime
        elif val <= bearish_thresh:
            regimes_class.append(1)  #Sideways/Neutral regime
        else:
            regimes_class.append(2)  #Bearish regime
    return np.array(regimes_class)

#Function to compute a set of technical indicators for each asset
def compute_micro_indicators(df):
    features = []
    for col in df.columns:
        #Using the same column series for close, high, and low (simplification)
        close = df[col]
        high = df[col]
        low = df[col]

        #Calculating On Balance Volume (OBV)
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=df[col]).on_balance_volume().fillna(0).values
        #Calculating Average Directional Index (ADX)
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx().fillna(0).values
        #Calculating Parabolic SAR with backfill for missing data
        psar = ta.trend.PSARIndicator(high=high, low=low, close=close).psar().bfill().values
        #Calculating Ichimoku indicator difference (conversion line - base line)
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low)
        ichimoku_diff = ichimoku.ichimoku_conversion_line().bfill().values - ichimoku.ichimoku_base_line().bfill().values
        #Relative Strength Index (RSI), filled with 50 for missing data
        rsi = ta.momentum.RSIIndicator(close=close).rsi().fillna(50).values
        #MACD difference
        macd_diff = ta.trend.MACD(close=close).macd_diff().fillna(0).values
        #Williams %R indicator
        williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r().fillna(-50).values
        #Average True Range (ATR) with backfill for missing data
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().bfill().values
        #Bollinger Bands width
        bb_width = ta.volatility.BollingerBands(close=close).bollinger_wband().fillna(0).values
        #Stochastic Oscillator %K
        stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch().fillna(0).values

        #Stacking all features vertically and transpose to shape (time_steps, features)
        asset_features = np.vstack([
            rsi, macd_diff, bb_width, stoch_k,
            obv,
            adx, psar, ichimoku_diff,
            williams_r,
            atr
        ]).T
        features.append(asset_features)

    #Horizontally stacking features of all assets to get final feature matrix for all assets
    return np.hstack(features)

#Custom OpenAI Gym environment to simulate stock portfolio trading
class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, regime_class, micro_indicators, macro_indicators, initial_amount=10000, risk_appetite=0.2, transaction_fee=0.001):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices                      #Historical prices of all assets
        self.regime_class = regime_class          #Market regime classification for each time step
        self.micro_indicators = micro_indicators  #Technical indicators for each time step
        self.macro_indicators = macro_indicators  #MACRO: macroeconomic features
        self.initial_amount = initial_amount      #Initial investment capital
        self.risk_appetite = risk_appetite        #Controls trade-off between return and risk
        self.transaction_fee = transaction_fee    #Proportional transaction fee per trade

        self.n_assets = prices.shape[1]           #Number of assets in the portfolio
        self.num_regimes = 3                      #Number of market regimes (bullish, sideways, bearish)
        self.current_step = 0                      #Current time step in environment
        self.current_holdings = np.zeros(self.n_assets)  #Number of shares currently held

        #Action space: portfolio weights for each asset (continuous between 0 and 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        #Observation space: normalized prices + portfolio value + risk appetite + regime one-hot + indicators
        obs_shape = (
            self.n_assets + 2 + self.num_regimes +
            micro_indicators.shape[1] +
            macro_indicators.shape[1]  #MACRO: adjust obs space
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        #Initially, invest equally across all assets
        self.asset_weights = np.ones(self.n_assets) / self.n_assets
        #Calculate initial holdings based on prices and weights
        self.current_holdings = (self.portfolio_value * self.asset_weights) / self.prices[self.current_step]
        return self._next_observation()

    def _next_observation(self):
        #Creating one‑hot encoding of current market regime
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1

        #Observation vector concatenates prices, portfolio info, regime, and indicators
        obs = np.concatenate([
            self.prices[self.current_step] / self.prices[0],       #Normalized prices
            [self.portfolio_value / self.initial_amount],          #Normalized portfolio value
            [self.risk_appetite],                                  #Risk appetite (constant)
            regime_onehot,                                         #One‑hot regime
            self.micro_indicators[self.current_step],              #Technical indicators
            self.macro_indicators[self.current_step]               #MACRO: Macro indicators
        ])
        return obs

    def _calculate_transaction_cost(self, new_weights):
        target_dollars = self.portfolio_value * new_weights
        current_dollars = self.current_holdings * self.prices[self.current_step]
        trades = target_dollars - current_dollars
        transaction_cost = np.sum(np.abs(trades)) * self.transaction_fee
        return transaction_cost, target_dollars

    def step(self, action):
        action = np.clip(action, 0, 1)
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

    def render(self, mode='human', close=False):
        current_prices = self.prices[self.current_step]
        holdings_value = self.current_holdings * current_prices
        total_value = np.sum(holdings_value)

        print(f"\nStep: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        for i, ticker in enumerate(tickers):
            print(f"{ticker}: {self.current_holdings[i]:.2f} shares (${holdings_value[i]:.2f}, {100*holdings_value[i]/total_value:.1f}%)")

        mu, cov = self.get_covariance_and_return()
        print(f"Expected returns: {np.round(mu, 4)}")
        print(f"Portfolio variance (mean diag): {np.round(np.mean(np.diag(cov)), 6)}")


    #Covariance matrix and expected returns calculation
    def get_covariance_and_return(self, window_size=20):
        
        if self.current_step < window_size:
            window_start = 0
        else:
            window_start = self.current_step - window_size

        window_prices = self.prices[window_start:self.current_step + 1]

        # Ensure we have enough data
        if len(window_prices) < 2:
            returns = np.zeros((1, self.n_assets))
        else:
            returns = np.diff(window_prices, axis=0) / window_prices[:-1]

        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))

        return mean_returns, cov_matrix


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


#Main Execution block
if __name__ == "__main__":
    print("Downloading stock data...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    adj_close_data = pd.DataFrame()
    for ticker in tickers:
        adj_close_data[ticker] = data[ticker]['Close']


    print("Downloading VIX data...")
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    adj_close_data = adj_close_data.loc[vix.index].dropna()
    vix = vix.loc[adj_close_data.index]

    scaler = MinMaxScaler()
    vix_norm = scaler.fit_transform(vix.values.reshape(-1, 1)).flatten()
    regime_classes = classify_regime(vix_norm)

    print("Computing technical indicators...")
    micro_indicators = compute_micro_indicators(adj_close_data)
    indicator_scaler = MinMaxScaler().fit(micro_indicators)
    micro_indicators = indicator_scaler.transform(micro_indicators)


    #MACRO: Loading and aligning macroeconomic data
    print("Processing macroeconomic data...")
    macro_df = pd.read_csv('macroeconomic_data_2010_2024.csv', parse_dates=['Date'])
    macro_df.set_index('Date', inplace=True)
    macro_df = macro_df.reindex(adj_close_data.index, method='ffill')

    #Select only the specified macro features
    macro_df = macro_df[selected_macro_columns]

    #Scale and transform
    macro_scaler = MinMaxScaler().fit(macro_df)
    macro_indicators = macro_scaler.transform(macro_df)

    prices = adj_close_data.values

    print("Saving preprocessed arrays for meta agent...")
    np.save("prices_stock.npy", prices)
    np.save("regime_stock.npy", regime_classes)
    np.save("micro_indicators_stock.npy", micro_indicators)
    np.save("macro_indicators_stock.npy", macro_indicators)

    print("Creating environment...")

    print("Creating environment...")
    env = StockPortfolioEnv(prices, regime_classes, micro_indicators, macro_indicators, 10000, 0.5, 0.001)
    env.seed(42)

    print("Training model...")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01)
    
    model.learn(total_timesteps=1500000, progress_bar=True)

    # After training:
    mu, cov = env.get_covariance_and_return()
    print("Mean expected returns:", mu)
    print("Covariance matrix shape:", cov.shape)


    print("Saving model and scalers...")
    model.save("ppo_stock_model")
    joblib.dump(scaler, "vix_scaler.pkl")
    joblib.dump(indicator_scaler, "indicator_scaler.pkl")
    joblib.dump(macro_scaler, "macro_scaler.pkl")  # MACRO: save macro scaler
    with open("tickers.json", "w") as f:
        json.dump(tickers, f)
