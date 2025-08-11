#Importing necessary libraries
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

#List of stock tickers to analyze in the portfolio
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "UNH"
]

#Defining the start and end dates for historical data download
start_date = '2015-01-01'
end_date = '2025-05-20'

#Selecting only the top macro features for PPO
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
        close = df[col]
        high = df[col]
        low = df[col]

        #Volume & Trend Indicators
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=df[col]).on_balance_volume().fillna(0).values
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx().fillna(0).values
        psar = ta.trend.PSARIndicator(high=high, low=low, close=close).psar().bfill().values
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low)
        ichimoku_diff = ichimoku.ichimoku_conversion_line().bfill().values - ichimoku.ichimoku_base_line().bfill().values
        
        #Momentum Indicators
        rsi = ta.momentum.RSIIndicator(close=close).rsi().fillna(50).values
        macd_diff = ta.trend.MACD(close=close).macd_diff().fillna(0).values
        williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r().fillna(-50).values
        
        #Volatility Indicators
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().bfill().values
        bb_width = ta.volatility.BollingerBands(close=close).bollinger_wband().fillna(0).values
        
        #Oscillators
        stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch().fillna(0).values

        #Combining all features into a single array for the asset
        asset_features = np.vstack([
            rsi, macd_diff, bb_width, stoch_k,
            obv,
            adx, psar, ichimoku_diff,
            williams_r,
            atr
        ]).T
        features.append(asset_features)

    return np.hstack(features)

#Custom OpenAI Gym environment to simulate stock portfolio trading
class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, regime_class, micro_indicators, macro_indicators,
                 initial_amount=10000, risk_appetite=0.2, transaction_fee=0.001,
                 window_size=60, turnover_penalty=1e-3):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.turnover_penalty = turnover_penalty

        #Number of assets in the portfolio
        self.n_assets = prices.shape[1]
        #Number of regimes
        self.num_regimes = 3
        self.current_step = 0
        self.current_holdings = np.zeros(self.n_assets)

        #Action Space: Portfolio weights for each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        #Observation Space: Concatenation of various features
        obs_shape = (
            self.n_assets +  #Price log returns
            self.n_assets +  #Current portfolio weights
            1 +             #Log normalized portfolio value
            1 +             #Risk appetite
            self.num_regimes +
            self.micro_indicators.shape[1] +
            self.macro_indicators.shape[1]
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        #Resetting the environment to initial state
        self.current_step = self.window_size  #Start after window for covariance
        self.portfolio_value = self.initial_amount
        self.asset_weights = np.ones(self.n_assets) / self.n_assets
        self.current_holdings = (self.portfolio_value * self.asset_weights) / self.prices[self.current_step]
        return self._next_observation()

    def _next_observation(self):
        #Calculating log returns for current step vs previous
        price_log_returns = np.log(self.prices[self.current_step] / self.prices[self.current_step - 1] + 1e-8)
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1
        log_portfolio_value = np.log(self.portfolio_value / self.initial_amount + 1e-8)

        obs = np.concatenate([
            price_log_returns,
            self.asset_weights,
            [log_portfolio_value],
            [self.risk_appetite],
            regime_onehot,
            self.micro_indicators[self.current_step],
            self.macro_indicators[self.current_step]
        ])
        return obs

    def _calculate_transaction_cost(self, new_weights):
        #Calculating transaction costs based on new weights
        target_dollars = self.portfolio_value * new_weights
        current_dollars = self.current_holdings * self.prices[self.current_step]
        trades = target_dollars - current_dollars
        transaction_cost = np.sum(np.abs(trades)) * self.transaction_fee
        return transaction_cost, target_dollars

    def get_covariance_and_return(self):
        #Computing mean returns and covariance matrix for the last window
        start = self.current_step - self.window_size
        window_prices = self.prices[start:self.current_step + 1]

        if len(window_prices) < 2:
            return np.zeros(self.n_assets), np.zeros((self.n_assets, self.n_assets))

        returns = np.diff(np.log(window_prices), axis=0)
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)

        #Regularizing covariance matrix to ensure stability
        cov_matrix += np.eye(self.n_assets) * 1e-6
        return mean_returns, cov_matrix

    def step(self, action):
        #Ensuring valid weights and minimum allocation
        action = np.clip(action, 0, 1)
        min_weight = 0.01
        action = np.clip(action, min_weight, None)
        new_weights = action / np.sum(action)

        #Applying transaction costs and updating portfolio
        transaction_cost, target_dollars = self._calculate_transaction_cost(new_weights)
        self.portfolio_value -= transaction_cost
        new_weights = target_dollars / self.portfolio_value if self.portfolio_value > 0 else np.zeros(self.n_assets)

        #Saving previous state for reward calculation
        prev_weights = self.asset_weights.copy()
        prev_portfolio_value = self.portfolio_value
        
        #Update holdings and portfolio value
        self.current_holdings = (self.portfolio_value * new_weights) / self.prices[self.current_step]
        self.asset_weights = new_weights

        #Calculating returns and new portfolio value
        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        returns = (next_prices - current_prices) / current_prices

        portfolio_return = np.dot(new_weights, returns)
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

        #Calculate portfolio variance as risk measure
        mean_returns, cov = self.get_covariance_and_return()
        portfolio_variance = np.dot(new_weights.T, np.dot(cov, new_weights))

        #Reward Calculation
        log_return = np.log(new_portfolio_value / prev_portfolio_value + 1e-8)
        turnover = np.sum(np.abs(new_weights - prev_weights))

        #Risk penalty scales inversely with risk appetite
        risk_penalty = (1 - self.risk_appetite) * portfolio_variance
        turnover_penalty = self.turnover_penalty * turnover
        transaction_cost_penalty = transaction_cost / self.portfolio_value
        reward = log_return - risk_penalty - turnover_penalty - transaction_cost_penalty

        #Advancing to next step
        self.current_step += 1
        self.portfolio_value = new_portfolio_value

        done = self.current_step >= len(self.prices) - 2

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        #Displaying current portfolio status
        current_prices = self.prices[self.current_step]
        holdings_value = self.current_holdings * current_prices
        total_value = np.sum(holdings_value)

        print(f"\nStep: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        for i, ticker in enumerate(tickers):
            print(f"{ticker}: {self.current_holdings[i]:.2f} shares (${holdings_value[i]:.2f}, {100*holdings_value[i]/total_value:.1f}%)")

        mean_returns, cov = self.get_covariance_and_return()
        print(f"Expected returns: {np.round(mean_returns, 4)}")
        print(f"Portfolio variance (mean diag): {np.round(np.mean(np.diag(cov)), 6)}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#Main Execution block
if __name__ == "__main__":
    print("Downloading stock data...")
    #Downloading historical stock data for the defined tickers
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    adj_close_data = pd.DataFrame()
    for ticker in tickers:
        adj_close_data[ticker] = data[ticker]['Close']

    print("Downloading VIX data...")
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    adj_close_data = adj_close_data.loc[vix.index].dropna()
    vix = vix.loc[adj_close_data.index]

    #Normalizing VIX data and classifying regimes
    scaler = MinMaxScaler()
    vix_norm = scaler.fit_transform(vix.values.reshape(-1, 1)).flatten()
    regime_classes = classify_regime(vix_norm)

    print("Computing technical indicators...")
    micro_indicators = compute_micro_indicators(adj_close_data)
    indicator_scaler = MinMaxScaler().fit(micro_indicators)
    micro_indicators = indicator_scaler.transform(micro_indicators)

    print("Processing macroeconomic data...")
    macro_df = pd.read_csv('macroeconomic_data_2010_2024.csv', parse_dates=['Date'])
    macro_df.set_index('Date', inplace=True)
    macro_df = macro_df.reindex(adj_close_data.index, method='ffill')

    macro_df = macro_df[selected_macro_columns]
    macro_scaler = MinMaxScaler().fit(macro_df)
    macro_indicators = macro_scaler.transform(macro_df)

    prices = adj_close_data.values

    #Creating the stock portfolio environment
    print("Creating environment...")
    env = StockPortfolioEnv(prices, regime_classes, micro_indicators, macro_indicators, 
                            initial_amount=10000, risk_appetite=0.5, transaction_fee=0.001)
    #Seeding the environment for reproducibility
    env.seed(42)

    #Training the PPO model
    print("Training model...")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01)

    #Model training timesteps with progress bar
    model.learn(total_timesteps=500000, progress_bar=True)

    #Saving and printing results
    mu, cov = env.get_covariance_and_return()
    print("Mean expected returns:", mu)
    print("Covariance matrix shape:", cov.shape)

    #Saving the trained model and scalers
    print("Saving model and scalers...")
    model.save("ppo_stock_model")
    joblib.dump(scaler, "vix_scaler.pkl")
    joblib.dump(indicator_scaler, "indicator_scaler.pkl")
    joblib.dump(macro_scaler, "macro_scaler.pkl")
    with open("tickers.json", "w") as f:
        json.dump(tickers, f)
