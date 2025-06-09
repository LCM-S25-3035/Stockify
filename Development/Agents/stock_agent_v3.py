# Import necessary libraries for technical indicators, data handling, plotting, RL, etc.
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
import joblib
import json

#List of stock tickers to analyze in the portfolio
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "UNH"
]

#Defining the start and end dates for historical data download
start_date = '2015-01-01'
end_date = '2025-05-20'

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
        #Calculating Accumulation/Distribution Index
        acc_dist = ta.volume.AccDistIndexIndicator(high=high, low=low, close=close, volume=df[col]).acc_dist_index().fillna(0).values
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
        #Ultimate Oscillator indicator
        ultimate_osc = ta.momentum.UltimateOscillator(high=high, low=low, close=close).ultimate_oscillator().fillna(50).values
        #Percentage Price Oscillator (PPO)
        ppo = ta.momentum.PercentagePriceOscillator(close=close).ppo().fillna(0).values
        #Average True Range (ATR) with backfill for missing data
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().bfill().values
        #Keltner Channel width (high band - low band)
        keltner = ta.volatility.KeltnerChannel(high=high, low=low, close=close).keltner_channel_hband().bfill().values - \
                  ta.volatility.KeltnerChannel(high=high, low=low, close=close).keltner_channel_lband().bfill().values
        #Exponential Moving Average (EMA) with 10-period window
        ema = ta.trend.EMAIndicator(close=close, window=10).ema_indicator().bfill().values
        #Bollinger Bands width
        bb_width = ta.volatility.BollingerBands(close=close).bollinger_wband().fillna(0).values
        #Stochastic Oscillator %K
        stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch().fillna(0).values

        #Stacking all features vertically and transpose to shape (time_steps, features)
        asset_features = np.vstack([
            rsi, macd_diff, ema, bb_width, stoch_k,
            obv, acc_dist,
            adx, psar, ichimoku_diff,
            williams_r, ultimate_osc, ppo,
            atr, keltner
        ]).T
        features.append(asset_features)

    #Horizontally stacking features of all assets to get final feature matrix for all assets
    return np.hstack(features)

#Custom OpenAI Gym environment to simulate stock portfolio trading
class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, regime_class, micro_indicators, initial_amount=10000, risk_appetite=0.2, transaction_fee=0.001):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices                      #Historical prices of all assets
        self.regime_class = regime_class          #Market regime classification for each time step
        self.micro_indicators = micro_indicators  #Technical indicators for each time step
        self.initial_amount = initial_amount      #Initial investment capital
        self.risk_appetite = risk_appetite        #Controls trade-off between return and risk
        self.transaction_fee = transaction_fee    #Proportional transaction fee per trade

        self.n_assets = prices.shape[1]           #Number of assets in the portfolio
        self.num_regimes = 3                      #Number of market regimes (bullish, sideways, bearish)
        self.current_step = 0                      #Current time step in environment
        self.current_holdings = np.zeros(self.n_assets)  #Number of shares currently held

        #Action space: portfolio weights for each asset (continuous between 0 and 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        #Observation space: normalized prices + portfolio value + risk appetite + regime one-hot + technical indicators
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1]
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
        #Creating one-hot encoding of current market regime
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1

        #Observation vector concatenates normalized prices, portfolio info, regime, and micro indicators
        obs = np.concatenate([
            self.prices[self.current_step] / self.prices[0],       #Normalized prices relative to initial prices
            [self.portfolio_value / self.initial_amount],          #Normalized portfolio value
            [self.risk_appetite],                                  #Risk appetite (constant)
            regime_onehot,                                         #One-hot encoded regime
            self.micro_indicators[self.current_step]              #Technical indicators at current step
        ])
        return obs

    def _calculate_transaction_cost(self, new_weights):
        #Calculating the dollar amount needed for target weights
        target_dollars = self.portfolio_value * new_weights
        #Current dollar value of holdings
        current_dollars = self.current_holdings * self.prices[self.current_step]
        #Dollar amount to trade (buy or sell)
        trades = target_dollars - current_dollars
        #Total transaction cost proportional to absolute trade amount
        transaction_cost = np.sum(np.abs(trades)) * self.transaction_fee
        return transaction_cost, target_dollars

    def step(self, action):
        #Clipping actions to valid range [0,1]
        action = np.clip(action, 0, 1)
        #Normalizing actions to sum to 1 (portfolio weights)
        new_weights = action / (np.sum(action) + 1e-8)
        #Calculating transaction cost for moving from current holdings to new weights
        transaction_cost, target_dollars = self._calculate_transaction_cost(new_weights)
        #Deducting transaction cost from portfolio value
        self.portfolio_value -= transaction_cost
        #Recalculating weights after transaction cost deduction
        new_weights = target_dollars / self.portfolio_value if self.portfolio_value > 0 else np.zeros(self.n_assets)
        #Updating current holdings based on new weights and current prices
        self.current_holdings = (self.portfolio_value * new_weights) / self.prices[self.current_step]
        self.asset_weights = new_weights

        #Calculating returns for each asset from current to next step
        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        returns = (next_prices - current_prices) / current_prices

        #Calculating portfolio return as weighted sum of asset returns
        portfolio_return = np.dot(new_weights, returns)
        #Updating portfolio value based on returns
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        #Calculating risk penalty as std deviation of returns
        risk_penalty = np.std(returns)
        #Calculating reward: portfolio return minus risk penalty and transaction cost penalty
        reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty - (transaction_cost / self.portfolio_value)

        self.current_step += 1
        self.portfolio_value = new_portfolio_value
        done = self.current_step >= len(self.prices) - 2  #End episode if at last step

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        #Printing current portfolio state: holdings and their values
        current_prices = self.prices[self.current_step]
        holdings_value = self.current_holdings * current_prices
        total_value = np.sum(holdings_value)
        print(f"\nStep: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        for i, ticker in enumerate(tickers):
            print(f"{ticker}: {self.current_holdings[i]:.2f} shares (${holdings_value[i]:.2f}, {100*holdings_value[i]/total_value:.1f}%)")

#Main Execution block
if __name__ == "__main__":
    print("Downloading stock data...")
    #Downloading historical stock data for all tickers from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    #Extracting adjusted close prices into a DataFrame
    adj_close_data = pd.DataFrame()
    for ticker in tickers:
        adj_close_data[ticker] = data[ticker]['Close']

    print("Downloading VIX data...")
    #Downloading VIX (market volatility index) data for regime classification
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']

    #Aligning VIX and stock data on dates and drop missing values
    adj_close_data = adj_close_data.loc[vix.index].dropna()
    vix = vix.loc[adj_close_data.index]

    #Normalizing VIX values between 0 and 1
    scaler = MinMaxScaler()
    vix_norm = scaler.fit_transform(vix.values.reshape(-1, 1)).flatten()
    #Classifing market regimes based on normalized VIX
    regime_classes = classify_regime(vix_norm)

    print("Computing technical indicators...")
    #Computing technical indicators for all assets
    micro_indicators = compute_micro_indicators(adj_close_data)
    #Normalizing technical indicators across all features
    indicator_scaler = MinMaxScaler().fit(micro_indicators)
    micro_indicators = indicator_scaler.transform(micro_indicators)

    #Extracting prices as numpy array for faster processing
    prices = adj_close_data.values

    print("Creating environment...")
    #Initializing the custom gym environment with prices, regimes, indicators, etc.
    env = StockPortfolioEnv(prices, regime_classes, micro_indicators, 10000, 0.5, 0.001)

    print("Training model...")
    #Creating PPO reinforcement learning model
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01)
    #Training the model for 1,000,000 time steps
    model.learn(total_timesteps=1000000)

    print("Saving model and scalers...")
    #Saving trained model and scalers for later use
    model.save("ppo_stock_model")
    joblib.dump(scaler, "vix_scaler.pkl")
    joblib.dump(indicator_scaler, "indicator_scaler.pkl")
    with open("tickers.json", "w") as f:
        json.dump(tickers, f)

    print("Testing model...")
    #Testing the trained model by running through the environment and tracking portfolio value
    obs = env.reset()
    done = False
    portfolio_values = [env.initial_amount]
    allocations = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        allocations.append(env.asset_weights)
        if done or len(portfolio_values) % 50 == 0:
            env.render()

    print("\nFinal Results")
    print(f"Initial Investment: ${env.initial_amount:,.2f}")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Profit: ${portfolio_values[-1] - env.initial_amount:,.2f}")
    print(f"Annualized Return: {100 * ((portfolio_values[-1] / env.initial_amount) ** (252 / len(portfolio_values)) - 1):.2f}%")

    #Showing individual asset returns from start to end
    initial_prices = prices[0]
    final_prices = prices[-1]
    print("\nIndividual Asset Performance:")
    for i, ticker in enumerate(tickers):
        asset_return = 100 * (final_prices[i] - initial_prices[i]) / initial_prices[i]
        print(f"{ticker}: {asset_return:.2f}%")

    #Plotting portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time (With Transaction Fees)")
    plt.xlabel("Days")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.show()

    #Plotting asset allocation weights over time
    allocations = np.array(allocations)
    plt.figure(figsize=(12, 6))
    for i in range(len(tickers)):
        plt.plot(allocations[:, i], label=tickers[i])
    plt.title("Asset Allocation Over Time")
    plt.xlabel("Days")
    plt.ylabel("Allocation Weight")
    plt.legend()
    plt.grid(True)
    plt.show()

    #Plotting normalized VIX and regime thresholds
    plt.figure(figsize=(12, 6))
    plt.plot(vix_norm, label='Normalized VIX')
    plt.plot([0.33] * len(vix_norm), 'g--', label='Bullish Threshold')
    plt.plot([0.66] * len(vix_norm), 'r--', label='Bearish Threshold')
    plt.title("Market Regime Classification")
    plt.xlabel("Days")
    plt.ylabel("VIX (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.show()
