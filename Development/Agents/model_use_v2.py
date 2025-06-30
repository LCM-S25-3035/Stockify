import joblib
import json
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from stable_baselines3 import PPO
from stock_agent_v3 import StockPortfolioEnv
import matplotlib.pyplot as plt

#User inputs for simulation for investment parameters
amount = float(input("Enter investment amount (e.g., 5000): "))  #Intial amount to invest
duration = int(input("Enter investment duration in days (e.g., 200): "))  #Investment period
risk = float(input("Enter risk appetite (0 to 1): "))  #Risk appetite between 0 (low) and 1 (high)

#Loading pre-trained reinforcement learning model and data scalers
#Loading the trained PPO model for stock trading
model = PPO.load("D:\\Big Data Analytics\\Term 3\\Capstone Project\\stockify\\Agents\\ppo_stock_model.zip")
vix_scaler = joblib.load("D:\\Big Data Analytics\\Term 3\\Capstone Project\\stockify\\Agents\\vix_scaler.pkl")
indicator_scaler = joblib.load("D:\\Big Data Analytics\\Term 3\\Capstone Project\\stockify\\Agents\\indicator_scaler.pkl")
 
#List of stock tickers to use in the portfolio
with open("tickers.json", "r") as f:
    tickers = json.load(f)                         

#Function to classify market regimes based on normalized VIX value
def classify_regime(vix_norm):
    #Regimes: 0 = low volatility, 1 = medium, 2 = high volatility
    return np.where(vix_norm <= 0.33, 0, np.where(vix_norm <= 0.66, 1, 2))

#Computing a wide range of micro-level technical indicators for each asset
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

#Preparing the trading environment with all necessary data and parameters
def prepare_env(n_days=duration, initial_amount=amount, risk_appetite=risk):
    #Defining start and end dates for historical data (including a buffer of 10 days)
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=n_days + 10)).strftime('%Y-%m-%d')

    #Downloading adjusted close prices for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    adj_close_data = pd.DataFrame()
    for ticker in tickers:
        adj_close_data[ticker] = data[ticker]['Close']
    #Removing dates with missing data
    adj_close_data = adj_close_data.dropna()  

    #Downloading VIX (volatility index) data and align it to price dates
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    vix = vix.loc[adj_close_data.index]

    #Normalizing VIX data and classify market regimes
    vix_norm = vix_scaler.transform(vix.values.reshape(-1, 1)).flatten()
    regimes = classify_regime(vix_norm)

    #Computing technical indicators for all assets and normalize them
    micro_indicators = compute_micro_indicators(adj_close_data)
    micro_indicators = indicator_scaler.transform(micro_indicators)

    prices = adj_close_data.to_numpy()  #Convertting price DataFrame to numpy array

    #Initializing the stock portfolio environment with all inputs
    env = StockPortfolioEnv(prices, regimes, micro_indicators,
                            initial_amount=initial_amount,
                            risk_appetite=risk_appetite,
                            transaction_fee=0.001)  #Setting transaction fee as 0.1%
    return env



#Running the trading simulation
env = prepare_env()  #Creating environment with  inputs
obs = env.reset()    #Reseting environment to initial state
done = False
portfolio_values = [env.initial_amount]  #Tracking portfolio value over time, starting with initial amount

#Stepping through environment until done (investment period ends)
while not done:
    action, _ = model.predict(obs)  #Model predicts portfolio allocation actions
    obs, reward, done, _ = env.step(action)  #Applying action, receive next observation and reward
    portfolio_values.append(env.portfolio_value)  #Recording portfolio value after the step

#Outputing final investment results
final_value = env.portfolio_value
total_profit = final_value - amount
print(f"\n✅ Final portfolio value after {duration} days: ${final_value:.2f}")
print(f"💰 Total profit: ${total_profit:.2f}")

#Plotting portfolio value growth over the investment period
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label='Portfolio Value')
plt.title(f"Portfolio Growth Over {duration} Days")
plt.xlabel("Days")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
