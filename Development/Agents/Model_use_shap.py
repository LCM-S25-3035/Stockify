import joblib
import json
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from stable_baselines3 import PPO
from stock_agent_v3 import StockPortfolioEnv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap

#User inputs for simulation for investment parameters
amount = float(input("Enter investment amount (e.g., 5000): "))  #Intial amount to invest
duration = int(input("Enter investment duration in days (e.g., 200): "))  #Investment period
risk = float(input("Enter risk appetite (0 to 1): "))  #Risk appetite between 0 (low) and 1 (high)

#Loading pre-trained reinforcement learning model and data scalers
#Loading the trained PPO model for stock trading
model = PPO.load("ppo_stock_model") 
#Scaler to normalize VIX volatility index data               
vix_scaler = joblib.load("vix_scaler.pkl")   
#Scaler for technical indicators       
indicator_scaler = joblib.load("indicator_scaler.pkl")  
#List of stock tickers to use in the portfolio
with open("tickers.json", "r") as f:
    tickers = json.load(f)                         

#Function to classify market regimes based on normalized VIX value
def classify_regime(vix_norm):
    #Regimes: 0 = low volatility, 1 = medium, 2 = high volatility
    return np.where(vix_norm <= 0.33, 0, np.where(vix_norm <= 0.66, 1, 2))

#Computing a wide range of micro-level technical indicators for each asset
def compute_micro_indicators(df):
    features = []
    # NOTE: Here 'high', 'low', and 'volume' are assigned close price, need to update with actual data.
    for col in df.columns:
        close = df[col]
        high = df[col]   
        low = df[col]

        #Calculating various technical indicators using the 'ta' library
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=close).on_balance_volume().fillna(0).values
        acc_dist = ta.volume.AccDistIndexIndicator(high=high, low=low, close=close, volume=close).acc_dist_index().fillna(0).values
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close).adx().fillna(0).values
        psar = ta.trend.PSARIndicator(high=high, low=low, close=close).psar().bfill().values
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low)
        ichimoku_diff = ichimoku.ichimoku_conversion_line().bfill().values - ichimoku.ichimoku_base_line().bfill().values
        rsi = ta.momentum.RSIIndicator(close=close).rsi().fillna(50).values
        macd_diff = ta.trend.MACD(close=close).macd_diff().fillna(0).values
        williams_r = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r().fillna(-50).values
        ultimate_osc = ta.momentum.UltimateOscillator(high=high, low=low, close=close).ultimate_oscillator().fillna(50).values
        ppo = ta.momentum.PercentagePriceOscillator(close=close).ppo().fillna(0).values
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().bfill().values
        keltner = ta.volatility.KeltnerChannel(high=high, low=low, close=close).keltner_channel_hband().bfill().values - \
                  ta.volatility.KeltnerChannel(high=high, low=low, close=close).keltner_channel_lband().bfill().values
        ema = ta.trend.EMAIndicator(close=close).ema_indicator().bfill().values
        bb_width = ta.volatility.BollingerBands(close=close).bollinger_wband().fillna(0).values
        stoch_k = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch().fillna(0).values

        #Combining all indicators into a single feature array per asset (rows = time steps, columns = features)
        asset_features = np.vstack([
            rsi, macd_diff, ema, bb_width, stoch_k,
            obv, acc_dist,
            adx, psar, ichimoku_diff,
            williams_r, ultimate_osc, ppo,
            atr, keltner
        ]).T
        features.append(asset_features)

    #Concatenating features horizontally for all assets
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
    adj_close_data.dropna(inplace=True)  

    #Downloading VIX (volatility index) data and align it to price dates
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    vix = vix.loc[adj_close_data.index]

    #Normalizing VIX data and classify market regimes
    vix_norm = vix_scaler.transform(vix.values.reshape(-1, 1)).flatten()
    regimes = classify_regime(vix_norm)

    #Computing technical indicators for all assets and normalize them
    micro_indicators = compute_micro_indicators(adj_close_data)
    micro_indicators = indicator_scaler.transform(micro_indicators)

    prices = adj_close_data.values  #Convertting price DataFrame to numpy array

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

X_state, y_action = [], []

while not done:
    action, _ = model.predict(obs)
    X_state.append(obs)
    y_action.append(np.argmax(action))
    obs, reward, done, _ = env.step(action)
    portfolio_values.append(env.portfolio_value)

final_value = env.portfolio_value
total_profit = final_value - amount
print(f"\n✅ Final portfolio value after {duration} days: ${final_value:.2f}")
print(f"💰 Total profit: ${total_profit:.2f}")

# Plot portfolio growth
plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label='Portfolio Value')
plt.title(f"Portfolio Growth Over {duration} Days")
plt.xlabel("Days")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === XAI Pipeline ===
print("\n🔍 Running XAI surrogate model explanation...")
X_state = np.array(X_state)
y_action = np.array(y_action)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_state, y_action)

explainer = shap.Explainer(rf_model)
shap_values = explainer(X_state[:100])
shap.summary_plot(shap_values, X_state[:100], show=True)
print("✅ SHAP explanation completed.")

