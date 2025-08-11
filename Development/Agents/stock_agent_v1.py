#Importing libraries
import ta.momentum
import ta.trend
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3 import PPO
#Technical Analysis library
import ta


#Downloading historical Data
#We can remove this part when we have seperate data file from API

#Defining tickers and date range

tickers = ['AAPL', 'GOOG', 'AMZN', 'MSFT']
start_date = '2015-01-01'
end_date = '2025-05-20'

#Downloading adjusted price for stocks from yahoo finance library
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
adj_close_data = pd.DataFrame()

#Looping through each tickers
for ticker in tickers:
    adj_close_data[ticker] = data[ticker]['Close']

#Downloading VIX index as a proxy for market volatiliy indicators (Regime indicator)
#Tells how volatile market will be
vix = yf.download('^VIX', start=start_date, end=end_date)['Close']

#Aligning stock and VIX data by date and dropping any missing values
adj_close_data = adj_close_data.loc[vix.index].dropna()
vix = vix.loc[adj_close_data.index]

#Extractin prices as numPy array
prices = adj_close_data.values

#Computing market regimes

#Normalizing VIX values to [0,1] range
vix_norm = (vix - vix.min()) / (vix.max() - vix.min())

#Function to define market regimes
def classify_regime(vix_norm):
    #Defining  thresholds (these can be tuned based on your data distribution)
    bullish_thresh = 0.33
    bearish_thresh = 0.66
    
    regimes_class = []
    for val in vix_norm:
        if val <= bullish_thresh:
            regimes_class.append(0)  # Bullish
        elif val <= bearish_thresh:
            regimes_class.append(1)  # Sideways
        else:
            regimes_class.append(2)  # Bearish
    return np.array(regimes_class)

#Regime classes
regime_classes = classify_regime(vix_norm.values)


#Computing micor indicators

def compute_micro_indicators(df):
    features = []
    for col in df.columns:
        close = df[col]
        #Computing RSI(Relative Strength Index)
        rsi = ta.momentum.RSIIndicator(close).rsi().fillna(50).values
        #Computing MACD difference (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close).macd_diff().fillna(0).values
        #Stacking indicators per stocks
        asset_features = np.vstack([rsi, macd]).T
        features.append(asset_features)

    #Concatenating all stock indicators side by side
    return np.hstack(features)

micro_indicators = compute_micro_indicators(adj_close_data)

#Saving Processed Data 
#To inspect or use again
adj_close_data.to_csv("../Data/adj_close_prices.csv")

#Saving normalized VIX data (regime indicator)
vix_norm_df = pd.DataFrame(regime_classes, index=adj_close_data.index, columns=["Regime Class"])
vix_norm_df.to_csv("../Data/regime_data.csv")

#Saving micro indicators (RSI, MACD)
micro_cols = []
for ticker in tickers:
    micro_cols.extend([f"{tickers}_RSI", f"{tickers}_MACD"])
micro_df = pd.DataFrame(micro_indicators, index=adj_close_data.index, columns=micro_cols)
micro_df.to_csv("../Data/micro_indicators.csv")

#Incoding regimes classes with one hot encoding
def one_hot_regime(regime_class, num_classes=3):
    one_hot = np.zeros(num_classes)
    one_hot[regime_class] = 1
    return one_hot

#Defining cutsome openAI Gym environment for the agent building

class StockPortfolioEnv(gym.Env):
    def __init__(self,prices,regime_class,micro_indicators,amount, risk_appetite, duration):
        super(StockPortfolioEnv, self).__init__()
        self.prices = prices
        self.regime_class = regime_class
        self.micro_indicators = micro_indicators
        #Initial ammount of investment
        self.amount = amount
        #Risk appetite of the agent
        #0-1 while 0 being safest and 1 being riskiest
        self.risk_appetite = risk_appetite
        #Duration of the investment
        self.duration = duration
        self.n_assets = prices.shape[1]

        #Observation shape: prices + portfolio_value + risk_appetite + regime_onehot + micro_indicators
        self.num_regimes = 3
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1]

        #Action Space: Allocation weight for each asset
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        #Observation spcae: price + indicators
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.amount
        return self._get_observation()
    
    def step(self, action):
        #Normalizing action to sum to 1
        action = np.clip(action, 0,1)
        action = action / (np.sum(action) + 1e-8)

        #Calculating retunr from previous to current step
        prev_prices = self.prices[self.current_step]
        self.current_step += 1
        curr_prices = self.prices[self.current_step]

        #Calculating daily return and protfolio return
        returns = (curr_prices - prev_prices) / prev_prices
        portfolio_return = np.dot(action, returns)
        self.portfolio_value *= (1 + portfolio_return)

        #Risk Penalty (Volatility penalty if risk appetite is low)
        risk_penalty = np.std(returns)
        adjusted_reward = portfolio_return - (1 - self.risk_appetite) * risk_penalty

        #Checking if simulation is done
        done = self.current_step >= self.duration or self.current_step >= len(self.prices) -1
        return self._get_observation(), adjusted_reward, done, {}
    
    def _get_observation(self):
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regime_class[self.current_step]] = 1
        return np.concatenate([
            #Stock Prices
            self.prices[self.current_step],
            #Current portfolio value
            [self.portfolio_value],
            #Risk Appetite
            [self.risk_appetite],
            #Market Regimes
            regime_onehot,
            #Technical indicators
            self.micro_indicators[self.current_step]

        ])
    
#Training RL model usin PPO

initial_amount = 100000
env = StockPortfolioEnv(prices, 
                        regime_classes, 
                        micro_indicators, 
                        amount=initial_amount, 
                        risk_appetite=0.5, 
                        duration=365)

#Initializing PPO agent with MLP Policy
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=250000)

#Testing the model and tracking allocations

obs = env.reset()
values = [env.portfolio_value]
allocations = []

for _ in range(252):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    #Normalizing and converting allocation to dollar values
    action = np.clip(action, 0, 1)
    action /= (np.sum(action) + 1e-8)
    dollar_allocation = env.portfolio_value * action
    allocation_dict = {ticker: round(dollar_allocation[i], 2) for i, ticker in enumerate(tickers)}
    allocations.append(allocation_dict)


    values.append(env.portfolio_value)

    if done:
        break

#Displaying final allocation and profit

print("\n Final Portfolio Allocation and Profit")
#Initializing value at 0 first 
total_value = 0
for ticker, val in allocations[-1].items():
    total_value += val
    print(f"{ticker}: ${val:,.2f}")

total_profit = total_value - initial_amount
print(f"\nTotal Profit: ${total_profit:,.2f}")


#Saving allocation history to CSV for inspection

pd.DataFrame(allocations).to_csv("../Data/allocation_history.csv", index=False)

#Plotting the portfolio values over time

plt.figure(figsize=(10, 5))
plt.plot(values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Days")
plt.ylabel("Value ($)")
plt.grid(True)
plt.show()
