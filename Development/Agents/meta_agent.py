import numpy as np
import joblib
import json
from stable_baselines3 import PPO, SAC
import gym
from gym import spaces

#Loading saved data and models
def load_agent_data(agent_name):
    #Loading tickers
    with open(f"{agent_name}_tickers.json", "r") as f:
        tickers = json.load(f)

    #Loading numpy arrays: prices, regimes, micro_indicators, macro_indicators
    prices = np.load(f"prices_{agent_name}.npy")
    regimes = np.load(f"regime_{agent_name}.npy")
    micro_indicators = np.load(f"micro_indicators_{agent_name}.npy")
    macro_indicators = np.load(f"macro_indicators_{agent_name}.npy")  # NEW line for macros

    #Loading PPO model
    model = PPO.load(f"ppo_{agent_name}_model.zip")

    #Loading scalers
    scaler_indicator = joblib.load(f"indicator_scaler_{agent_name}.pkl")
    scaler_macro = joblib.load(f"macro_scaler_{agent_name}.pkl")  

    #Printing shapes for debugging
    print(f"{agent_name} micro_indicators shape: {micro_indicators.shape}")
    print(f"{agent_name} macro_indicators shape: {macro_indicators.shape}")

    return tickers, prices, regimes, micro_indicators, macro_indicators, model, scaler_indicator, scaler_macro


#Custom Gym environment for each agent
class AgentEnv(gym.Env):
    def __init__(self, prices, regimes, micro_indicators, macro_indicators, initial_amount=10000, risk_appetite=0.5, transaction_fee=0.001):
        super(AgentEnv, self).__init__()
        self.prices = prices
        self.regimes = regimes
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee

        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        self.current_step = 0

        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1] + macro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        return self._next_observation()

    def _next_observation(self):
        regime_onehot = np.zeros(self.num_regimes)
        regime_onehot[self.regimes[self.current_step]] = 1
        obs = np.concatenate([
            self.prices[self.current_step] / self.prices[0],
            [self.portfolio_value / self.initial_amount],
            [self.risk_appetite],
            regime_onehot,
            self.micro_indicators[self.current_step],
            self.macro_indicators[self.current_step]
        ])
        return obs

    def get_covariance_and_return(self, window_size=20):
        step = self.current_step
        start = max(0, step - window_size)
        window_prices = self.prices[start:step+1]
        if len(window_prices) < 2:
            returns = np.zeros((1, self.n_assets))
        else:
            returns = np.diff(window_prices, axis=0) / window_prices[:-1]
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))
        return mu, cov


#Loading
stock_data = load_agent_data("stock")
crypto_data = load_agent_data("crypto")
etf_data = load_agent_data("etf")

#Unbacking data
tickers_stock, prices_stock, regimes_stock, micro_stock, macro_stock, ppo_stock, scaler_stock, macro_scaler_stock = stock_data
tickers_crypto, prices_crypto, regimes_crypto, micro_crypto, macro_crypto, ppo_crypto, scaler_crypto, macro_scaler_crypto = crypto_data
tickers_etf, prices_etf, regimes_etf, micro_etf, macro_etf, ppo_etf, scaler_etf, macro_scaler_etf = etf_data

#Initialize environments
env_stock = AgentEnv(prices_stock, regimes_stock, micro_stock, macro_stock)
env_crypto = AgentEnv(prices_crypto, regimes_crypto, micro_crypto, macro_crypto)
env_etf = AgentEnv(prices_etf, regimes_etf, micro_etf, macro_etf)

env_stock.reset()
#Now printing shapes to debug
print("Stock Prices shape:", prices_stock.shape)
print("Stock Micro indicators shape:", micro_stock.shape)
print("Stock Expected obs shape:", env_stock.observation_space.shape)
print("Stock Sample obs shape:", env_stock._next_observation().shape)

env_crypto.reset()
print("Crypto Prices shape:", prices_crypto.shape)
print("Crypto Micro indicators shape:", micro_crypto.shape)
print("Crypto Expected obs shape:", env_crypto.observation_space.shape)
print("Crypto Sample obs shape:", env_crypto._next_observation().shape)

env_etf.reset()
print("ETF Prices shape:", prices_etf.shape)
print("ETF Micro indicators shape:", micro_etf.shape)
print("ETF Expected obs shape:", env_etf.observation_space.shape)
print("ETF Sample obs shape:", env_etf._next_observation().shape)

#Function to get mu, cov, and weights from agent portfolios
def get_agent_portfolio(ppo_model, env):
    env.reset()
    obs = env._next_observation().reshape(1, -1)
    action, _ = ppo_model.predict(obs)
    weights = action / (np.sum(action) + 1e-8)
    mu, cov = env.get_covariance_and_return()
    return mu, cov, weights

mu_stock, cov_stock, w_stock = get_agent_portfolio(ppo_stock, env_stock)
mu_crypto, cov_crypto, w_crypto = get_agent_portfolio(ppo_crypto, env_crypto)
mu_etf, cov_etf, w_etf = get_agent_portfolio(ppo_etf, env_etf)

#Combine all agent portfolios
mu_all = np.concatenate([mu_stock, mu_crypto, mu_etf])

#Block diagonal covariance
cov_all = np.block([
    [cov_stock, np.zeros((len(mu_stock), len(mu_crypto))), np.zeros((len(mu_stock), len(mu_etf)))],
    [np.zeros((len(mu_crypto), len(mu_stock))), cov_crypto, np.zeros((len(mu_crypto), len(mu_etf)))],
    [np.zeros((len(mu_etf), len(mu_stock))), np.zeros((len(mu_etf), len(mu_crypto))), cov_etf]
])

#Meta Environment for portfolio optimization
class MetaPortfolioEnv(gym.Env):
    def __init__(self, mu, cov, amount, risk, duration):
        super(MetaPortfolioEnv, self).__init__()
        self.mu = mu
        self.cov = cov
        self.amount = amount
        self.risk = risk
        self.duration = duration
        self.n_assets = len(mu)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets * 2,), dtype=np.float32)

    def reset(self):
        self.weights = np.ones(self.n_assets) / self.n_assets
        return np.concatenate([self.mu, np.diag(self.cov)])

    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights = weights / (weights.sum() + 1e-8)
        expected_return = np.dot(weights, self.mu)
        risk_penalty = self.risk * np.dot(weights.T, np.dot(self.cov, weights))
        reward = expected_return - risk_penalty
        done = True
        info = {"weights": weights}
        return self.reset(), reward, done, info

#User input for meta agent
amount = float(input("Enter total investment amount ($): "))
risk = float(input("Enter risk appetite (0 - low to 1 - high): "))
duration = int(input("Enter investment duration (days): "))

#MetaEnv for SAC model train
meta_env = MetaPortfolioEnv(mu_all, cov_all, amount, risk, duration)

meta_model = SAC("MlpPolicy", meta_env, verbose=1)
meta_model.learn(total_timesteps=3000)

obs = meta_env.reset()
action, _ = meta_model.predict(obs)
weights = np.clip(action, 0, 1)
weights /= weights.sum()

#Calculate portfolio values
daily_return = np.dot(weights, mu_all)
yearly_return = (1 + daily_return) ** 252 - 1
valuation = amount * ((1 + daily_return) ** duration)
profit = valuation - amount
allocations = weights * amount

#Printing the results
print("\n====== Meta Portfolio Allocation Summary ======")
idx = 0
for agent_name, tickers_list in zip(["Stock", "Crypto", "ETF"], [tickers_stock, tickers_crypto, tickers_etf]):
    print(f"\n{agent_name} Assets:")
    for ticker in tickers_list:
        alloc = allocations[idx]
        pct = weights[idx] * 100
        print(f"  {ticker}: ${alloc:.2f} ({pct:.2f}%)")
        idx += 1

print(f"\nTotal Investment Amount: ${amount:,.2f}")
print(f"Projected Portfolio Valuation (after {duration} days): ${valuation:,.2f}")
print(f"Projected Profit: ${profit:,.2f}")
print(f"Estimated Yearly Return: {yearly_return*100:.2f}%")
