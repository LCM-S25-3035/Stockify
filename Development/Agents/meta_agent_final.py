#Importing necessary libraries
import numpy as np
import joblib
import json
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')  #Suppressing warnings for cleaner output


#Function to load agent data
def load_agent_data(agent_name):
    """
    Load data and trained PPO model for a given asset class (stock, crypto, ETF).

    Parameters:
        agent_name (str): Name of the sub-agent ('stock', 'crypto', 'etf').

    Returns:
        tickers, prices, regimes, micro_indicators, macro_indicators,
        model, scaler_indicator, scaler_macro
    """
    with open(f"{agent_name}_tickers.json", "r") as f:
        tickers = json.load(f)

    #Loading preprocessed numpy arrays
    prices = np.load(f"prices_{agent_name}.npy")
    regimes = np.load(f"regime_{agent_name}.npy")
    micro_indicators = np.load(f"micro_indicators_{agent_name}.npy")
    macro_indicators = np.load(f"macro_indicators_{agent_name}.npy")

    #Loading pre-trained PPO model
    model = PPO.load(f"ppo_{agent_name}_model.zip")

    #Loading feature scalers
    scaler_indicator = joblib.load(f"indicator_scaler_{agent_name}.pkl")
    scaler_macro = joblib.load(f"macro_scaler_{agent_name}.pkl")

    return tickers, prices, regimes, micro_indicators, macro_indicators, model, scaler_indicator, scaler_macro


#Custome gym environment for individual asset class agents
class AgentEnv(gym.Env):
    """
    Simulated environment for individual asset class agents (Stock, Crypto, ETF).
    """

    def __init__(self, prices, regimes, micro_indicators, macro_indicators,
                 initial_amount=10000, risk_appetite=0.5, transaction_fee=0.001):
        super(AgentEnv, self).__init__()

        #Storing data & parameters
        self.prices = prices
        self.regimes = regimes
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee

        #Asset & regime info
        self.n_assets = prices.shape[1]
        self.num_regimes = 3
        self.current_step = 0

        #Observation space = prices + portfolio value + risk + regime + indicators
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1] + macro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        #Action space = allocation weights for each asset (0 to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

    def reset(self):
        """Reset environment to starting state."""
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        return self._next_observation()

    def _next_observation(self):
        """Get the next observation vector for the model."""
        #Normalize prices relative to first day
        prices_part = np.asarray(self.prices[self.current_step] / self.prices[0], dtype=np.float32).ravel()

        #Portfolio and risk info
        value_part = np.asarray([self.portfolio_value / self.initial_amount], dtype=np.float32)
        risk_part = np.asarray([self.risk_appetite], dtype=np.float32)

        #One-hot encoding for current market regime
        regime_onehot = np.zeros(self.num_regimes, dtype=np.float32)
        regime_onehot[int(self.regimes[self.current_step])] = 1.0

        #Technical and macroeconomic indicators
        micro_part = np.asarray(self.micro_indicators[self.current_step], dtype=np.float32).ravel()
        macro_part = np.asarray(self.macro_indicators[self.current_step], dtype=np.float32).ravel()

        #Concatenating all features
        obs = np.concatenate([
            prices_part,
            value_part,
            risk_part,
            regime_onehot,
            micro_part,
            macro_part
        ], axis=0).astype(np.float32)

        return obs.reshape(-1)

    def get_covariance_and_return(self, window_size=60):
        """Compute mean returns and covariance matrix for assets."""
        step = self.current_step
        start = max(0, step - window_size)
        window_prices = self.prices[start:step + 1]

        if len(window_prices) < 2:
            returns = np.zeros((1, self.n_assets))
        else:
            returns = np.diff(np.log(window_prices), axis=0)

        mu = np.mean(returns, axis=0)
        cov = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))
        return mu, cov

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


#Align observation vector to model's expected input size
def align_obs_to_model(obs: np.ndarray, model_obs_space: spaces.Box):
    """
    Ensure observation vector matches model's expected input size.
    Pads or truncates if needed.
    """
    target_len = int(np.prod(model_obs_space.shape)) if isinstance(model_obs_space.shape, tuple) else int(model_obs_space.shape)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    cur_len = obs.shape[0]

    if cur_len == target_len:
        return obs
    elif cur_len < target_len:
        pad = np.zeros((target_len - cur_len,), dtype=np.float32)
        return np.concatenate([obs, pad], axis=0)
    else:
        return obs[:target_len]


#Getting portfolio data from trained PPO agent
def get_agent_portfolio(ppo_model: PPO, env: AgentEnv, window_size=60):
    """
    Run the PPO agent to get weights, expected returns, and covariance.
    """
    env.reset()
    env.current_step = window_size  #Ensuring enough data for returns calculation

    obs = env._next_observation()
    aligned_obs = align_obs_to_model(obs, ppo_model.observation_space)

    #Predicting allocation weights
    action, _ = ppo_model.predict(aligned_obs, deterministic=True)
    weights = np.asarray(action, dtype=np.float32)
    s = weights.sum()
    weights = np.ones_like(weights) / len(weights) if s <= 1e-8 else weights / s

    mu, cov = env.get_covariance_and_return(window_size=window_size)
    return mu, cov, weights


#Meta Portfolio Environment
class MetaPortfolioEnv(gym.Env):
    """
    Meta-portfolio environment that allocates across multiple PPO sub-agents.
    """

    def __init__(self, mu, cov, amount, risk, duration,
                 micro_indicators, macro_indicators, regimes, num_regimes=3):
        super(MetaPortfolioEnv, self).__init__()

        #Data
        self.mu = mu
        self.cov = cov
        self.amount = amount
        self.risk = risk
        self.duration = duration
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.regimes = regimes
        self.num_regimes = num_regimes

        #Dimensions
        self.n_assets = len(mu)
        self.micro_dim = micro_indicators.shape[1]
        self.macro_dim = macro_indicators.shape[1]

        #Observation & action spaces
        obs_dim = self.n_assets * 3 + 1 + self.micro_dim + self.macro_dim + self.num_regimes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        #State variables
        self.current_step = 0
        self.portfolio_value = amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets

    def reset(self):
        """Reset portfolio to starting state."""
        self.current_step = 0
        self.portfolio_value = self.amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs()

    def _get_obs(self):
        """Construct observation vector."""
        norm_value = self.portfolio_value / self.amount
        regime_onehot = np.zeros(self.num_regimes, dtype=np.float32)
        idx = min(self.current_step, len(self.regimes) - 1)
        regime_onehot[int(self.regimes[idx])] = 1.0

        obs = np.concatenate([
            np.asarray(self.mu, dtype=np.float32).ravel(),
            np.asarray(np.diag(self.cov), dtype=np.float32).ravel(),
            np.asarray(self.prev_weights, dtype=np.float32).ravel(),
            np.asarray([norm_value], dtype=np.float32),
            np.asarray(self.micro_indicators[idx], dtype=np.float32).ravel(),
            np.asarray(self.macro_indicators[idx], dtype=np.float32).ravel(),
            regime_onehot
        ], axis=0).astype(np.float32)

        return obs.reshape(-1)

    def step(self, action):
        """Take a step in the environment using the given allocation."""
        self.current_step += 1

        #Normalizing weights
        weights = np.clip(action, 0, 1)
        total = np.sum(weights)
        weights = np.ones_like(weights) / len(weights) if total < 1e-8 else weights / total

        #Computing reward = return - risk penalty
        expected_return = np.dot(weights, self.mu)
        variance = np.dot(weights.T, np.dot(self.cov, weights))
        risk_penalty = self.risk * variance

        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + expected_return)

        log_return = np.log(self.portfolio_value / prev_value + 1e-8)
        reward = log_return - risk_penalty

        done = self.current_step >= self.duration
        self.prev_weights = weights

        return self._get_obs(), reward, done, {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "log_return": log_return,
            "risk_penalty": risk_penalty,
            "reward": reward
        }

    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


#Main execution block
if __name__ == "__main__":
    #Loading each sub-agent's data
    stock_data = load_agent_data("stock")
    crypto_data = load_agent_data("crypto")
    etf_data = load_agent_data("etf")

    #Unpacking each dataset
    tickers_stock, prices_stock, regimes_stock, micro_stock, macro_stock, ppo_stock, scaler_stock, macro_scaler_stock = stock_data
    tickers_crypto, prices_crypto, regimes_crypto, micro_crypto, macro_crypto, ppo_crypto, scaler_crypto, macro_scaler_crypto = crypto_data
    tickers_etf, prices_etf, regimes_etf, micro_etf, macro_etf, ppo_etf, scaler_etf, macro_scaler_etf = etf_data

    #Creating environments for each agent
    env_stock = AgentEnv(prices_stock, regimes_stock, micro_stock, macro_stock)
    env_crypto = AgentEnv(prices_crypto, regimes_crypto, micro_crypto, macro_crypto)
    env_etf = AgentEnv(prices_etf, regimes_etf, micro_etf, macro_etf)

    #Debugging info
    print("Prices Stock sample:", prices_stock[:5])
    print("Prices Crypto sample:", prices_crypto[:5])
    print("Prices ETF sample:", prices_etf[:5])

    print("Stock env obs_space:", env_stock.observation_space.shape)
    print("Crypto env obs_space:", env_crypto.observation_space.shape)
    print("ETF env obs_space:", env_etf.observation_space.shape)
    print("PPO-stock model obs_space:", ppo_stock.observation_space.shape)
    print("PPO-crypto model obs_space:", ppo_crypto.observation_space.shape)
    print("PPO-etf model obs_space:", ppo_etf.observation_space.shape)

    #Gettting portfolios from each sub-agent
    mu_stock, cov_stock, w_stock = get_agent_portfolio(ppo_stock, env_stock)
    mu_crypto, cov_crypto, w_crypto = get_agent_portfolio(ppo_crypto, env_crypto)
    mu_etf, cov_etf, w_etf = get_agent_portfolio(ppo_etf, env_etf)

    print("Stock expected returns:", mu_stock)
    print("Crypto expected returns:", mu_crypto)
    print("ETF expected returns:", mu_etf)

    #Combining expected returns into a single vector
    mu_all = np.concatenate([mu_stock, mu_crypto, mu_etf])

    #Building block covariance matrix
    cov_all = np.block([
        [cov_stock, np.zeros((len(mu_stock), len(mu_crypto))), np.zeros((len(mu_stock), len(mu_etf)))],
        [np.zeros((len(mu_crypto), len(mu_stock))), cov_crypto, np.zeros((len(mu_crypto), len(mu_etf)))],
        [np.zeros((len(mu_etf), len(mu_stock))), np.zeros((len(mu_etf), len(mu_crypto))), cov_etf]
    ])

    #Aligning micro/macro indicator lengths
    min_rows = min(
        micro_stock.shape[0], micro_crypto.shape[0], micro_etf.shape[0],
        macro_stock.shape[0], macro_crypto.shape[0], macro_etf.shape[0]
    )
    micro_stock = micro_stock[:min_rows]
    micro_crypto = micro_crypto[:min_rows]
    micro_etf = micro_etf[:min_rows]

    macro_stock = macro_stock[:min_rows]
    macro_crypto = macro_crypto[:min_rows]
    macro_etf = macro_etf[:min_rows]

    #Combining indicators
    combined_micro = np.hstack([micro_stock, micro_crypto, micro_etf])
    combined_macro = np.hstack([macro_stock, macro_crypto, macro_etf])

    #Combining regimes by majority vote
    combined_regimes_result = mode(np.vstack([regimes_stock[:min_rows], regimes_crypto[:min_rows], regimes_etf[:min_rows]]), axis=0)
    combined_regimes = combined_regimes_result.mode.squeeze()

    print("Micro indicators shape:", combined_micro.shape)
    print("Macro indicators shape:", combined_macro.shape)

    #Simulation parameters
    amount = 100000
    risk = 0.5
    duration = 252  #1 year of trading days

    #Creating meta-portfolio environment
    meta_env = MetaPortfolioEnv(
        mu=mu_all,
        cov=cov_all,
        amount=amount,
        risk=risk,
        duration=duration,
        micro_indicators=combined_micro,
        macro_indicators=combined_macro,
        regimes=combined_regimes
    )
    print("Meta Env observation space:", meta_env.observation_space.shape)
    meta_env.seed(42)

    #Training SAC meta-agent
    meta_model = SAC(
        "MlpPolicy",
        meta_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=128,
        learning_starts=500,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        ent_coef='auto_0.1',
        target_entropy='auto',
        use_sde=True,
        policy_kwargs=dict(log_std_init=-2, net_arch=[128, 128]),
    )

    meta_model.set_random_seed(42)
    meta_model.learn(total_timesteps=200000, progress_bar=True)

    #Running final allocation
    obs = meta_env.reset()
    action, _ = meta_model.predict(obs)
    weights = np.clip(action, 0, 1)
    weights /= weights.sum()

    #Computing performance metrics
    daily_return = np.dot(weights, mu_all)
    yearly_return = (1 + daily_return) ** 252 - 1
    valuation = amount * ((1 + daily_return) ** duration)
    profit = valuation - amount
    allocations = weights * amount

    #Displaying results
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

    #Saving trained meta-model
    meta_model.save("sac_meta_model")
