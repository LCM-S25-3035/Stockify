#Import necessary libraries
import numpy as np
import json
import logging
from stable_baselines3 import SAC
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
from scipy.stats import mode
import os
import gym
from gym import spaces

#Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Helper to normalize ticker symbols
def normalize_ticker(ticker):
    """Normalize ticker symbols by removing hyphens and slashes."""
    return ticker.replace("-", "").replace("/", "")

#Load agent data
def load_agent_data(agent_name):
    """Load agent-specific data from files.
    
    Args:
        agent_name (str): Name of the agent ('stock', 'crypto', or 'etf')
    
    Returns:
        tuple: Tickers, prices, regimes, micro/macro indicators
    """
    logger.info(f"Loading data for {agent_name}...")
    files = [
        f"{agent_name}_tickers.json",
        f"prices_{agent_name}.npy",
        f"regime_{agent_name}.npy",
        f"micro_indicators_{agent_name}.npy",
        f"macro_indicators_{agent_name}.npy",
    ]
    for file in files:
        if not os.path.exists(file):
            logger.error(f"Missing file for {agent_name}: {file}")
            raise FileNotFoundError(f"Missing file for {agent_name}: {file}")
    with open(f"{agent_name}_tickers.json", "r") as f:
        tickers = json.load(f)
    prices = np.load(f"prices_{agent_name}.npy")
    if np.any(prices <= 0) or np.any(np.isnan(prices)):
        raise ValueError(f"Invalid prices for {agent_name}")
    regimes = np.load(f"regime_{agent_name}.npy")
    micro_indicators = np.load(f"micro_indicators_{agent_name}.npy")
    macro_indicators = np.load(f"macro_indicators_{agent_name}.npy")
    logger.info(f"Data for {agent_name} loaded successfully")
    return tickers, prices, regimes, micro_indicators, macro_indicators

#Meta Portfolio Environment
class MetaPortfolioEnv(gym.Env):
    """Gym environment for the SAC meta-agent to optimize portfolio across all assets."""
    def __init__(self, mu, cov, amount, risk, duration, micro_indicators, macro_indicators, regimes, num_regimes=3, transaction_fee=0.001):
        """Initialize the meta portfolio environment.
        
        Args:
            mu (np.ndarray): Expected returns for all assets
            cov (np.ndarray): Covariance matrix for all assets
            amount (float): Initial investment amount
            risk (float): Risk appetite (0 to 1)
            duration (int): Number of days to simulate
            micro_indicators (np.ndarray): Combined micro indicators
            macro_indicators (np.ndarray): Combined macro indicators
            regimes (np.ndarray): Combined market regimes
            num_regimes (int): Number of market regimes
            transaction_fee (float): Fee per transaction as a proportion of turnover (default: 0.1%)
        """
        super(MetaPortfolioEnv, self).__init__()
        lw = LedoitWolf()
        self.cov = lw.fit(cov).covariance_ if cov.shape[0] > 1 else cov + np.eye(cov.shape[0]) * 1e-6
        self.mu = mu
        self.amount = amount
        self.risk = risk
        self.duration = duration
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.regimes = regimes
        self.num_regimes = num_regimes
        self.n_assets = len(mu)
        self.transaction_fee = transaction_fee
        obs_dim = self.n_assets * 3 + 1 + self.micro_indicators.shape[1] + self.macro_indicators.shape[1] + self.num_regimes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.current_step = 0
        self.portfolio_value = amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.allocation_history = []

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.allocation_history = []
        return self._get_obs()

    def _get_obs(self):
        """Generate observation for the current step."""
        norm_value = self.portfolio_value / self.amount
        regime_onehot = np.zeros(self.num_regimes)
        idx = min(self.current_step, len(self.regimes) - 1)
        regime_onehot[self.regimes[idx]] = 1
        obs = np.concatenate([
            self.mu,
            np.diag(self.cov),
            self.prev_weights,
            [norm_value],
            self.micro_indicators[idx],
            self.macro_indicators[idx],
            regime_onehot
        ])
        return obs.astype(np.float32)

    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action (np.ndarray): Portfolio weights from SAC agent
        
        Returns:
            tuple: Next observation, reward, done flag, and info dictionary
        """
        self.current_step += 1
        weights = np.clip(action, 0, 1)
        weights = np.clip(weights, 0, 0.2)
        total = np.sum(weights)
        if total < 1e-8:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= total
        expected_return = np.dot(weights, self.mu)
        expected_return = np.clip(expected_return, -0.002, 0.002)
        variance = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))
        prev_value = self.portfolio_value
        #Calculate transaction cost based on turnover
        turnover = np.sum(np.abs(weights - self.prev_weights))
        transaction_cost = turnover * prev_value * self.transaction_fee
        #Update portfolio value with return and subtract transaction cost
        self.portfolio_value = prev_value * (1 + expected_return) - transaction_cost
        allocation = weights * self.portfolio_value
        self.allocation_history.append(allocation)
        log_return = np.log(self.portfolio_value / (prev_value + 1e-12) + 1e-12)
        log_return = np.clip(log_return, -0.002, 0.002)
        reward = log_return - self.risk * variance
        done = self.current_step >= self.duration
        self.prev_weights = weights
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": weights,
            "log_return": log_return,
            "risk_penalty": self.risk * variance,
            "reward": reward,
            "variance": variance
        }
        return self._get_obs(), reward, done, info

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

#Function to process weights with risk adjustments (from original code)
def normalize_weights(weights):
    """Normalize weights to sum to 1, ensuring non-negative values."""
    weights = np.clip(weights, 0, 1)
    total = weights.sum()
    if total < 1e-8:
        return np.ones_like(weights) / len(weights)
    return weights / total

def apply_risk_adjustment(weights, risk, stock_indices, crypto_indices):
    """Apply risk-based boost to stock or crypto weights."""
    boost_multiplier = 0.5
    if risk < 0.45 and len(stock_indices) > 0:
        weights[stock_indices] *= (1 + (1 - risk) * boost_multiplier)
    elif risk > 0.55 and len(crypto_indices) > 0:
        weights[crypto_indices] *= (1 + risk * boost_multiplier)
    return weights

def cap_weights(weights, max_weight=0.2):
    """Cap individual weights and renormalize."""
    weights = np.clip(weights, 0, max_weight)
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights

def enforce_minimum_stock(weights, stock_indices, crypto_indices, etf_indices, min_stock_weight=0.5):
    """Ensure minimum stock weight by scaling other assets."""
    stock_weight = weights[stock_indices].sum() if len(stock_indices) > 0 else 0.0
    if stock_weight >= min_stock_weight or len(stock_indices) == 0:
        return weights

    crypto_weight = weights[crypto_indices].sum() if len(crypto_indices) > 0 else 0.0
    etf_weight = weights[etf_indices].sum() if len(etf_indices) > 0 else 0.0
    total_other = crypto_weight + etf_weight

    if total_other > 0:
        diff = min_stock_weight - stock_weight
        scale = max((total_other - diff) / (total_other + 1e-12), 0.0)
        if len(crypto_indices) > 0:
            weights[crypto_indices] *= scale
        if len(etf_indices) > 0:
            weights[etf_indices] *= scale
    else:
        weights = _reset_weights(weights, stock_indices, crypto_indices, etf_indices)

    new_stock_weight = weights[stock_indices].sum() if len(stock_indices) > 0 else 0.0
    if new_stock_weight > 0:
        weights[stock_indices] *= min_stock_weight / (new_stock_weight + 1e-12)

    return normalize_weights(weights)

def _reset_weights(weights, stock_indices, crypto_indices, etf_indices):
    """Reset weights to prioritize stocks when no other assets have weight."""
    if len(stock_indices) > 0:
        weights[stock_indices] = 1.0
    if len(crypto_indices) > 0:
        weights[crypto_indices] = 0.0
    if len(etf_indices) > 0:
        weights[etf_indices] = 0.0
    return weights

def process_weights(action, risk, stock_len, crypto_len, etf_len):
    """Process SAC action into portfolio weights with risk-based adjustments."""
    weights = action.flatten()
    stock_indices = np.arange(stock_len)
    crypto_indices = np.arange(stock_len, stock_len + crypto_len)
    etf_indices = np.arange(stock_len + crypto_len, stock_len + crypto_len + etf_len)

    weights = normalize_weights(weights)
    weights = apply_risk_adjustment(weights, risk, stock_indices, crypto_indices)
    weights = normalize_weights(weights)
    weights = cap_weights(weights)
    weights = enforce_minimum_stock(weights, stock_indices, crypto_indices, etf_indices)
    return weights

# Main backtest function
def backtest_sac():
    """Backtest the SAC meta-model using historical price data and indicators."""
    # Load data for all agents
    _, prices_stock, regimes_stock, micro_stock, macro_stock = load_agent_data("stock")[:5]
    _, prices_crypto, regimes_crypto, micro_crypto, macro_crypto = load_agent_data("crypto")[:5]
    _, prices_etf, regimes_etf, micro_etf, macro_etf = load_agent_data("etf")[:5]

    # Find minimum number of rows across all arrays to align data
    min_rows = min([
        prices_stock.shape[0], prices_crypto.shape[0], prices_etf.shape[0],
        regimes_stock.shape[0], regimes_crypto.shape[0], regimes_etf.shape[0],
        micro_stock.shape[0], micro_crypto.shape[0], micro_etf.shape[0],
        macro_stock.shape[0], macro_crypto.shape[0], macro_etf.shape[0]
    ])
    logger.info(f"Aligning all arrays to minimum rows: {min_rows}")

    # Truncate all arrays to the last min_rows rows
    prices_stock = prices_stock[-min_rows:]
    prices_crypto = prices_crypto[-min_rows:]
    prices_etf = prices_etf[-min_rows:]
    regimes_stock = regimes_stock[-min_rows:]
    regimes_crypto = regimes_crypto[-min_rows:]
    regimes_etf = regimes_etf[-min_rows:]
    micro_stock = micro_stock[-min_rows:]
    micro_crypto = micro_crypto[-min_rows:]
    micro_etf = micro_etf[-min_rows:]
    macro_stock = macro_stock[-min_rows:]
    macro_crypto = macro_crypto[-min_rows:]
    macro_etf = macro_etf[-min_rows:]

    # Combine prices
    prices_all = np.hstack([prices_stock, prices_crypto, prices_etf])

    # Get lengths
    stock_len = prices_stock.shape[1]
    crypto_len = prices_crypto.shape[1]
    etf_len = prices_etf.shape[1]

    # Combine indicators and regimes
    combined_micro = np.hstack([micro_stock, micro_crypto, micro_etf])
    combined_macro = np.hstack([macro_stock, macro_crypto, macro_etf])
    combined_regimes_result = mode(np.vstack([regimes_stock, regimes_crypto, regimes_etf]), axis=0)
    combined_regimes = combined_regimes_result.mode.squeeze()

    # Load SAC model
    if not os.path.exists("sac_meta_model.zip"):
        raise FileNotFoundError("SAC meta model file 'sac_meta_model.zip' not found")
    meta_model = SAC.load("sac_meta_model.zip")
    logger.info("SAC meta model loaded successfully")

    # Backtest parameters
    window_size = 60
    initial_amount = 10000.0
    risk = 0.5
    transaction_fee = 0.001  # 0.1% transaction fee
    portfolio_values = [initial_amount]
    daily_returns = []

    #Check if sufficient data is available
    T = prices_all.shape[0]
    if T <= window_size:
        raise ValueError(f"Insufficient data for backtesting: {T} rows available, need at least {window_size + 1}")

    #Backtest loop
    for t in range(window_size, T - 1):
        #Compute rolling mu and cov
        window_prices = prices_all[t - window_size:t]
        log_returns = np.diff(np.log(window_prices + 1e-12), axis=0)
        mu = np.mean(log_returns, axis=0)
        mu = np.clip(mu, -0.001, 0.001)
        cov = np.cov(log_returns.T) if log_returns.shape[0] > 1 else np.zeros((prices_all.shape[1], prices_all.shape[1]))

        #Create env for current step
        meta_env = MetaPortfolioEnv(
            mu=mu,
            cov=cov,
            amount=portfolio_values[-1],
            risk=risk,
            duration=1,
            micro_indicators=combined_micro,
            macro_indicators=combined_macro,
            regimes=combined_regimes,
            transaction_fee=transaction_fee
        )
        meta_env.current_step = t % min_rows  #Cycle through indicators if needed

        obs = meta_env.reset()

        #Predict with SAC
        action, _ = meta_model.predict(obs.reshape(1, -1), deterministic=True)

        #Process weights with risk adjustments
        weights = process_weights(action, risk, stock_len, crypto_len, etf_len)

        #Compute actual next return
        actual_returns = (prices_all[t + 1] - prices_all[t]) / (prices_all[t] + 1e-12)

        #Portfolio return
        portfolio_return = np.dot(weights, actual_returns)
        new_portfolio_value = portfolio_values[-1] * (1 + portfolio_return)
        portfolio_values.append(new_portfolio_value)
        daily_returns.append(portfolio_return)

        logger.info(f"Step {t}: Portfolio Return = {portfolio_return:.4f}, Value = {new_portfolio_value:.2f}")

    #Compute performance metrics
    cumulative_return = (portfolio_values[-1] / initial_amount) - 1
    mean_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)
    sharpe_ratio = mean_daily_return / std_daily_return * np.sqrt(252) if std_daily_return > 0 else 0  #Annualized Sharpe

    #Print results
    print("Backtest Results:")
    print(f"Cumulative Return: {cumulative_return * 100:.2f}%")
    print(f"Mean Daily Return: {mean_daily_return * 100:.4f}%")
    print(f"Daily Volatility: {std_daily_return * 100:.4f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    #Plot equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.savefig('backtest_equity_curve.png')
    plt.close()
    print("Equity curve saved as 'backtest_equity_curve.png'")

if __name__ == "__main__":
    backtest_sac()