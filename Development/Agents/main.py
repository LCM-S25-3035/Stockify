#Importing necessary libraries
import streamlit as st 
import numpy as np 
import joblib  
import json 
from stable_baselines3 import PPO, SAC
from scipy.stats import mode  
import gym  
from gym import spaces  
import warnings  
import pandas as pd  
import math  
import logging  
import os  
from sklearn.covariance import LedoitWolf  
import alpaca_trade_api as tradeapi  
import plotly.express as px  
import matplotlib.pyplot as plt  
import time  
warnings.filterwarnings('ignore')  

#Configuring logging to capture info and error messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Helper function to normalize ticker symbols for Alpaca API compatibility
def normalize_ticker(ticker):
    """Normalize ticker symbols by removing hyphens and slashes."""
    return ticker.replace("-", "").replace("/", "")

#Loading data and models for a specific agent (stock, crypto, or ETF)
def load_agent_data(agent_name):
    """Load agent-specific data and models from files.
    
    Args:
        agent_name (str): Name of the agent ('stock', 'crypto', or 'etf')
    
    Returns:
        tuple: Tickers, prices, regimes, micro/macro indicators, PPO model, and scalers
    """
    logger.info(f"Loading data for {agent_name}...")
    #Listting of required files for the agent
    files = [
        f"{agent_name}_tickers.json",
        f"prices_{agent_name}.npy",
        f"regime_{agent_name}.npy",
        f"micro_indicators_{agent_name}.npy",
        f"macro_indicators_{agent_name}.npy",
        f"ppo_{agent_name}_model.zip",
        f"indicator_scaler_{agent_name}.pkl",
        f"macro_scaler_{agent_name}.pkl"
    ]
    #Checking if all required files exist
    for file in files:
        if not os.path.exists(file):
            logger.error(f"Missing file for {agent_name}: {file}")
            raise FileNotFoundError(f"Missing file for {agent_name}: {file}")
    
    #Loading ticker list from JSON
    logger.info(f"Loading {agent_name}_tickers.json")
    with open(f"{agent_name}_tickers.json", "r") as f:
        tickers = json.load(f)
    
    #Loading price data and validate
    logger.info(f"Loading prices_{agent_name}.npy")
    prices = np.load(f"prices_{agent_name}.npy")
    if np.any(prices <= 0) or np.any(np.isnan(prices)):
        logger.error(f"Invalid prices for {agent_name}: contains negative or NaN values")
        raise ValueError(f"Invalid prices for {agent_name}: contains negative or NaN values")
    
    #Loading regime, micro, and macro indicators
    logger.info(f"Loading regime_{agent_name}.npy")
    regimes = np.load(f"regime_{agent_name}.npy")
    logger.info(f"Loading micro_indicators_{agent_name}.npy")
    micro_indicators = np.load(f"micro_indicators_{agent_name}.npy")
    logger.info(f"Loading macro_indicators_{agent_name}.npy")
    macro_indicators = np.load(f"macro_indicators_{agent_name}.npy")
    
    #Loading PPO model and scalers
    logger.info(f"Loading ppo_{agent_name}_model.zip")
    model = PPO.load(f"ppo_{agent_name}_model.zip")
    logger.info(f"Loading indicator_scaler_{agent_name}.pkl")
    scaler_indicator = joblib.load(f"indicator_scaler_{agent_name}.pkl")
    logger.info(f"Loading macro_scaler_{agent_name}.pkl")
    scaler_macro = joblib.load(f"macro_scaler_{agent_name}.pkl")
    
    logger.info(f"Data for {agent_name} loaded successfully")
    return tickers, prices, regimes, micro_indicators, macro_indicators, model, scaler_indicator, scaler_macro

#Custom Gym environment for PPO agents
class AgentEnv(gym.Env):
    """Gym environment for individual PPO agents managing a portfolio."""
    def __init__(self, prices, regimes, micro_indicators, macro_indicators,
                 initial_amount=10000, risk_appetite=0.5, transaction_fee=0.001):
        """Initialize the environment with price data and indicators.
        
        Args:
            prices (np.ndarray): Historical price data for assets
            regimes (np.ndarray): Market regime indicators
            micro_indicators (np.ndarray): Micro-level indicators
            macro_indicators (np.ndarray): Macro-level indicators
            initial_amount (float): Initial portfolio value
            risk_appetite (float): Risk preference (0 to 1)
            transaction_fee (float): Fee per transaction
        """
        super(AgentEnv, self).__init__()
        self.prices = prices
        self.regimes = regimes
        self.micro_indicators = micro_indicators
        self.macro_indicators = macro_indicators
        self.initial_amount = initial_amount
        self.risk_appetite = risk_appetite
        self.transaction_fee = transaction_fee
        self.n_assets = prices.shape[1]  #Number of assets
        self.num_regimes = 3  #Number of market regimes
        self.current_step = 0  #Current time step
        #Defining observation space (prices, portfolio value, risk, regimes, indicators)
        obs_shape = self.n_assets + 2 + self.num_regimes + micro_indicators.shape[1] + macro_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)  # Portfolio weights

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        return self._next_observation()

    def _next_observation(self):
        """Generate the observation for the current step."""
        prices_part = self.prices[self.current_step] / (self.prices[0] + 1e-12)  #Normalize prices
        value_part = np.array([self.portfolio_value / self.initial_amount])  #Normalize portfolio value
        risk_part = np.array([self.risk_appetite])  #Risk appetite
        regime_onehot = np.zeros(self.num_regimes)  #One-hot encode regime
        regime_onehot[self.regimes[self.current_step]] = 1
        #Combine all observation components
        obs = np.concatenate([
            prices_part,
            value_part,
            risk_part,
            regime_onehot,
            self.micro_indicators[self.current_step],
            self.macro_indicators[self.current_step]
        ])
        return obs.astype(np.float32)

    def get_covariance_and_return(self, window_size=60):
        """Calculate expected returns and covariance matrix for a time window.
        
        Args:
            window_size (int): Number of time steps for calculation
        
        Returns:
            tuple: Expected returns (mu) and covariance matrix (cov)
        """
        step = self.current_step
        start = max(0, step - window_size)
        window_prices = self.prices[start:step + 1]
        if len(window_prices) < 2:
            returns = np.zeros((1, self.n_assets))
        else:
            returns = np.diff(np.log(window_prices + 1e-12), axis=0)  #Log returns
        mu = np.mean(returns, axis=0)  #Expected returns
        mu = np.clip(mu, -0.001, 0.001)
        cov = np.cov(returns.T) if returns.shape[0] > 1 else np.zeros((self.n_assets, self.n_assets))  #Covariance matrix
        logger.info(f"get_covariance_and_return: mu={mu}, cov_diagonal={np.diag(cov)}")
        return mu, cov

#Extracting portfolio weights and statistics for an agent
def get_agent_portfolio(ppo_model, env, window_size=60):
    """Use PPO model to generate portfolio weights and statistics.
    
    Args:
        ppo_model: Trained PPO model
        env: AgentEnv instance
        window_size (int): Time window for covariance calculation
    
    Returns:
        tuple: Expected returns, covariance matrix, and portfolio weights
    """
    start_time = time.time()  #Start timing for entire function
    env.reset()
    env.current_step = min(window_size, env.prices.shape[0] - 1)  #Set step for recent data
    obs = env._next_observation().astype(np.float32)
    expected_len = int(np.prod(ppo_model.observation_space.shape))
    cur_len = obs.shape[0]
    #Adjusting observation size to match model expectations
    if cur_len < expected_len:
        obs = np.concatenate([obs, np.zeros(expected_len - cur_len, dtype=np.float32)])
    elif cur_len > expected_len:
        obs = obs[:expected_len]
    obs = obs.reshape(1, -1)
    
    #Measuring PPO model prediction time
    ppo_start_time = time.time()
    action, _ = ppo_model.predict(obs, deterministic=True)
    ppo_inference_time = time.time() - ppo_start_time
    logger.info(f"PPO model inference time: {ppo_inference_time:.4f} seconds")
    
    #Normalizing portfolio weights
    weights = action.flatten()
    weights_sum = weights.sum()
    if weights_sum < 1e-8:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights_sum
    mu, cov = env.get_covariance_and_return(window_size=window_size)  #Get returns and covariance
    
    logger.info(f"Agent Portfolio - mu: {mu}, weights: {weights}")
    total_time = time.time() - start_time
    logger.info(f"Total get_agent_portfolio execution time: {total_time:.4f} seconds")
    
    return mu, cov, weights

#Meta Portfolio Environment for SAC meta-agent
class MetaPortfolioEnv(gym.Env):
    """Gym environment for the SAC meta-agent to optimize portfolio across all assets."""
    def __init__(self, mu, cov, amount, risk, duration, micro_indicators, macro_indicators, regimes, num_regimes=3):
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
        """
        super(MetaPortfolioEnv, self).__init__()
        #Regularizing covariance matrix using Ledoit-Wolf shrinkage
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

        #Defining observation space
        obs_dim = self.n_assets * 3 + 1 + self.micro_indicators.shape[1] + self.macro_indicators.shape[1] + self.num_regimes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.current_step = 0
        self.portfolio_value = amount
        self.prev_weights = np.ones(self.n_assets) / self.n_assets  #Initial uniform weights
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
        weights = np.clip(action, 0, 1)  #Ensuring weights are non-negative
        weights = np.clip(weights, 0, 0.2)  #Capping individual weights at 20%
        total = np.sum(weights)
        if total < 1e-8:
            weights = np.ones_like(weights) / len(weights)  #Uniform weights if sum is near zero
        else:
            weights /= total
        expected_return = np.dot(weights, self.mu)  #Calculating portfolio return
        expected_return = np.clip(expected_return, -0.002, 0.002)
        variance = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))  #Portfolio risk
        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + expected_return)  #Updating portfolio value
        allocation = weights * self.portfolio_value
        self.allocation_history.append(allocation)
        log_return = np.log(self.portfolio_value / (prev_value + 1e-12) + 1e-12)  # Log return
        log_return = np.clip(log_return, -0.002, 0.002)
        logger.info(f"Step {self.current_step}: log_return={log_return:.6f}, portfolio_value={self.portfolio_value:.2f}, expected_return={expected_return:.6f}")
        reward = log_return - self.risk * variance  #Reward: return minus risk penalty
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

#Helper function to get last prices for order submission
def _get_last_prices_for_tickers(prices_stock, prices_crypto, prices_etf, tickers_stock, tickers_crypto, tickers_etf):
    """Extract last prices for all tickers, handling missing or invalid data.
    
    Args:
        prices_stock, prices_crypto, prices_etf (np.ndarray): Price data for each asset class
        tickers_stock, tickers_crypto, tickers_etf (list): Ticker lists for each asset class
    
    Returns:
        tuple: All tickers, last prices, and normalized crypto tickers
    """
    norm_tickers_stock = [normalize_ticker(t) for t in tickers_stock]
    norm_tickers_crypto = [normalize_ticker(t) for t in tickers_crypto]
    norm_tickers_etf = [normalize_ticker(t) for t in tickers_etf]
    all_tickers = norm_tickers_stock + norm_tickers_crypto + norm_tickers_etf
    last_prices = []
    #Extracting last prices for stocks
    if prices_stock is not None and len(prices_stock) > 0 and len(prices_stock[-1]) == len(tickers_stock):
        last_prices.extend(prices_stock[-1])
    else:
        logger.warning(f"Invalid stock prices: {len(prices_stock[-1]) if prices_stock is not None else 0} prices, {len(tickers_stock)} tickers")
        last_prices.extend([0.0] * len(tickers_stock))
    #Extracting last prices for crypto
    if prices_crypto is not None and len(prices_crypto) > 0 and len(prices_crypto[-1]) == len(tickers_crypto):
        last_prices.extend(prices_crypto[-1])
    else:
        logger.warning(f"Invalid crypto prices: {len(prices_crypto[-1]) if prices_crypto is not None else 0} prices, {len(tickers_crypto)} tickers")
        last_prices.extend([0.0] * len(tickers_crypto))
    #Extracting last prices for ETFs
    if prices_etf is not None and len(prices_etf) > 0 and len(prices_etf[-1]) == len(tickers_etf):
        last_prices.extend(prices_etf[-1])
    else:
        logger.warning(f"Invalid ETF prices: {len(prices_etf[-1]) if prices_etf is not None else 0} prices, {len(tickers_etf)} tickers")
        last_prices.extend([0.0] * len(tickers_etf))
    last_prices = np.array(last_prices)
    #Validating ticker-price alignment
    if len(all_tickers) != len(last_prices):
        logger.error(f"Mismatch: {len(all_tickers)} tickers, {len(last_prices)} prices")
        st.error(f"Price data mismatch: {len(all_tickers)} tickers but {len(last_prices)} prices. Some orders may be skipped.")
    return all_tickers, last_prices, norm_tickers_crypto

#Main application function
def main():
    """Main function for the Streamlit portfolio allocation app."""
    #Initializing session state for persistent data
    if "paper_trading_enabled" not in st.session_state:
        st.session_state["paper_trading_enabled"] = False
    if "alpaca_api_key" not in st.session_state:
        st.session_state["alpaca_api_key"] = ""
    if "alpaca_secret_key" not in st.session_state:
        st.session_state["alpaca_secret_key"] = ""
    if "alpaca_base_url" not in st.session_state:
        st.session_state["alpaca_base_url"] = "https://paper-api.alpaca.markets"
    if "alpaca_client" not in st.session_state:
        st.session_state["alpaca_client"] = None
    if "allocation_log" not in st.session_state:
        st.session_state["allocation_log"] = None
    if "all_tickers" not in st.session_state:
        st.session_state["all_tickers"] = None
    if "display_tickers" not in st.session_state:
        st.session_state["display_tickers"] = None
    if "returns" not in st.session_state:
        st.session_state["returns"] = None
    if "summary_df" not in st.session_state:
        st.session_state["summary_df"] = None
    if "alloc_df" not in st.session_state:
        st.session_state["alloc_df"] = None

    #Configuring Streamlit page
    st.set_page_config(page_title="Meta Portfolio + Alpaca Paper Trading", layout="wide")
    st.title("Meta Portfolio Allocation with SAC Meta-Agent + Alpaca Paper Trading")

    #Sidebar: Paper trading controls
    st.sidebar.header("Paper Trading (Alpaca)")
    enable_checkbox = st.sidebar.checkbox("Enable Paper Trading", value=st.session_state["paper_trading_enabled"])
    st.session_state["paper_trading_enabled"] = enable_checkbox

    if st.session_state["paper_trading_enabled"]:
        st.sidebar.markdown("**Alpaca Paper Account Keys** (kept in session only)")
        st.session_state["alpaca_api_key"] = st.sidebar.text_input("API Key", value=st.session_state["alpaca_api_key"])
        st.session_state["alpaca_secret_key"] = st.sidebar.text_input("Secret Key", value=st.session_state["alpaca_secret_key"], type="password")
        st.session_state["alpaca_base_url"] = st.sidebar.text_input("Base URL", value=st.session_state["alpaca_base_url"])
        if st.sidebar.button("Save & Connect to Alpaca"):
            if tradeapi is None:
                st.sidebar.error("alpaca-trade-api package not installed. Run `pip install alpaca-trade-api`.")
            elif (not st.session_state["alpaca_api_key"]) or (not st.session_state["alpaca_secret_key"]):
                st.sidebar.error("Please fill both API Key and Secret.")
            else:
                try:
                    #Initializing Alpaca API client
                    client = tradeapi.REST(
                        st.session_state["alpaca_api_key"],
                        st.session_state["alpaca_secret_key"],
                        base_url=st.session_state["alpaca_base_url"],
                        api_version='v2'
                    )
                    account = client.get_account()
                    st.sidebar.success(f"Connected to Alpaca (status: {account.status}, Buying Power: ${float(account.buying_power):,.2f})")
                    st.session_state["alpaca_client"] = client
                except Exception as e:
                    logger.error(f"Alpaca connection failed: {e}")
                    st.sidebar.error(f"Failed to connect: {e}")
                    st.session_state["alpaca_client"] = None
        if st.sidebar.button("Reset Session State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.sidebar.success("Session state reset. Please re-enter Alpaca keys if needed.")
    else:
        st.session_state["alpaca_client"] = None

    #Loading model/data in the left column
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Simulation Inputs")
        st.write(f"Current working directory: {os.getcwd()}")
        try:
            #Loading data for stock, crypto, and ETF agents
            logger.info("Loading stock data...")
            stock_data = load_agent_data("stock")
            logger.info("Loading crypto data...")
            crypto_data = load_agent_data("crypto")
            logger.info("Loading ETF data...")
            etf_data = load_agent_data("etf")
            tickers_stock, prices_stock, regimes_stock, micro_stock, macro_stock, ppo_stock, scaler_stock, macro_scaler_stock = stock_data
            tickers_crypto, prices_crypto, regimes_crypto, micro_crypto, macro_crypto, ppo_crypto, scaler_crypto, macro_scaler_crypto = crypto_data
            tickers_etf, prices_etf, regimes_etf, micro_etf, macro_etf, ppo_etf, scaler_etf, macro_scaler_etf = etf_data

            #Storing original tickers for display
            st.session_state["display_tickers"] = tickers_stock + tickers_crypto + tickers_etf

            #Normalizing tickers for Alpaca
            tickers_stock = [normalize_ticker(t) for t in tickers_stock]
            tickers_crypto = [normalize_ticker(t) for t in tickers_crypto]
            tickers_etf = [normalize_ticker(t) for t in tickers_etf]
            load_error = None
            logger.info("All data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models/data: {e}")
            st.error(f"Error loading models/data: {e}")
            st.write("Please ensure all required files (tickers.json, prices.npy, regime.npy, micro_indicators.npy, macro_indicators.npy, ppo_model.zip, scalers.pkl) are in the correct directory.")
            tickers_stock = tickers_crypto = tickers_etf = []
            prices_stock = prices_crypto = prices_etf = np.array([])
            regimes_stock = regimes_crypto = regimes_etf = np.array([])
            micro_stock = micro_crypto = micro_etf = np.array([[]])
            macro_stock = macro_crypto = macro_etf = np.array([[]])
            ppo_stock = ppo_crypto = ppo_etf = None
            st.session_state["display_tickers"] = []
            load_error = e

        #Input controls for simulation parameters
        amount = st.number_input("Total Investment Amount ($)", min_value=1000.0, max_value=10_000_000.0, value=10000.0, step=1000.0, format="%.2f")
        risk = st.slider("Risk Appetite (0: low - 1: high)", 0.0, 1.0, 0.5, 0.01)
        duration = st.number_input("Investment Duration (days)", min_value=1, max_value=3650, value=30, step=1)
        run_button = st.button("Run Portfolio Allocation")

    #Simulation output containers in the right column
    with col2:
        st.subheader("Simulation Output")
        summary_container = st.container()
        alloc_table_container = st.container()
        trends_plot_container = st.container()
        pie_charts_container = st.container()

    #Running allocation and simulation when button is clicked
    if run_button:
        if load_error:
            st.error("Cannot run allocation due to load error. Fix model/data loading first.")
        elif amount <= 0:
            st.error("Investment amount must be positive.")
        else:
            with st.spinner("Running portfolio allocation..."):
                progress_bar = st.progress(0)
                #Initializing environments for each agent
                env_stock = AgentEnv(prices_stock, regimes_stock, micro_stock, macro_stock, risk_appetite=risk)
                env_crypto = AgentEnv(prices_crypto, regimes_crypto, micro_crypto, macro_crypto, risk_appetite=risk)
                env_etf = AgentEnv(prices_etf, regimes_etf, micro_etf, macro_etf, risk_appetite=risk)

                #Getting portfolio statistics from PPO models
                mu_stock, cov_stock, w_stock = get_agent_portfolio(ppo_stock, env_stock)
                mu_crypto, cov_crypto, w_crypto = get_agent_portfolio(ppo_crypto, env_crypto)
                mu_etf, cov_etf, w_etf = get_agent_portfolio(ppo_etf, env_etf)


                #Combining expected returns
                mu_all = np.concatenate([mu_stock, mu_crypto, mu_etf])
                logger.info(f"Combined mu_all: {mu_all}")
                #Constructing block-diagonal covariance matrix
                cov_all = np.block([
                    [cov_stock, np.zeros((len(mu_stock), len(mu_crypto))), np.zeros((len(mu_stock), len(mu_etf)))],
                    [np.zeros((len(mu_crypto), len(mu_stock))), cov_crypto, np.zeros((len(mu_crypto), len(mu_etf)))],
                    [np.zeros((len(mu_etf), len(mu_stock))), np.zeros((len(mu_etf), len(mu_crypto))), cov_etf]
                ])

                #Finding minimum number of rows for indicators
                min_rows = min(
                    micro_stock.shape[0] if hasattr(micro_stock, "shape") else 0,
                    micro_crypto.shape[0] if hasattr(micro_crypto, "shape") else 0,
                    micro_etf.shape[0] if hasattr(micro_etf, "shape") else 0,
                    macro_stock.shape[0] if hasattr(macro_stock, "shape") else 0,
                    macro_crypto.shape[0] if hasattr(macro_crypto, "shape") else 0,
                    macro_etf.shape[0] if hasattr(macro_etf, "shape") else 0
                )
                if min_rows <= 0:
                    st.error("Indicator arrays have invalid shapes. Ensure micro/macro arrays have rows.")
                else:
                    #Validating indicator column consistency
                    micro_cols = [micro_stock.shape[1] if micro_stock.size > 0 else 0,
                                  micro_crypto.shape[1] if micro_crypto.size > 0 else 0,
                                  micro_etf.shape[1] if micro_etf.size > 0 else 0]
                    macro_cols = [macro_stock.shape[1] if macro_stock.size > 0 else 0,
                                  macro_crypto.shape[1] if macro_crypto.size > 0 else 0,
                                  macro_etf.shape[1] if macro_etf.size > 0 else 0]
                    if len(set(micro_cols)) > 1 or len(set(macro_cols)) > 1:
                        st.error("Inconsistent number of columns in micro/macro indicators across agents.")
                    else:
                        #Combining indicators across agents
                        combined_micro = np.hstack([
                            micro_stock[:min_rows],
                            micro_crypto[:min_rows],
                            micro_etf[:min_rows]
                        ])
                        combined_macro = np.hstack([
                            macro_stock[:min_rows],
                            macro_crypto[:min_rows],
                            macro_etf[:min_rows]
                        ])

                        #Combining regimes using mode
                        combined_regimes_result = mode(np.vstack([
                            regimes_stock[:min_rows], regimes_crypto[:min_rows], regimes_etf[:min_rows]
                        ]), axis=0)
                        combined_regimes = combined_regimes_result.mode.squeeze()

                        #Initializing meta portfolio environment
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
                        meta_env.seed(42)  #Setting seed for reproducibility
                        try:
                            #Loading SAC meta model
                            logger.info("Loading SAC meta model...")
                            if not os.path.exists("sac_meta_model.zip"):
                                raise FileNotFoundError("SAC meta model file 'sac_meta_model.zip' not found")
                            meta_model = SAC.load("sac_meta_model.zip")
                            logger.info("SAC meta model loaded successfully")
                        except Exception as e:
                            logger.error(f"Failed to load SAC meta model: {e}")
                            st.error(f"Failed to load SAC meta model: {e}")
                            st.write("Please ensure the 'sac_meta_model.zip' file is in the working directory.")
                            meta_model = None
                        if meta_model is None:
                            st.warning("Meta model not loaded; cannot simulate allocations without 'sac_meta_model.zip'.")
                        else:
                            #Running simulation with SAC meta-agent
                            obs = meta_env.reset()
                            allocation_log = []
                            returns = []
                            total_sac_inference_time = 0.0
                            sac_inference_count = 0
                            for step in range(duration):
                                obs_reshaped = obs.reshape(1, -1)
                                
                                #Measuring SAC model prediction time
                                sac_start_time = time.time()
                                action, _ = meta_model.predict(obs_reshaped, deterministic=True)
                                sac_inference_time = time.time() - sac_start_time
                                total_sac_inference_time += sac_inference_time
                                sac_inference_count += 1
                                logger.info(f"Step {step}: SAC model inference time: {sac_inference_time:.4f} seconds")
                                
                                #Processing SAC action into portfolio weights
                                weights = np.clip(action.flatten(), 0, 1)
                                weights_sum = weights.sum()
                                if weights_sum < 1e-8:
                                    weights = np.ones_like(weights) / len(weights)
                                else:
                                    weights /= weights_sum
                                #Applying risk-based asset class adjustments
                                stock_len = len(tickers_stock)
                                crypto_len = len(tickers_crypto)
                                etf_len = len(tickers_etf)
                                stock_indices = np.arange(stock_len)
                                crypto_indices = np.arange(stock_len, stock_len + crypto_len)
                                etf_indices = np.arange(stock_len + crypto_len, stock_len + crypto_len + etf_len)
                                boost_multiplier = 0.5
                                if risk < 0.45:
                                    boost = (1 - risk) * boost_multiplier
                                    if stock_len > 0:
                                        weights[stock_indices] *= (1 + boost)
                                elif risk > 0.55:
                                    boost = risk * boost_multiplier
                                    if crypto_len > 0:
                                        weights[crypto_indices] *= (1 + boost)
                                weights_sum = weights.sum()
                                if weights_sum > 0:
                                    weights /= weights_sum
                                weights = np.clip(weights, 0, 0.2)
                                weights_sum = weights.sum()
                                if weights_sum > 0:
                                    weights /= weights_sum

                                #Ensuring minimum stock allocation
                                stock_weight = weights[stock_indices].sum() if stock_len > 0 else 0.0
                                crypto_weight = weights[crypto_indices].sum() if crypto_len > 0 else 0.0
                                etf_weight = weights[etf_indices].sum() if etf_len > 0 else 0.0
                                min_stock_weight = 0.5
                                if stock_weight < min_stock_weight and stock_len > 0:
                                    diff = min_stock_weight - stock_weight
                                    total_other = crypto_weight + etf_weight
                                    if total_other > 0:
                                        scale = max((total_other - diff) / (total_other + 1e-12), 0.0)
                                        if crypto_len > 0:
                                            weights[crypto_indices] *= scale
                                        if etf_len > 0:
                                            weights[etf_indices] *= scale
                                    else:
                                        if stock_len > 0:
                                            weights[stock_indices] = 1.0
                                        if crypto_len > 0:
                                            weights[crypto_indices] = 0.0
                                        if etf_len > 0:
                                            weights[etf_indices] = 0.0
                                    new_stock_weight = weights[stock_indices].sum() if stock_len > 0 else 0.0
                                    if new_stock_weight > 0:
                                        weights[stock_indices] *= (stock_weight + diff) / (new_stock_weight + 1e-12)
                                    weights_sum = weights.sum()
                                    if weights_sum > 0:
                                        weights /= weights_sum
                                logger.info(f"Step {step}: Adjusted weights after risk adjustment: {weights}")
                                #Stepping the environment
                                obs, reward, done, info = meta_env.step(weights)
                                allocation_log.append(info["weights"] * meta_env.portfolio_value)
                                returns.append(info["log_return"])
                                progress_bar.progress((step + 1) / duration)

                            #Logging average SAC inference time
                            avg_sac_inference_time = total_sac_inference_time / sac_inference_count if sac_inference_count > 0 else 0.0
                            logger.info(f"Average SAC model inference time per step: {avg_sac_inference_time:.4f} seconds")
                            logger.info(f"Total SAC model inference time for {duration} steps: {total_sac_inference_time:.4f} seconds")
                            
                            #Storing simulation results
                            st.session_state["allocation_log"] = allocation_log
                            st.session_state["all_tickers"] = tickers_stock + tickers_crypto + tickers_etf
                            st.session_state["returns"] = returns
                            final_alloc = allocation_log[-1]
                            profit = meta_env.portfolio_value - amount

                            #Creating summary DataFrame
                            summary_df = pd.DataFrame({
                                "Metric": ["Estimating Total Investment Amount", f"Projected Valuation (after {duration} days)", "Projected Profit"],
                                "Value": [f"${amount:,.2f}", f"${meta_env.portfolio_value:,.2f}", f"${profit:,.2f}"]
                            })
                            #Creating allocation DataFrame
                            alloc_df = pd.DataFrame({
                                "Asset": st.session_state["display_tickers"],
                                "Allocation ($)": final_alloc,
                                "Allocation (%)": final_alloc / (final_alloc.sum() + 1e-12) * 100
                            })
                            st.session_state["summary_df"] = summary_df
                            st.session_state["alloc_df"] = alloc_df

    #Displaying simulation results
    if st.session_state["summary_df"] is not None:
        with summary_container:
            st.subheader("Meta Portfolio Allocation Summary")
            st.table(st.session_state["summary_df"])
    if st.session_state["alloc_df"] is not None:
        with alloc_table_container:
            st.subheader("Asset Allocations on Final Day")
            alloc_df_display = st.session_state["alloc_df"].copy()
            alloc_df_display["Allocation ($)"] = alloc_df_display["Allocation ($)"].map("${:,.2f}".format)
            alloc_df_display["Allocation (%)"] = alloc_df_display["Allocation (%)"].map("{:.2f}%".format)
            st.dataframe(alloc_df_display)
    if st.session_state["allocation_log"] is not None:
        with trends_plot_container:
            st.subheader("Daily Allocation Trends")
            alloc_log_df = pd.DataFrame(st.session_state["allocation_log"], columns=st.session_state["display_tickers"])
            alloc_log_df['Day'] = alloc_log_df.index
            alloc_log_melted = alloc_log_df.melt(id_vars=['Day'], value_vars=st.session_state["display_tickers"],
                                                 var_name='Asset', value_name='Allocation ($)')
            #Creating Plotly line plot for allocation trends
            fig_alloc = px.line(
                alloc_log_melted,
                x='Day',
                y='Allocation ($)',
                color='Asset',
                title='Daily Allocation Trends',
                labels={'Day': 'Days', 'Allocation ($)': 'Allocation ($)'},
                template='plotly_white'
            )
            fig_alloc.update_layout(
                legend=dict(
                    x=1.05,
                    y=1,
                    xanchor='left',
                    yanchor='top',
                    title='Asset'
                ),
                margin=dict(r=200),
                showlegend=True,
                xaxis_title='Days',
                yaxis_title='Allocation ($)',
                title_x=0.5
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
            st.download_button(
                label="Download Allocation Log as CSV",
                data=alloc_log_df.drop(columns=['Day']).to_csv(index_label="Day"),
                file_name="allocation_log.csv",
                mime="text/csv"
            )
    if st.session_state["allocation_log"] is not None and st.session_state["all_tickers"] is not None:
        with pie_charts_container:
            st.subheader("Asset Allocations by Category")
            def plot_pie_chart(ax, labels, sizes, title):
                """Create a pie chart for asset allocations."""
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
                centre_circle = plt.Circle((0,0),0.70,fc='white')
                ax.add_artist(centre_circle)
                ax.axis('equal')
                ax.set_title(title)
            cols = st.columns(3)
            idx = 0
            final_alloc = st.session_state["allocation_log"][-1]
            #Creating pie charts for each asset class
            for col, (agent_name, tickers_list) in zip(cols, [("Stock", tickers_stock), ("Crypto", tickers_crypto), ("ETF", tickers_etf)]):
                labels = st.session_state["display_tickers"][idx:idx + len(tickers_list)]
                sizes = final_alloc[idx: idx + len(tickers_list)]
                idx += len(tickers_list)
                total_alloc = sum(sizes)
                sizes_pct = [(s / (total_alloc + 1e-12) * 100) if total_alloc > 0 else 0 for s in sizes]
                fig, ax = plt.subplots()
                plot_pie_chart(ax, labels, sizes_pct, f"{agent_name} Allocations (%)")
                col.pyplot(fig)

    #Showing daily allocation log
    daily_log_container = st.container()
    with daily_log_container:
        if st.button("Show Daily Allocations Log"):
            if st.session_state["allocation_log"] is not None:
                all_tickers = st.session_state.get("display_tickers", [f"Asset {i}" for i in range(len(st.session_state["allocation_log"][0]))])
                alloc_log_df = pd.DataFrame(st.session_state["allocation_log"], columns=all_tickers)
                st.dataframe(alloc_log_df)
            else:
                st.info("No allocation log available. Run portfolio allocation first.")

    #Paper trade submission controls
    st.sidebar.markdown("---")
    st.sidebar.header("Submit Final Allocation to Alpaca (Paper)")
    cancel_orders = st.sidebar.checkbox("Cancel all open orders before submitting", value=False)
    if st.sidebar.button("Prepare Submit Orders"):
        if st.session_state["allocation_log"] is None:
            st.sidebar.error("No allocation_log found. Run portfolio allocation first.")
        else:
            st.sidebar.info("Ready to submit orders. Configure options below.")
    frac_allowed = st.sidebar.checkbox("Allow fractional shares", value=False)
    orders_container = st.sidebar.container()
    with orders_container:
        if st.button("Submit Paper Orders (final allocation)"):
            if st.session_state["allocation_log"] is None:
                st.error("No allocation_log found. Run portfolio allocation first.")
            elif not st.session_state["paper_trading_enabled"]:
                st.error("Paper trading disabled. Enable it in the sidebar.")
            elif tradeapi is None:
                st.error("alpaca-trade-api package not installed. Run `pip install alpaca-trade-api`.")
            elif (not st.session_state["alpaca_api_key"]) or (not st.session_state["alpaca_secret_key"]):
                st.error("API keys not set. Enter keys in the sidebar and Save & Connect.")
            elif st.session_state["alpaca_client"] is None:
                st.error("Not connected to Alpaca. Click 'Save & Connect to Alpaca' in the sidebar.")
            else:
                client = st.session_state["alpaca_client"]
                try:
                    #Checking account status and buying power
                    account = client.get_account()
                    buying_power = float(account.buying_power)
                    total_alloc = sum(st.session_state["allocation_log"][-1])
                    if buying_power < total_alloc:
                        st.error(f"Insufficient buying power: ${buying_power:,.2f} available, ${total_alloc:,.2f} needed.")
                    else:
                        #Displaying current positions
                        positions = client.list_positions()
                        if positions:
                            st.subheader("Current Positions")
                            pos_df = pd.DataFrame([(p.symbol, p.qty, p.current_price) for p in positions],
                                                  columns=["Symbol", "Quantity", "Current Price"])
                            st.dataframe(pos_df)
                        #Canceling existing orders if requested
                        if cancel_orders:
                            try:
                                client.cancel_all_orders()
                                st.success("All open orders cancelled.")
                            except Exception as e:
                                logger.error(f"Failed to cancel orders: {e}")
                                st.warning(f"Failed to cancel orders: {e}")
                        #Preparing and submit orders
                        final_alloc = np.array(st.session_state["allocation_log"][-1])
                        all_tickers, last_prices, norm_tickers_crypto = _get_last_prices_for_tickers(prices_stock, prices_crypto, prices_etf,
                                                                                                    tickers_stock, tickers_crypto, tickers_etf)
                        orders_report = []
                        for symbol, dollars, price in zip(all_tickers, final_alloc, last_prices):
                            if price is None or price == 0 or np.isnan(price):
                                orders_report.append((symbol, "skipped", "no price data"))
                                logger.warning(f"Skipping order for {symbol}: no valid price")
                                continue
                            qty = dollars / price if frac_allowed else math.floor(dollars / price)
                            if qty <= 0:
                                orders_report.append((symbol, "skipped", "quantity is 0"))
                                logger.warning(f"Skipping order for {symbol}: quantity is 0")
                                continue
                            try:
                                #Verifying asset tradability
                                asset = client.get_asset(symbol)
                                if not asset.tradable:
                                    orders_report.append((symbol, "skipped", "asset not tradable"))
                                    logger.warning(f"Skipping order for {symbol}: asset not tradable")
                                    continue
                                time_in_force = 'gtc' if symbol in norm_tickers_crypto else 'day'

                                #Submitting market order
                                order = client.submit_order(
                                    symbol=symbol,
                                    qty=float(qty) if frac_allowed else int(qty),
                                    side='buy',
                                    type='market',
                                    time_in_force=time_in_force
                                )
                                orders_report.append((symbol, "submitted", f"qty={qty:.2f}, id={order.id}"))
                                logger.info(f"Order submitted for {symbol}: qty={qty:.2f}, id={order.id}")
                            except Exception as e:
                                orders_report.append((symbol, "error", str(e)))
                                logger.error(f"Order failed for {symbol}: {e}")

                                
                        #Displaying order submission report
                        report_df = pd.DataFrame(orders_report, columns=["Symbol", "Status", "Details"])
                        st.subheader("Order Submission Report")
                        st.dataframe(report_df)
                        if report_df[report_df["Status"] == "error"].shape[0] > 0:
                            st.warning("Some orders failed. Check the report for details.")
                        elif report_df[report_df["Status"] == "submitted"].shape[0] == 0:
                            st.warning("No orders were submitted. Check asset tradability or price data.")
                        else:
                            st.success("Orders submitted successfully. See report for details.")
                except Exception as e:
                    logger.error(f"Alpaca account check failed: {e}")
                    st.error(f"Failed to access account: {e}")

if __name__ == "__main__":
    main()