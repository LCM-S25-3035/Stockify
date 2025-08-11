import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from environment import FinancePortfolioEnv

# Load preprocessed financial data
df = pd.read_csv("financial_data.csv", index_col=0)
env = FinancePortfolioEnv(df)

# Best hyperparameters from Optuna (remove sigma!)
sigma = 0.2  # for action noise only
params = {
    "learning_rate": 1e-4,
    "buffer_size": 50000,
    "batch_size": 128,
    "tau": 0.01,
    "gamma": 0.99
}

# Set up action noise
noise = NormalActionNoise(mean=np.zeros(env.n_assets), sigma=sigma * np.ones(env.n_assets))

# Train TD3 model
model = TD3("MlpPolicy", env, action_noise=noise, verbose=1, **params)
model.learn(total_timesteps=20000)

# Evaluate agent
obs, _ = env.reset()
portfolio_values = [env.portfolio_value]
for _ in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    portfolio_values.append(info['portfolio_value'])
    if done or truncated:
        break

# Plot
plt.plot(portfolio_values, label='Portfolio Value')
plt.axhline(env.initial_investment, linestyle='--', color='r', label='Initial Investment')
plt.title("TD3 Portfolio Performance (Tuned)")
plt.xlabel("Step")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
