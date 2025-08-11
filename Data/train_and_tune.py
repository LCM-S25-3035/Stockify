# 3_train_and_tune.py
import pandas as pd
import optuna
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from environment import FinancePortfolioEnv

df = pd.read_csv("financial_data.csv", index_col=0)

def make_env():
    return FinancePortfolioEnv(df)

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    batch = trial.suggest_categorical("batch_size", [64, 128, 256])
    tau = trial.suggest_uniform("tau", 0.005, 0.02)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    sigma = trial.suggest_uniform("sigma", 0.05, 0.3)

    env = make_env()
    noise = NormalActionNoise(mean=[0]*6, sigma=sigma*np.ones(6))

    model = TD3("MlpPolicy", env, learning_rate=lr, buffer_size=buffer, batch_size=batch,
                tau=tau, gamma=gamma, action_noise=noise, verbose=0)
    model.learn(total_timesteps=10000)

    obs, _ = env.reset()
    total_reward = 0
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    return -total_reward

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("✅ Best Hyperparameters:", study.best_params)
