"""
FinRL Demo Step 2: Train PPO model on A-share data (Tushare)
Uses env_stocktrading_np which accepts simple arrays
"""

import os
import pandas as pd
import numpy as np
from finrl.config import INDICATORS
from finrl.main import check_and_make_directories

# 固定输出到项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from stable_baselines3.common.logger import configure

os.environ["NO_PROXY"] = "*"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    os.environ.pop(k, None)
import urllib.request
urllib.request.getproxies = lambda: {}

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

print("=" * 50)
print("[Step 2] Training model")
print("=" * 50)

# Read data
train = pd.read_csv("demo_train_data.csv")
print(f"Train data: {len(train)} rows")
print(f"Columns: {list(train.columns)}")
print(f"Tickers: {train['tic'].unique()}")

# Convert to numpy arrays for env_stocktrading_np
# Need: price_array, tech_array, turbulence_array
# Pivot: rows=dates, columns=tickers
stock_list = sorted(train["tic"].unique().tolist())

# Price array: (num_dates, num_stocks)
price_df = train.pivot_table(index="date", columns="tic", values="close")
price_df = price_df[stock_list]  # ensure column order
price_array = price_df.values.astype(np.float32)

# Tech indicator array: (num_dates, num_stocks * num_indicators)
tech_cols = []
for indicator in INDICATORS:
    if indicator in train.columns:
        tech_cols.append(indicator)

tech_arrays = []
for tic in stock_list:
    tic_data = train[train["tic"] == tic].set_index("date").sort_index()
    for indicator in tech_cols:
        if indicator in tic_data.columns:
            tech_arrays.append(tic_data[indicator].values)

# Stack: each row has indicators for all stocks
tech_data = {}
for indicator in tech_cols:
    pivot = train.pivot_table(index="date", columns="tic", values=indicator)
    pivot = pivot[stock_list]
    tech_data[indicator] = pivot.values.astype(np.float32)

tech_array = np.hstack(list(tech_data.values())).astype(np.float32)

# Turbulence array (fake, since we turned it off)
turbulence_array = np.zeros(len(price_df), dtype=np.float32)

print(f"\nPrice array shape: {price_array.shape}")
print(f"Tech array shape: {tech_array.shape}")
print(f"Stocks: {stock_list}")

# Create environment
env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": True,
}

e_train_gym = StockTradingEnv(config=env_config)
print(f"Env created: state_dim={e_train_gym.state_dim}, action_dim={e_train_gym.action_dim}")

# Train PPO using stable-baselines3
print("\nTraining PPO model (20000 steps)...", flush=True)

from stable_baselines3 import PPO

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

model = PPO(
    "MlpPolicy",
    e_train_gym,
    verbose=1,
    tensorboard_log=RESULTS_DIR + "/ppo",
    **PPO_PARAMS,
)

model.learn(total_timesteps=20000)
model.save(TRAINED_MODEL_DIR + "/agent_ppo")

print(f"\nOK! Model saved to: {TRAINED_MODEL_DIR}/agent_ppo")
