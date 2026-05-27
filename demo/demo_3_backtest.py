"""
FinRL Demo Step 3: Backtest PPO model on A-share data (text output)
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from finrl.config import INDICATORS

# 固定读取项目根目录的模型
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models")
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

os.environ["NO_PROXY"] = "*"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    os.environ.pop(k, None)
import urllib.request
urllib.request.getproxies = lambda: {}

print("=" * 50)
print("[Step 3] Backtesting")
print("=" * 50)

trade = pd.read_csv("demo_trade_data.csv")
print(f"Trade data: {len(trade)} rows")
print(f"Tickers: {trade['tic'].unique()}")

stock_list = sorted(trade["tic"].unique().tolist())

price_df = trade.pivot_table(index="date", columns="tic", values="close")
price_df = price_df[stock_list]
price_array = price_df.values.astype(np.float32)

tech_cols = [i for i in INDICATORS if i in trade.columns]
tech_data = {}
for indicator in tech_cols:
    pivot = trade.pivot_table(index="date", columns="tic", values=indicator)
    pivot = pivot[stock_list]
    tech_data[indicator] = pivot.values.astype(np.float32)
tech_array = np.hstack(list(tech_data.values())).astype(np.float32)
turbulence_array = np.zeros(len(price_df), dtype=np.float32)
dates = price_df.index.tolist()

print(f"Price array: {price_array.shape}, Tech array: {tech_array.shape}")

env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": False,
}
e_test_gym = StockTradingEnv(config=env_config)

print("\nLoading PPO model...")
model = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo")

print("Running backtest...\n")
obs, info = e_test_gym.reset(seed=42)
account_values = [e_test_gym.initial_capital]
daily_returns = []
done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = e_test_gym.step(action)
    done = terminated or truncated
    total_asset = e_test_gym.amount + np.sum(e_test_gym.stocks * e_test_gym.price_ary[e_test_gym.day])
    account_values.append(total_asset)
    daily_returns.append((account_values[-1] / account_values[-2] - 1) * 100)
    step += 1

# ===== Results =====
initial = account_values[0]
final = account_values[-1]
pnl = final - initial
pnl_pct = (final / initial - 1) * 100
max_val = max(account_values)
min_val = min(account_values[1:])  # skip initial
peak_idx = account_values.index(max_val)
max_dd = 0
for i in range(1, len(account_values)):
    peak = max(account_values[:i+1])
    dd = (peak - account_values[i]) / peak * 100
    max_dd = max(max_dd, dd)

print("=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
print(f"Period:        {dates[0]} ~ {dates[-1]}")
print(f"Trading days:  {step}")
print(f"Stocks:        {stock_list}")
print(f"")
print(f"Initial:       ${initial:>12,.0f}")
print(f"Final:         ${final:>12,.0f}")
print(f"PnL:           ${pnl:>+12,.0f}")
print(f"Return:        {pnl_pct:>+11.2f}%")
print(f"Max equity:    ${max_val:>12,.0f} (day {peak_idx})")
print(f"Max drawdown:  {max_dd:>11.2f}%")
print(f"Avg daily ret: {np.mean(daily_returns):>+11.4f}%")
print(f"Daily vol:     {np.std(daily_returns):>11.4f}%")
print(f"Sharpe (ann.): {np.mean(daily_returns)/np.std(daily_returns)*np.sqrt(252):>+11.2f}" if np.std(daily_returns) > 0 else "Sharpe: N/A")

# Save daily account values to CSV
result_df = pd.DataFrame({
    "date": dates[:len(daily_returns)],
    "account_value": account_values[1:],
    "daily_return_pct": daily_returns,
})
result_df.to_csv("demo_backtest_result.csv", index=False)
print(f"\nResults saved: demo_backtest_result.csv")

# Print last 10 days
print(f"\nLast 10 trading days:")
print(f"{'Date':<14} {'Account Value':>14} {'Daily Return':>14}")
print("-" * 44)
for _, row in result_df.tail(10).iterrows():
    print(f"{row['date']:<14} ${row['account_value']:>12,.0f} {row['daily_return_pct']:>+13.3f}%")
