"""
FinRL Demo Step 5: Parameter Tuning Experiment
================================================
训练不同配置，对比收益，理解参数怎么影响结果

实验设计：
  1. Baseline（基线）    - PPO, lr=0.00025, 20000步（上次的结果）
  2. More Steps          - PPO, lr=0.00025, 100000步
  3. Higher LR           - PPO, lr=0.001,   100000步
  4. Lower LR            - PPO, lr=0.0001,  100000步
  5. A2C                 - A2C, lr=0.0007,  100000步
  6. SAC                 - SAC, lr=0.0001,  100000步
"""

import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

os.environ["NO_PROXY"] = "*"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    os.environ.pop(k, None)
import urllib.request
urllib.request.getproxies = lambda: {}

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. Load and prepare data
# ============================================================
print("=" * 60)
print("[Step 5] Parameter Tuning Experiment")
print("=" * 60)

train = pd.read_csv(os.path.join(DEMO_DIR, "demo_train_data.csv"))
trade = pd.read_csv(os.path.join(DEMO_DIR, "demo_trade_data.csv"))
stock_list = sorted(train["tic"].unique().tolist())

def df_to_arrays(df):
    """Convert DataFrame to numpy arrays for env_stocktrading_np"""
    price_df = df.pivot_table(index="date", columns="tic", values="close")
    price_df = price_df[stock_list]
    tech_cols = [i for i in INDICATORS if i in df.columns]
    tech_data = {}
    for ind in tech_cols:
        pivot = df.pivot_table(index="date", columns="tic", values=ind)
        pivot = pivot[stock_list]
        tech_data[ind] = pivot.values.astype(np.float32)
    price_array = price_df.values.astype(np.float32)
    tech_array = np.hstack(list(tech_data.values())).astype(np.float32)
    turbulence_array = np.zeros(len(price_df), dtype=np.float32)
    return price_array, tech_array, turbulence_array, price_df.index.tolist()

price_train, tech_train, turb_train, _ = df_to_arrays(train)
price_trade, tech_trade, turb_trade, trade_dates = df_to_arrays(trade)

print(f"Train: {price_train.shape[0]} days x {price_train.shape[1]} stocks")
print(f"Trade: {price_trade.shape[0]} days x {price_trade.shape[1]} stocks")


# ============================================================
# 2. Define experiments
# ============================================================
#
# --- 参数解读 ---
#
# total_timesteps: 训练总步数，越多学得越充分，但越慢
# learning_rate:   学习率，太大容易震荡，太小学不动
# n_steps:         每次更新前收集多少步经验（PPO专用）
# batch_size:      每次训练用的数据量
# ent_coef:        熵系数，越大探索越多，越小越保守
# reward_scaling:  奖励缩放，影响训练稳定性

EXPERIMENTS = [
    {
        "name": "1_baseline",
        "label": "Baseline (PPO 20k)",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128},
        "timesteps": 20000,
    },
    {
        "name": "2_more_steps",
        "label": "More Steps (PPO 100k)",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128},
        "timesteps": 100000,
    },
    {
        "name": "3_higher_lr",
        "label": "Higher LR (PPO lr=0.001)",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.001, "batch_size": 128},
        "timesteps": 100000,
    },
    {
        "name": "4_lower_lr",
        "label": "Lower LR (PPO lr=0.0001)",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.0001, "batch_size": 128},
        "timesteps": 100000,
    },
    {
        "name": "5_a2c",
        "label": "A2C (100k)",
        "algo": "A2C",
        "params": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
        "timesteps": 100000,
    },
    {
        "name": "6_more_explore",
        "label": "More Explore (PPO ent=0.1)",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.1, "learning_rate": 0.00025, "batch_size": 128},
        "timesteps": 100000,
    },
]


# ============================================================
# 3. Train all experiments
# ============================================================
print("\n--- Training ---\n")

ALGO_MAP = {"PPO": PPO, "A2C": A2C}
results = []

for exp in EXPERIMENTS:
    print(f"  [{exp['label']}] training {exp['timesteps']} steps...", end=" ", flush=True)
    t0 = time.time()

    env = StockTradingEnv(config={
        "price_array": price_train,
        "tech_array": tech_train,
        "turbulence_array": turb_train,
        "if_train": True,
    })

    algo_cls = ALGO_MAP[exp["algo"]]
    model = algo_cls("MlpPolicy", env, verbose=0, **exp["params"])
    model.learn(total_timesteps=exp["timesteps"])

    save_path = os.path.join(MODEL_DIR, f"exp_{exp['name']}")
    model.save(save_path)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s, saved to {save_path}")


# ============================================================
# 4. Backtest all experiments
# ============================================================
print("\n--- Backtesting ---\n")

all_account_values: Dict[str, List[float]] = {}

for exp in EXPERIMENTS:
    label = str(exp["label"])
    save_path = os.path.join(MODEL_DIR, f"exp_{exp['name']}")
    algo_cls = ALGO_MAP[exp["algo"]]
    model = algo_cls.load(save_path)

    env = StockTradingEnv(config={
        "price_array": price_trade,
        "tech_array": tech_trade,
        "turbulence_array": turb_trade,
        "if_train": False,
    })

    obs, info = env.reset(seed=42)
    account_values = [env.initial_capital]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_asset = env.amount + np.sum(env.stocks * env.price_ary[env.day])
        account_values.append(total_asset)

    initial = account_values[0]
    final = account_values[-1]
    ret_pct = (final / initial - 1) * 100

    # Max drawdown
    peak = account_values[0]
    max_dd = 0
    for v in account_values[1:]:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    # Sharpe
    daily_rets = np.diff(account_values) / account_values[:-1]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0

    results.append({
        "label": exp["label"],
        "algo": exp["algo"],
        "timesteps": exp["timesteps"],
        "lr": exp["params"]["learning_rate"],
        "return_pct": ret_pct,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "final_value": final,
    })
    all_account_values[label] = account_values

    print(f"  [{label}] Return: {ret_pct:+.2f}%  MaxDD: {max_dd:.2f}%  Sharpe: {sharpe:+.2f}")


# ============================================================
# 5. Print comparison table
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
header = f"{'Experiment':<25} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'Final Value':>14}"
print(header)
print("-" * 65)
for r in sorted(results, key=lambda x: x["return_pct"], reverse=True):
    print(f"{r['label']:<25} {r['return_pct']:>+7.2f}% {r['max_dd']:>7.2f}% {r['sharpe']:>+7.2f}   ${r['final_value']:>12,.0f}")


# ============================================================
# 6. Save chart (plotly)
# ============================================================
print("\nGenerating chart...")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.08,
    subplot_titles=("Account Value Comparison", "Daily Return %"),
)

colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

for i, exp in enumerate(EXPERIMENTS):
    label = exp["label"]
    values = all_account_values[label]
    dates = trade_dates[:len(values)-1]
    daily_rets = np.diff(values) / values[:-1] * 100
    r = [x for x in results if x["label"] == label][0]
    legend = f"{label} (Ret:{r['return_pct']:+.1f}%, Sharpe:{r['sharpe']:+.2f})"

    fig.add_trace(
        go.Scatter(
            x=dates, y=values[:-1], name=legend,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate="%{fullData.name}<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=dates, y=daily_rets, name=label,
            marker_color=colors[i % len(colors)], opacity=0.5,
            showlegend=False,
            hovertemplate="%{fullData.name}<br>Date: %{x}<br>Ret: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

fig.add_hline(y=1000000, line_dash="dash", line_color="gray", opacity=0.4, row=1, col=1)

fig.update_layout(
    title="Parameter Tuning Experiment - Algorithm & Hyperparameter Comparison",
    template="plotly_white",
    height=700,
    width=1200,
    hovermode="x unified",
    margin=dict(l=60, r=30, t=60, b=40),
)
fig.update_yaxes(row=1, col=1, tickprefix="$", tickformat=",")
fig.update_yaxes(row=2, col=1, ticksuffix="%")
fig.update_xaxes(row=2, col=1, tickangle=-45, tickfont=dict(size=9))

chart_path = os.path.join(DEMO_DIR, "demo_5_experiment_chart.html")
fig.write_html(chart_path)

# Save results to CSV
results_df = pd.DataFrame(results)
csv_path = os.path.join(DEMO_DIR, "demo_5_experiment_results.csv")
results_df.to_csv(csv_path, index=False)

print(f"Chart saved: {chart_path}")
print(f"Results saved: {csv_path}")
print("\nDone!")
