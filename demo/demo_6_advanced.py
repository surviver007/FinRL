"""
FinRL Demo Step 6: Advanced Tuning - More stocks, More features, Better strategy
================================================================================
升级版调优实验，从多个维度提升收益

改进点：
  1. 股票池扩大：5只 → 15只（不同行业、不同波动率）
  2. 更多技术指标：8个 → 16个
  3. 奖励缩放对比
  4. 集成策略（多模型投票）
"""

import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
import tushare as ts

os.environ["NO_PROXY"] = "*"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    os.environ.pop(k, None)
import urllib.request
urllib.request.getproxies = lambda: {}

from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from stable_baselines3 import PPO, A2C

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

TS_TOKEN = os.environ.get("TUSHARE_TOKEN")
if not TS_TOKEN:
    raise RuntimeError(
        "Missing TUSHARE_TOKEN environment variable. "
        "Set it before running this script, for example: "
        "$env:TUSHARE_TOKEN='your_token'"
    )
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 扩展技术指标
INDICATORS_EXTENDED = [
    "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
    "close_30_sma", "close_60_sma",
    "rsi_14",          # 短期RSI
    "rsi_60",          # 长期RSI
    "cci_14",          # 短期CCI
    "dx_14",           # 短期动向指数
    "close_5_sma",     # 5日均线
    "close_10_sma",    # 10日均线
    "close_20_sma",    # 20日均线
    "close_120_sma",   # 半年线
]

# 扩展股票池 - 15只，覆盖不同行业和波动率
TICKER_MAP = {
    # 金融
    "000001.SZ": "pingan_bank",   # 平安银行
    "601318.SH": "pingan_ins",    # 中国平安
    "600036.SH": "cmb",           # 招商银行
    # 消费
    "600519.SH": "maotai",        # 贵州茅台
    "000858.SZ": "wuliangye",     # 五粮液
    "000333.SZ": "midea",         # 美的集团
    # 科技
    "002415.SZ": "hikvision",     # 海康威视
    "000725.SZ": "boe",           # 京东方A
    # 新能源
    "300750.SZ": "catl",          # 宁德时代
    "002594.SZ": "byd",           # 比亚迪
    # 医药
    "600276.SH": "jiangsu_hengrui",# 恒瑞医药
    "000538.SZ": "yunnan_baiyao", # 云南白药
    # 周期
    "601088.SH": "china_shenhua", # 中国神华
    "600028.SH": "sinopec",       # 中国石化
    # 地产
    "000002.SZ": "vanke",         # 万科A
}

TRAIN_START = "20200101"  # 多拉2年数据
TRAIN_END = "20241231"
TRADE_START = "20250101"
TRADE_END = "20260527"


# ============================================================
# 1. Download data
# ============================================================
print("=" * 60)
print("[Step 6] Advanced Tuning - 15 stocks, 16 indicators")
print("=" * 60)

# Check if data already downloaded
train_csv = os.path.join(DEMO_DIR, "demo_6_train_data.csv")
trade_csv = os.path.join(DEMO_DIR, "demo_6_trade_data.csv")

if os.path.exists(train_csv) and os.path.exists(trade_csv):
    print("Using cached data...")
    train = pd.read_csv(train_csv)
    trade = pd.read_csv(trade_csv)
else:
    print(f"\nDownloading {len(TICKER_MAP)} stocks via Tushare (前复权)...")
    all_dfs = []
    for ts_code, name in TICKER_MAP.items():
        try:
            print(f"  {ts_code} ({name})...", end=" ", flush=True)
            df_t = ts.pro_bar(ts_code=ts_code, start_date=TRAIN_START, end_date=TRADE_END, adj='qfq')
            if df_t is not None and len(df_t) > 0:
                df_t = df_t.rename(columns={"trade_date": "date", "vol": "volume"})
                df_t["tic"] = name
                df_t["adjcp"] = df_t["close"]
                df_t["date"] = pd.to_datetime(df_t["date"]).dt.strftime("%Y-%m-%d")
                df_t = df_t.sort_values("date")
                df_t = df_t[["date", "open", "high", "low", "close", "adjcp", "volume", "tic"]]
                all_dfs.append(df_t)
                print(f"{len(df_t)} rows")
            else:
                print("no data!")
        except Exception as e:
            print(f"error: {str(e)[:60]}")

    if not all_dfs:
        print("ERROR: No data!")
        sys.exit(1)

    df_raw = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(df_raw)} rows")

    # Compute indicators
    print("Computing technical indicators (16)...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS_EXTENDED,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df_raw)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])
    processed_full = processed_full.fillna(0)

    # 只保留成功下载的指标列
    available_indicators = [i for i in INDICATORS_EXTENDED if i in processed_full.columns]
    print(f"Available indicators: {len(available_indicators)} -> {available_indicators}")

    train = data_split(processed_full, "2020-01-01", "2024-12-31")
    trade = data_split(processed_full, "2025-01-01", "2026-05-27")

    train.to_csv(train_csv, index=False)
    trade.to_csv(trade_csv, index=False)
    print(f"Train: {len(train)} rows, Trade: {len(trade)} rows")
    print(f"Data cached to demo_6_train/trade_data.csv")

stock_list = sorted(train["tic"].unique().tolist())
actual_indicators = [i for i in INDICATORS_EXTENDED if i in train.columns]
print(f"\nStocks: {len(stock_list)}, Indicators: {len(actual_indicators)}")


# ============================================================
# 2. Prepare arrays
# ============================================================
def df_to_arrays(df):
    price_df = df.pivot_table(index="date", columns="tic", values="close")
    price_df = price_df[stock_list]
    tech_data = {}
    for ind in actual_indicators:
        if ind in df.columns:
            pivot = df.pivot_table(index="date", columns="tic", values=ind)
            pivot = pivot[stock_list]
            tech_data[ind] = pivot.values.astype(np.float32)
    price_array = price_df.values.astype(np.float32)
    tech_array = np.hstack(list(tech_data.values())).astype(np.float32)
    turbulence_array = np.zeros(len(price_df), dtype=np.float32)
    return price_array, tech_array, turbulence_array, price_df.index.tolist()

price_train, tech_train, turb_train, _ = df_to_arrays(train)
price_trade, tech_trade, turb_trade, trade_dates = df_to_arrays(trade)

print(f"Train: {price_train.shape[0]} days x {price_train.shape[1]} stocks, tech: {tech_train.shape}")
print(f"Trade: {price_trade.shape[0]} days x {price_trade.shape[1]} stocks")


# ============================================================
# 3. Define experiments
# ============================================================
EXPERIMENTS = [
    {
        "name": "v2_baseline",
        "label": "v2 Baseline (PPO 50k)",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.05, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 50000,
    },
    {
        "name": "v2_ppo_100k",
        "label": "v2 PPO 100k",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.05, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 100000,
    },
    {
        "name": "v2_ppo_200k",
        "label": "v2 PPO 200k",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.05, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 200000,
    },
    {
        "name": "v2_a2c_100k",
        "label": "v2 A2C 100k",
        "algo": "A2C",
        "params": {"n_steps": 10, "ent_coef": 0.05, "learning_rate": 0.0007},
        "timesteps": 100000,
    },
    {
        "name": "v2_a2c_200k",
        "label": "v2 A2C 200k",
        "algo": "A2C",
        "params": {"n_steps": 10, "ent_coef": 0.05, "learning_rate": 0.0007},
        "timesteps": 200000,
    },
    {
        "name": "v2_ppo_lowent",
        "label": "v2 PPO Low Explore",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.001, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 100000,
    },
    {
        "name": "v2_ppo_200k_lowent",
        "label": "v2 PPO 200k LowEnt",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.001, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 200000,
    },
    {
        "name": "v2_ppo_300k",
        "label": "v2 PPO 300k",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.05, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 300000,
    },
    {
        "name": "v2_ppo_300k_lowent",
        "label": "v2 PPO 300k LowEnt",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.001, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 300000,
    },
    {
        "name": "v2_ppo_500k",
        "label": "v2 PPO 500k",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.05, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 500000,
    },
    {
        "name": "v2_ppo_500k_lowent",
        "label": "v2 PPO 500k LowEnt",
        "algo": "PPO",
        "params": {"n_steps": 2048, "ent_coef": 0.001, "learning_rate": 0.0003, "batch_size": 256},
        "timesteps": 500000,
    },
]


# ============================================================
# 4. Train & Backtest
# ============================================================
print("\n--- Training & Backtesting ---\n")

TRAIN_SEED = 42

ALGO_MAP = {"PPO": PPO, "A2C": A2C}
all_account_values = {}
results = []


def run_backtest(model, price_array, tech_array, turbulence_array, seed=42):
    """Run backtest and return metrics + account values."""
    env = StockTradingEnv(config={
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    })
    obs, info = env.reset(seed=seed)
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
    peak = initial
    max_dd = 0
    for v in account_values[1:]:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak * 100)
    daily_rets = np.diff(account_values) / account_values[:-1]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    calmar = ret_pct / max_dd if max_dd > 0 else 0
    return {"ret_pct": ret_pct, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
            "final_value": final}, account_values


for exp in EXPERIMENTS:
    print(f"  [{exp['label']}] training {exp['timesteps']//1000}k...", end=" ", flush=True)
    t0 = time.time()

    env = StockTradingEnv(config={
        "price_array": price_train,
        "tech_array": tech_train,
        "turbulence_array": turb_train,
        "if_train": True,
    })

    algo_cls = ALGO_MAP[exp["algo"]]
    model = algo_cls("MlpPolicy", env, verbose=0, seed=TRAIN_SEED, **exp["params"])
    model.learn(total_timesteps=exp["timesteps"])

    save_path = os.path.join(MODEL_DIR, f"exp_{exp['name']}")
    model.save(save_path)
    train_time = time.time() - t0

    # In-sample backtest (train set)
    in_metrics, _ = run_backtest(model, price_train, tech_train, turb_train, seed=42)
    # Out-of-sample backtest (trade set)
    out_metrics, out_values = run_backtest(model, price_trade, tech_trade, turb_trade, seed=42)

    results.append({
        "label": exp["label"],
        "algo": exp["algo"],
        "timesteps": exp["timesteps"],
        "lr": exp["params"]["learning_rate"],
        "ent_coef": exp["params"]["ent_coef"],
        # out-of-sample
        "return_pct": out_metrics["ret_pct"],
        "max_dd": out_metrics["max_dd"],
        "sharpe": out_metrics["sharpe"],
        "calmar": out_metrics["calmar"],
        "final_value": out_metrics["final_value"],
        # in-sample
        "in_return_pct": in_metrics["ret_pct"],
        "in_max_dd": in_metrics["max_dd"],
        "in_sharpe": in_metrics["sharpe"],
        # gap
        "sharpe_gap": in_metrics["sharpe"] - out_metrics["sharpe"],
        "return_gap": in_metrics["ret_pct"] - out_metrics["ret_pct"],
        "train_time": train_time,
    })
    all_account_values[exp["label"]] = out_values

    print(f"done ({train_time:.0f}s) | In: {in_metrics['ret_pct']:+.2f}%/S:{in_metrics['sharpe']:+.2f} | Out: {out_metrics['ret_pct']:+.2f}%/S:{out_metrics['sharpe']:+.2f} | Gap: {in_metrics['sharpe'] - out_metrics['sharpe']:+.2f}")


# ============================================================
# 5. Ensemble strategy
# ============================================================
print("\n--- Ensemble Strategy ---\n")

# 用所有训练好的模型，对每个动作取平均（投票）
print("  [Ensemble (all models vote)] ...", end=" ", flush=True)

# Load all models
models = []
for exp in EXPERIMENTS:
    save_path = os.path.join(MODEL_DIR, f"exp_{exp['name']}")
    algo_cls = ALGO_MAP[exp["algo"]]
    models.append(algo_cls.load(save_path))


def run_ensemble_backtest(models, price_array, tech_array, turbulence_array, seed=42):
    """Run ensemble backtest with average voting."""
    env = StockTradingEnv(config={
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    })
    obs, info = env.reset(seed=seed)
    account_values = [env.initial_capital]
    done = False
    while not done:
        actions = []
        for m in models:
            a, _ = m.predict(obs, deterministic=True)
            actions.append(a)
        action = np.mean(actions, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_asset = env.amount + np.sum(env.stocks * env.price_ary[env.day])
        account_values.append(total_asset)

    initial = account_values[0]
    final = account_values[-1]
    ret_pct = (final / initial - 1) * 100
    peak = initial
    max_dd = 0
    for v in account_values[1:]:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak * 100)
    daily_rets = np.diff(account_values) / account_values[:-1]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    calmar = ret_pct / max_dd if max_dd > 0 else 0
    return {"ret_pct": ret_pct, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
            "final_value": final}, account_values


ens_in, _ = run_ensemble_backtest(models, price_train, tech_train, turb_train, seed=42)
ens_out, ens_values = run_ensemble_backtest(models, price_trade, tech_trade, turb_trade, seed=42)

results.append({
    "label": "Ensemble (all vote)",
    "algo": "Ensemble",
    "timesteps": "-",
    "lr": "-",
    "ent_coef": "-",
    "return_pct": ens_out["ret_pct"],
    "max_dd": ens_out["max_dd"],
    "sharpe": ens_out["sharpe"],
    "calmar": ens_out["calmar"],
    "final_value": ens_out["final_value"],
    "in_return_pct": ens_in["ret_pct"],
    "in_max_dd": ens_in["max_dd"],
    "in_sharpe": ens_in["sharpe"],
    "sharpe_gap": ens_in["sharpe"] - ens_out["sharpe"],
    "return_gap": ens_in["ret_pct"] - ens_out["ret_pct"],
    "train_time": 0,
})
all_account_values["Ensemble (all vote)"] = ens_values
print(f"In: {ens_in['ret_pct']:+.2f}%/S:{ens_in['sharpe']:+.2f} | Out: {ens_out['ret_pct']:+.2f}%/S:{ens_out['sharpe']:+.2f} | Gap: {ens_in['sharpe'] - ens_out['sharpe']:+.2f}")


# ============================================================
# 6. Print results
# ============================================================
print("\n" + "=" * 100)
print("COMPARISON TABLE (sorted by Out-of-Sample Sharpe)")
print("=" * 100)
header = f"{'Experiment':<28} {'In-Ret':>8} {'In-Sharpe':>10} {'Out-Ret':>8} {'Out-Sharpe':>11} {'Gap':>6} {'MaxDD':>8} {'Time':>6}"
print(header)
print("-" * 100)
for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
    t_str = f"{r['train_time']:.0f}s" if isinstance(r['train_time'], (int, float)) else "-"
    in_ret = r.get("in_return_pct", 0)
    in_sharpe = r.get("in_sharpe", 0)
    gap = r.get("sharpe_gap", 0)
    overfit_mark = " <<" if gap > 1.0 else ""
    print(f"{r['label']:<28} {in_ret:>+7.2f}% {in_sharpe:>+9.2f} {r['return_pct']:>+7.2f}% {r['sharpe']:>+10.2f} {gap:>+5.2f}{overfit_mark} {r['max_dd']:>7.2f}% {t_str:>6}")

print(f"\n注: Gap = In-Sharpe - Out-Sharpe, Gap > 1.0 标记为 '<<' 表示可能过拟合")


# ============================================================
# 7. Chart
# ============================================================
print("\nGenerating chart...")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.08,
    subplot_titles=("Account Value - 15 Stocks, 16 Indicators", "Daily Return %"),
)

colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

for i, r in enumerate(sorted(results, key=lambda x: x["sharpe"], reverse=True)):
    label = r["label"]
    if label not in all_account_values:
        continue
    values = all_account_values[label]
    dates = trade_dates[:len(values)-1]
    daily_rets = np.diff(values) / values[:-1] * 100
    legend = f"{label} (Ret:{r['return_pct']:+.1f}%, Sharpe:{r['sharpe']:+.2f})"
    c = colors[i % len(colors)]

    fig.add_trace(go.Scatter(
        x=dates, y=values[:-1], name=legend,
        line=dict(color=c, width=2),
        hovertemplate="%{fullData.name}<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=dates, y=daily_rets, name=label,
        marker_color=c, opacity=0.4, showlegend=False,
    ), row=2, col=1)

fig.add_hline(y=1000000, line_dash="dash", line_color="gray", opacity=0.4, row=1, col=1)

fig.update_layout(
    title="Advanced Tuning - 15 Stocks, 16 Indicators, Ensemble",
    template="plotly_white", height=700, width=1200,
    hovermode="x unified",
    margin=dict(l=60, r=30, t=60, b=40),
)
fig.update_yaxes(row=1, col=1, tickprefix="$", tickformat=",")
fig.update_yaxes(row=2, col=1, ticksuffix="%")
fig.update_xaxes(row=2, col=1, tickangle=-45, tickfont=dict(size=9))

chart_path = os.path.join(DEMO_DIR, "demo_6_advanced_chart.html")
fig.write_html(chart_path)

# Save results
results_df = pd.DataFrame(results)
csv_path = os.path.join(DEMO_DIR, "demo_6_advanced_results.csv")
results_df.to_csv(csv_path, index=False)

print(f"\nChart: {chart_path}")
print(f"Results: {csv_path}")
print("Done!")
