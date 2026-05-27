"""
FinRL Demo Step 1: Download real A-share data via Tushare
"""

import os
import itertools
import pandas as pd
import tushare as ts

# Disable proxy
os.environ["NO_PROXY"] = "*"
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"]:
    os.environ.pop(k, None)
import urllib.request
urllib.request.getproxies = lambda: {}

from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer

# ===== Tushare config =====
TS_TOKEN = "8347a171dac7fe6fc71fdb12e9455a424754db51efc025b5b9acdc63"
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# A股热门股票：平安银行、贵州茅台、五粮液、中国平安、美的集团
TICKER_MAP = {
    "000001.SZ": "pingan_bank",
    "600519.SH": "maotai",
    "000858.SZ": "wuliangye",
    "601318.SH": "pingan_ins",
    "000333.SZ": "midea",
}

TRAIN_START = "20220101"
TRAIN_END = "20241231"
TRADE_START = "20250101"
TRADE_END = "20250501"

print("=" * 50)
print("[Step 1] Downloading A-share data via Tushare...")
print(f"   Stocks: {list(TICKER_MAP.values())}")
print("=" * 50)

all_dfs = []
for ts_code, name in TICKER_MAP.items():
    try:
        print(f"  Downloading {ts_code} ({name})...", flush=True)
        df_t = pro.daily(ts_code=ts_code, start_date=TRAIN_START, end_date=TRADE_END)
        if df_t is not None and len(df_t) > 0:
            # Tushare 返回的列：ts_code, trade_date, open, high, low, close, vol, amount 等
            df_t = df_t.rename(columns={
                "trade_date": "date",
                "vol": "volume",
            })
            df_t["tic"] = name
            df_t["adjcp"] = df_t["close"]
            df_t["date"] = pd.to_datetime(df_t["date"]).dt.strftime("%Y-%m-%d")
            df_t = df_t.sort_values("date")
            df_t = df_t[["date", "open", "high", "low", "close", "adjcp", "volume", "tic"]]
            all_dfs.append(df_t)
            print(f"    {name}: {len(df_t)} rows OK", flush=True)
        else:
            print(f"    {name}: no data!", flush=True)
    except Exception as e:
        print(f"    {name}: error - {str(e)[:120]}", flush=True)

if not all_dfs:
    print("ERROR: No data downloaded!")
    exit(1)

df_raw = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal downloaded: {len(df_raw)} rows")
print(df_raw.head(10))

# Technical indicators
print("\n[Step 1b] Computing technical indicators...", flush=True)
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=False,           # A股不需要VIX
    use_turbulence=False,    # 关掉，省内存
    user_defined_feature=False,
)

processed = fe.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])
processed_full = processed_full.fillna(0)

print(f"Processed: {len(processed_full)} rows")

TRAIN_START_FMT = "2022-01-01"
TRAIN_END_FMT = "2024-12-31"
TRADE_START_FMT = "2025-01-01"
TRADE_END_FMT = "2025-05-01"

train = data_split(processed_full, TRAIN_START_FMT, TRAIN_END_FMT)
trade = data_split(processed_full, TRADE_START_FMT, TRADE_END_FMT)

print(f"\nTrain: {len(train)} rows ({TRAIN_START_FMT} ~ {TRAIN_END_FMT})")
print(f"Trade: {len(trade)} rows ({TRADE_START_FMT} ~ {TRADE_END_FMT})")

train.to_csv("demo_train_data.csv", index=False)
trade.to_csv("demo_trade_data.csv", index=False)
print("\nOK! Data saved: demo_train_data.csv, demo_trade_data.csv")
