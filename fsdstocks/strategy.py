from dataclasses import dataclass
import numpy as np
import pandas as pd
from fsdstocks.indicators import compute_indicators
from fsdstocks.utils import TRAIN_YEAR, run_backtest_equity_only, score_equity 

@dataclass
class StrategyParams:
    ma_fast: int = 50
    ma_slow: int = 200
    rsi_buy: float = 55.0
    rsi_sell: float = 45.0
    rsi_window: int = 14

def generate_signals(df, params: StrategyParams):
    d = compute_indicators(df, params.ma_fast, params.ma_slow, params.rsi_window)
    fast, slow = d[f"MA{params.ma_fast}"], d[f"MA{params.ma_slow}"]

    # --- Trend Filter (long-term direction) ---
    d["MA200"] = d["Adj Close"].rolling(200, min_periods=1).mean()
    uptrend = d["Adj Close"] > (d["MA200"] * 0.95)

    # --- Volatility Filter (ATR proxy using rolling range) --- 
    d["Volatility"] = d["High"].rolling(14, min_periods=1).max() - d["Low"].rolling(14, min_periods=1).min()
    vol_thresh = d["Volatility"].rolling(50, min_periods=1).mean() * 0.5
    vol_ok = d["Volatility"] > vol_thresh

    # --- RSI Momentum Confirmation ---
    rsi = d["RSI"]
    
    # --- Core Buy/Sell Conditions ---
    buy_condition = (
        (fast > slow) & 
        uptrend & 
        (rsi > params.rsi_buy - 15) &
        vol_ok
    )

    sell_condition = (
        (fast < slow) | 
        (rsi < params.rsi_sell) |
        (~uptrend)
    )

    d["signal"] = 0
    d.loc[buy_condition, "signal"] = 1
    d.loc[sell_condition, "signal"] = 0

    return d

def optimize_params_for_year(all_df, year=TRAIN_YEAR):
    year_df = all_df[all_df["Date"].dt.year == year]
    best, best_params = -np.inf, StrategyParams()
    for f in [5, 10, 20, 30, 50]:
        for s in [50, 100, 150, 200]:
            if f >= s: continue
            for rb in [45, 50, 55]:
                for rs in [35, 40, 45]:
                    params = StrategyParams(f, s, rb, rs)
                    eq = run_backtest_equity_only(year_df, params)
                    score = score_equity(eq)
                    if score > best:
                        best, best_params = score, params
    return best_params
