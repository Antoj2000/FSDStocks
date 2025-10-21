import pandas as pd
import numpy as np

def compute_indicators(df, ma_fast=None, ma_slow=None, rsi_window=14, boll_window=None, boll_std=2):
    d = df.copy().reset_index(drop=True)  # 🩵 reset index to ensure rolling works correctly
    
    
    # --- Handle Price Column ---
    if "Adj Close" in d.columns:
        price_col = "Adj Close"
    elif "Adj_Close" in d.columns:
        price_col = "Adj_Close"
    else:
        raise KeyError("No 'Adj Close' or 'Adj_Close' column found in dataframe")

    price = d[price_col].astype(float)

    # --- Moving Averages ---
    if ma_fast is not None:
        d[f"MA{int(ma_fast)}"] = price.rolling(window=int(ma_fast), min_periods=1).mean().copy()
    if ma_slow is not None:
        d[f"MA{int(ma_slow)}"] = price.rolling(window=int(ma_slow), min_periods=1).mean().copy()

    # --- RSI calculation (EMA-based smoothing) ---
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / rsi_window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / rsi_window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands --- 
    if boll_window:
        sma = price.rolling(boll_window, min_periods=1).mean()
        std = price.rolling(boll_window, min_periods=1).std()
        d["Upper_Band"] = sma + (boll_std * std)
        d["Lower_Band"] = sma - (boll_std * std)
        d["BB_Middle"] = sma

    return d