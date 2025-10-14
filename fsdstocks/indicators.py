import pandas as pd
import numpy as np

def compute_indicators(df, ma_fast, ma_slow, rsi_window=14):
    d = df.copy()
    price = d['Adj Close'].astype(float)

    d[f"MA{ma_fast}"] = price.rolling(ma_fast, min_periods=ma_fast).mean()
    d[f"MA{ma_slow}"] = price.rolling(ma_slow, min_periods=ma_slow).mean()

    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / rsi_window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / rsi_window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    d['RSI'] = 100 - (100 / (1 + rs))
    return d