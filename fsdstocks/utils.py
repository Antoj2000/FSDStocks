import os 
import numpy as np
import pandas as pd

DEFAULT_TEST_YEARS = list(range(2014, 2026))
TRAIN_YEAR = 2024
OUTPUT_DIR = "output"
RESULTS_TXT = os.path.join(OUTPUT_DIR, "results.txt")
PLOT_FMT = os.path.join(OUTPUT_DIR, "equity_{year}.png")
XLSX_FMT = os.path.join(OUTPUT_DIR, "results_{year}.xlsx")

def run_backtest_equity_only(df, params):
    if df.empty: 
        idx = pd.date_range(start="2024-01-01", end="2024-12-31", freq="B")
        return pd.Series(1.0, index=idx)
    
    from fsdstocks.strategy import generate_signals
    
    df = df.copy()
    df = generate_signals(df, params)
    df = df.sort_values("Date")

    capital, in_pos, entry = 1.0, False, 0
    equity = []
    for _, row in df.iterrows():
        if in_pos and row.signal == 0:
            capital *= row["Adj Close"] / entry
            in_pos = False
        elif not in_pos and row.signal == 1:
            entry = row["Adj Close"]
            in_pos = True
        equity.append(capital)
    return pd.Series(equity, index=df.Date)

def score_equity(equity):
    ret = equity.pct_change().dropna()
    if ret.empty or ret.std() == 0:
        return -np.inf
    total = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe_like = ret.mean() / ret.std() * np.sqrt(252)
    return total * max (sharpe_like, 0)