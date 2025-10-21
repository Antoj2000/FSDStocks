import os 
import numpy as np
import pandas as pd

# --- Core Paths ---
SRC_DIR = os.path.dirname(__file__)

DEFAULT_TEST_YEARS = list(range(2014, 2026))
TRAIN_YEAR = 2024

# --- Directories ---
OUTPUT_DIR = os.path.join(SRC_DIR, "output")
REPORTS_DIR = os.path.join(SRC_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- File Paths ---
RESULTS_TXT = os.path.join(REPORTS_DIR, "summary_results.txt")   # moved from output → reports
PERFORMANCE_CSV = os.path.join(REPORTS_DIR, "strategy_performance.csv")

PLOT_FMT = os.path.join(OUTPUT_DIR, "equity_{year}.png")
XLSX_FMT = os.path.join(OUTPUT_DIR, "results_{year}.xlsx")

# === Simplified Backtest & Scoring for Param Search === 

def run_backtest_equity_only(df, params):
    # Equity curve generator for param optimization
    if df.empty: 
        idx = pd.date_range(start="2024-01-01", end="2024-12-31", freq="B")
        return pd.Series(1.0, index=idx)
    
    from fsdstocks.strategy import generate_signals
    
    df = df.copy()
    df = generate_signals(df, params)
    df = df.sort_values("Date")

    # --- handle both 'Adj Close' and 'Adj_Close' ---
    price_col = "Adj Close" if "Adj Close" in df.columns else "Adj_Close"

    capital, in_pos, entry = 1.0, False, 0
    equity = []
    for _, row in df.iterrows():
        if in_pos and row.signal == 0:
            capital *= row[price_col] / entry
            in_pos = False
        elif not in_pos and row.signal == 1:
            entry = row[price_col]
            in_pos = True
        equity.append(capital)
    return pd.Series(equity, index=df.Date)

def score_equity(equity):
     # --- Safety checks ---
    if equity is None or len(equity) < 2:
        return -9999.0

    equity = pd.Series(equity).dropna()
    if equity.nunique() < 2:
        return -9999.0

    ret = equity.pct_change().dropna()
    if ret.empty or ret.std() == 0:
        return -9999.0

    total = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe_like = (ret.mean() / ret.std()) * np.sqrt(252)

    # Compute max drawdown
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Combine metrics with balanced weights
    score = (total * 200) + (sharpe_like * 5) + (max_dd * 20)

    # Small numeric noise fix
    return float(np.round(score, 4))