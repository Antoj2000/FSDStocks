import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fsdstocks.strategy import generate_signals
from fsdstocks.utils import OUTPUT_DIR, PLOT_FMT, XLSX_FMT, RESULTS_TXT

@dataclass
class Trade:
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    ticker: str
    entry: float
    exit: float
    days: int
    pct: float

def backtest_year(all_df, year, params):
    year_df = all_df[all_df["Date"].dt.year == year]
    if year_df.empty:
        raise ValueError(f"No data for year {year}")
    
    dfs = [generate_signals(tdf, params) for _, tdf in year_df.groupby("Ticker")]
    df = pd.concat(dfs).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    trades, in_pos, entry_price, entry_ticker, entry_date = [], False, 0, "", None
    equity_vals, equity_dates, capital = [], [], 1.0

    STOP_LOSS = -0.05
    TAKE_PROFIT = 0.10

    for day, g in df.groupby("Date"):
        if in_pos:
            row = g[g["Ticker"] == entry_ticker]
            if not row.empty:
                price = row.iloc[0]["Adj Close"]
                rsi = row.iloc[0].get("RSI", np.nan)

                pct_change = (price / entry_price) - 1.0

                if (
                    pct_change <= STOP_LOSS or
                    pct_change >= TAKE_PROFIT or
                    (not np.isnan(rsi) and rsi < 40)
                ):
                    exit_price = price
                    pct = exit_price / entry_price - 1.0
                    capital *= (1 + pct)
                    days = (day - entry_date).days
                    trades.append(Trade(entry_date, day, entry_ticker, entry_price, exit_price, days, pct * 100))
                    in_pos = False

                elif row.iloc[0]["signal"] == 0:
                    exit_price = price
                    pct = exit_price / entry_price - 1.0
                    capital *= (1 + pct)
                    days = (day - entry_date).days
                    trades.append(Trade(entry_date, day, entry_ticker, entry_price, exit_price, days, pct * 100))
                    in_pos = False
        
        if not in_pos:
            cands = g[g["signal"] == 1]
            if not cands.empty:
                first = cands.iloc[0]
                entry_price, entry_ticker, entry_date, in_pos = first["Adj Close"], first["Ticker"], day, True
           
        equity_vals.append(capital)
        equity_dates.append(day)

    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates)).asfreq("B").ffill()

    # --- Metrics ---

    if trades: 
        avg_return = np.mean([t.pct for t in trades]) if trades else 0
        avg_days = np.mean([t.days for t in trades]) if trades else 0
        num_trades = len(trades)
        est_total_return = (1 + avg_return / 100) ** num_trades - 1.0
    else:
        avg_return, avg_days, num_trades, est_total_return = 0.0, 0.0, 0, 0.0

    metrics = {
        "avg_return_pct": avg_return,
        "avg_hold_days": avg_days,
        "num_trades": num_trades,
        "est_total_return": est_total_return * 100,
    }

    tickers = year_df["Ticker"].unique()
    ticker_folder = tickers[0] if len(tickers) == 1 else "MULTI"
    year_folder = str(year)

    output_path = os.path.join(OUTPUT_DIR, ticker_folder, year_folder)
    os.makedirs(output_path, exist_ok=True)

    plot_path = os.path.join(output_path, f"equity_{year}.png")
    xslx_path = os.path.join(output_path, f"results_{year}.xlsx")
    results_txt = os.path.join(output_path, "results.txt")


    plt.figure(figsize=(10, 5))
    equity.plot(title=f"{ticker_folder} Equity Curve {year}", ylabel="Equity", xlabel="Date")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    # --- Additional Insight Graph: Buy/Sell Signals ---
    sample_ticker = year_df["Ticker"].unique()[0]
    tdf = df[df["Ticker"] == sample_ticker].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(tdf["Date"], tdf["Adj Close"], label="Adj Close", color="black", linewidth=1)
    plt.plot(tdf["Date"], tdf[f"MA{params.ma_fast}"], label=f"MA{params.ma_fast}", color="blue", alpha=0.7)
    plt.plot(tdf["Date"], tdf[f"MA{params.ma_slow}"], label=f"MA{params.ma_slow}", color="red", alpha=0.7)

    for t in trades:
        plt.scatter(t.open_date, t.entry, marker="^", color="green", s=100, label="Buy Signal" if t == trades[0] else "")
        plt.scatter(t.close_date, t.exit, marker="v", color="red", s=100, label="Sell Signal" if t == trades[0] else "")

    plt.title(f"{sample_ticker} Price with Buy/Sell Signals {year}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    signal_plot_path = os.path.join(output_path, f"signals_{sample_ticker}_{year}.png")
    plt.savefig(signal_plot_path, dpi=120)
    plt.close()

    trades_df = pd.DataFrame([
        {
            "OpenDate": t.open_date.strftime("%Y-%m-%d"),
            "CloseDate": t.close_date.strftime("%Y-%m-%d"),
            "Ticker": t.ticker,
            "Entry": round(t.entry, 4),
            "Exit": round(t.exit, 4),
            "HoldDays": t.days,
            "ReturnPct": round(t.pct, 3),
        } 
        for t in trades
    ])

    with pd.ExcelWriter(xslx_path, engine="openpyxl") as w:
        trades_df.to_excel(w, index=False, sheet_name="Trades")
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_excel(w, index=False, sheet_name="Summary")

    write_mode = "w" if year == sorted(all_df["Date"].dt.year.unique())[-1] else "a"

    est_total_return = metrics.get("est_total_return", 0.0)

    with open(results_txt, write_mode, encoding="utf-8") as f:
        f.write(
            f"Year: {year} | "
            f"AvgRet: {metrics.get('avg_return_pct', 0):.2f}% | "
            f"AvgHold: {metrics.get('avg_hold_days', 0):.1f}d | "
            f"Trades: {metrics.get('num_trades', 0)} | "
            f"Total: {est_total_return:.2f}%\n"
        )
    
    # --- Update Overall Results Overview ---
    overview_path = os.path.join(OUTPUT_DIR, ticker_folder, "results_overview.txt")

    latest_year = sorted(all_df["Date"].dt.year.unique())[-1]
    if year == latest_year:
        if os.path.exists(overview_path):
            os.remove(overview_path)
        mode = "w"
        header = True
    else:
        mode = "a"
        header = False

    with open(overview_path, mode, encoding="utf-8") as f:
        if header:
            f.write(f"{ticker_folder} Backtest Results Overview\n{'-' * 60}\n")
        f.write(
            f"Year: {year} | "
            f"AvgRet: {metrics.get('avg_return_pct', 0):.2f}% | "
            f"AvgHold: {metrics.get('avg_hold_days', 0):.1f}d | "
            f"Trades: {metrics.get('num_trades', 0)} | "
            f"Total: {est_total_return:.2f}%\n"
        )

    return trades_df, equity, metrics