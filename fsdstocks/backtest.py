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
    # --- Persistent Trackers for overview ---
    if not hasattr(backtest_year, "_compounded_return"):
        backtest_year._compounded_return = 1.0
    if not hasattr(backtest_year, "_results_written"):
        backtest_year._results_written = False

    # --- filter to year & compute signals ---
    year_df = all_df[all_df["Date"].dt.year == year]
    if year_df.empty:
        raise ValueError(f"No data for year {year}")
    
    # Generate signals per ticker, then concat
    dfs = [generate_signals(tdf, params) for _, tdf in year_df.groupby("Ticker")]
    df = pd.concat(dfs).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # --- Vectorized baseline : daily returns & next-day execution ---
    # No lookahead: we can only act on a signal the NEXT bar
    df["daily_ret"] = df.groupby("Ticker")["Adj Close"].pct_change()
    df["position_raw"] = df.groupby("Ticker")["signal"].shift(1).fillna(0).astype(int)
    df["position_adj"] = df["position_raw"].copy()


    # --- Reconstruct trade segments from vectorized positions ---
    # We’ll derive entry/exit pairs from position transitions,
    # then enforce STOP/TP inside each segment (light loop over trades only).

    trades = []
    STOP_LOSS = -0.05
    TAKE_PROFIT = 0.10

    for ticker, g in df.groupby("Ticker", sort=False):
        g = g.copy()
        # transitions: +1 = entry, -1 = exit
        g["pos_change"] = g["position_raw"].diff().fillna(g["position_raw"])
        entries = g.index[g["pos_change"] == 1].tolist()
        exits   = g.index[g["pos_change"] == -1].tolist()

        # If we start in a position and no prior exit, ensure pairing is valid
        # Balance entries/exits lengths
        if len(entries) and (len(exits) == 0 or entries[0] > exits[0]):
            # There was an exit before a corresponding entry; drop it
            exits = [e for e in exits if e > entries[0]]

        if len(entries) > len(exits):
            # last position is still open until the final index
            exits.append(g.index[-1])

        # Process each intended trade segment to apply SL/TP/RSI exit earlier if hit
        for ent_idx, ex_idx in zip(entries, exits):
            ent_row = g.loc[ent_idx]
            # Segment slice is (ent_idx+1 .. ex_idx], because position starts next bar
            seg = g.loc[ent_idx+1:ex_idx].copy()
            if seg.empty:
                continue

            entry_price = ent_row["Adj Close"]

            # Running return from entry; if any breach SL/TP, cut at first breach
            seg["ret_from_entry"] = (seg["Adj Close"] / entry_price) - 1.0

            hit_sl = seg.index[seg["ret_from_entry"] <= STOP_LOSS]
            hit_tp = seg.index[seg["ret_from_entry"] >= TAKE_PROFIT]

            # Optional RSI emergency exit as in your looped version
            if "RSI" in seg.columns:
                hit_rsi = seg.index[seg["RSI"] < 40]
            else:
                hit_rsi = pd.Index([])

            # earliest exit among any condition
            candidates = []
            if len(hit_sl): candidates.append(hit_sl[0])
            if len(hit_tp): candidates.append(hit_tp[0])
            if len(hit_rsi): candidates.append(hit_rsi[0])

            if candidates:
                true_exit_idx = min(candidates)  # first hit
            else:
                true_exit_idx = ex_idx  # normal exit by signal

            close_row = g.loc[true_exit_idx]
            pct = (close_row["Adj Close"] / entry_price - 1.0) * 100.0
            days = int((close_row["Date"] - ent_row["Date"]).days)

            trades.append(
                Trade(
                    open_date=ent_row["Date"],
                    close_date=close_row["Date"],
                    ticker=ticker,
                    entry=float(entry_price),
                    exit=float(close_row["Adj Close"]),
                    days=days,
                    pct=float(np.round(pct, 6)),
                )
            )

            # Zero out position after the *true* exit until the next entry
            # i.e., from the day AFTER true_exit_idx up to the original planned exit
            after_exit_slice = g.index[(g.index > true_exit_idx) & (g.index <= ex_idx)]
            if len(after_exit_slice):
                df.loc[after_exit_slice, "position_adj"] = 0

    # --- Build adjusted strategy returns & equity (vectorized) ---
    df["strat_ret"] = df["daily_ret"] * df["position_adj"]
    # If multiple tickers passed, average equally by date; otherwise single ticker
    strat_by_date = df.groupby("Date")["strat_ret"].mean()
    equity = (1.0 + strat_by_date.fillna(0)).cumprod()

    # --- Metrics ---

    if trades: 
        avg_return = float(np.mean([t.pct for t in trades]))
        avg_days = float(np.mean([t.days for t in trades]))
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


    # --- Output paths ---
    tickers = year_df["Ticker"].unique()
    ticker_folder = tickers[0] if len(tickers) == 1 else "MULTI"
    if not hasattr(backtest_year, "_current_ticker") or backtest_year._current_ticker != ticker_folder:
        backtest_year._current_ticker = ticker_folder
        backtest_year._compounded_return = 1.0
        backtest_year._results_written = False
    year_folder = str(year)
    output_path = os.path.join(OUTPUT_DIR, ticker_folder, year_folder)
    os.makedirs(output_path, exist_ok=True)


    
    results_txt = os.path.join(output_path, "results.txt")

    # --- Plots (equity) ---
    plt.figure(figsize=(10, 5))
    equity.plot(title=f"{ticker_folder} Equity Curve {year}", ylabel="Equity", xlabel="Date")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"equity_{year}.png"), dpi=120)
    plt.close()

    # --- Additional Insight Graph: Buy/Sell Signals ---
    sample_ticker = year_df["Ticker"].unique()[0]
    tdf = df[df["Ticker"] == sample_ticker].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(tdf["Date"], tdf["Adj Close"], label="Adj Close", color="black", linewidth=1)
    if f"MA{params.ma_fast}" in tdf.columns:
        plt.plot(tdf["Date"], tdf[f"MA{params.ma_fast}"], label=f"MA{params.ma_fast}", color="blue", alpha=0.7)
    if f"MA{params.ma_slow}" in tdf.columns:
        plt.plot(tdf["Date"], tdf[f"MA{params.ma_slow}"], label=f"MA{params.ma_slow}", color="red", alpha=0.7)

    # mark trade entries/exits for the sample ticker
    t_trades = [t for t in trades if t.ticker == sample_ticker]
    for i, t in enumerate(t_trades):
        plt.scatter(t.open_date, t.entry, marker="^", color="green", s=80, label="Buy" if i == 0 else "")
        plt.scatter(t.close_date, t.exit, marker="v", color="red", s=80, label="Sell" if i == 0 else "")
    plt.title(f"{sample_ticker} Price with Buy/Sell Signals {year}")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"signals_{sample_ticker}_{year}.png"), dpi=120)
    plt.close()


    # --- Spreadsheets ---
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
    xslx_path = os.path.join(output_path, f"results_{year}.xlsx")
    with pd.ExcelWriter(xslx_path, engine="openpyxl") as w:
        trades_df.to_excel(w, index=False, sheet_name="Trades")
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_excel(w, index=False, sheet_name="Summary")


    # --- Yearly results.txt --- 
    results_txt = os.path.join(output_path, "results.txt")
    latest_year = sorted(all_df["Date"].dt.year.unique())[-1]
    write_mode = "w" if year == latest_year else "a"
    with open(results_txt, write_mode, encoding="utf-8") as f:
        f.write(
            f"Year: {year} | "
            f"AvgRet: {metrics.get('avg_return_pct', 0):.2f}% | "
            f"AvgHold: {metrics.get('avg_hold_days', 0):.1f}d | "
            f"Trades: {metrics.get('num_trades', 0)} | "
            f"Total: {metrics.get('est_total_return', 0):.2f}%\n"
        )
    
    # --- Update Overall Results Overview ---
    overview_path = os.path.join(OUTPUT_DIR, ticker_folder, "results_overview.txt")
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
            f"Total: {metrics.get('est_total_return', 0):.2f}%\n"
        )


    # --- Append total componded/average annual once after oldest year --- 
    years_sorted = sorted(all_df["Date"].dt.year.unique())
    first_year, last_year = years_sorted[0], years_sorted[-1]
    backtest_year._compounded_return *= (1 + metrics["est_total_return"] / 100)

    if year == years_sorted[0] and not backtest_year._results_written:
        total_return_pct = (backtest_year._compounded_return - 1) * 100
        avg_annual_return = (backtest_year._compounded_return ** (1 / len(years_sorted)) - 1) * 100

        with open(overview_path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"Total Compounded Return ({first_year}-{last_year}): {total_return_pct:.2f}%\n")
            f.write(f"Avg Annual Return: {avg_annual_return:.2f}%\n")
        backtest_year._results_written = True
        
    return trades_df, equity, metrics