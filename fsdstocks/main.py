import argparse
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from fsdstocks.data_loader import load_csvs
from fsdstocks.strategy import optimize_params_for_year
from fsdstocks.backtest import backtest_year
from fsdstocks.utils import DEFAULT_TEST_YEARS, TRAIN_YEAR, OUTPUT_DIR

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="FSDStocks backtester")
    parser.add_argument("csvs", nargs="+", help="Paths to CSV files")
    parser.add_argument("--years", nargs="*", type=int, default=DEFAULT_TEST_YEARS, help="Years to backtest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stocks to test (e.g., 25 or 50)")
    parser.add_argument("--quiet", action="store_true", help="Run without printing per-stock output")
    parser.add_argument("--strategy", type=str, default="ma_rsi_combo", help="Name of strategy to use (from strategies.yaml)")
    args = parser.parse_args()

    # --- Load strategies.yaml ---
    import yaml

    yaml_path = os.path.join(os.path.dirname(__file__), "strategies.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Missing strategies.yaml at {yaml_path}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if "strategies" not in yaml_data:
        raise ValueError("strategies.yaml missing top-level 'strategies' key")

    strategies = yaml_data["strategies"]
    selected_strategy = strategies.get(args.strategy)
    if not selected_strategy:
        raise ValueError(f"Strategy '{args.strategy}' not found in strategies.yaml")

    print(f"\n🧠 Loaded strategy config: {args.strategy}")
    print(f"   Description: {selected_strategy.get('description', 'No description')}")

    # --- Limit ticker list if requested ---
    all_csvs = args.csvs[:args.limit] if args.limit else args.csvs
    print(f"\n📊 Loaded {len(all_csvs)} stock files for testing")

    # Deduplicate and sort years
    years = list(dict.fromkeys(args.years))

    # --- Strategy-specific output directory ---
    strategy_name = args.strategy
    strategy_dir = os.path.join(OUTPUT_DIR, strategy_name)
    os.makedirs(strategy_dir, exist_ok=True)

    summary_path = os.path.join(strategy_dir, "summary_results.txt")
    runtime_log_path = os.path.join(strategy_dir, "runtime_log.txt")

    # clean previous summary if it exists
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # --- Train and optimize strategy parameters ---
    print(f"Optimizing parameters for year {TRAIN_YEAR}...")
    all_df = load_csvs(all_csvs)
    params = optimize_params_for_year(all_df, TRAIN_YEAR, strategy_name)
    print("Best parameters:", params)

    # --- Global trackers ---
    total_trades = 0
    tickers_tested = len(all_csvs)
    yearly_returns = []
    win_years = 0
    stock_equities = {}
    ticker_wins = {os.path.splitext(os.path.basename(p))[0]: 0 for p in all_csvs}


    # --- Run backtests year by year (portfolio-style average) ---
    for yr in years:
        print(f"\n=== Running backtests for year {yr} across all stocks ===")
        year_returns = []
        trades_this_year = 0
        profitable_stocks = 0
        total_stocks = 0

        for csv_path in all_csvs:
            ticker = os.path.splitext(os.path.basename(csv_path))[0]
            try:
                stock_df = load_csvs([csv_path])
                _, _, metrics = backtest_year(stock_df, yr, params, tested_years=years, strategy_cfg=selected_strategy, strategy_name=args.strategy)
                yr_ret = metrics.get("est_total_return", 0.0)
                trades_this_year += metrics.get("num_trades", 0)
                total_stocks += 1
                year_returns.append(yr_ret)

                # --- Update per-stock compounded equity ---
                if ticker not in stock_equities:
                    stock_equities[ticker] = 1.0
                stock_equities[ticker] *= (1 + yr_ret / 100)

                if yr_ret > 0:
                    profitable_stocks += 1
                
                ticker_wins[ticker] += 1 if yr_ret > 0 else 0

                if not args.quiet:
                    print(
                        f"  {ticker}: AvgRet={metrics['avg_return_pct']:.2f}% | "
                        f"Trades={metrics['num_trades']} | YearTotal={yr_ret:.2f}%"
                    )
            except Exception as e:
                if not args.quiet:
                    print(f"  {ticker}: backtest failed ({e})")

        # Compute average across tickers for that year (no compounding)
        if total_stocks > 0:
            avg_yearly_return = np.mean(year_returns)
            yearly_returns.append(avg_yearly_return)
            total_trades += trades_this_year
            if profitable_stocks / total_stocks > 0.5:
                win_years += 1
            if not args.quiet:
                print(f"→ Year {yr} portfolio return: {avg_yearly_return:.2f}%\n")
        else:
            if not args.quiet:
                print(f"→ Year {yr}: no valid data for any stock\n")
        

    # --- Debug yearly returns used for global compounding ---
    clean_returns = [float(r) for r in yearly_returns]
    #print(f"\n[DEBUG] Yearly portfolio returns used for compounding: {clean_returns}")

    # --- Compute overall portfolio compounding properly ---
    if yearly_returns:
        global_equity = 1.0
        profitable_years = 0
        for yr_ret in yearly_returns:
            global_equity *= (1 + yr_ret / 100)
            if yr_ret > 0:
                profitable_years += 1

        total_return_pct = (global_equity - 1) * 100
        avg_annual_return = (global_equity ** (1 / len(yearly_returns)) - 1) * 100
        win_rate = (profitable_years / len(yearly_returns) * 100)
    else:
        total_return_pct, avg_annual_return, win_rate = 0, 0, 0

    avg_trades_per_stock = total_trades / tickers_tested if tickers_tested > 0 else 0

    # --- Rank top/bottom stocks ---
    stock_perf = {t: (eq - 1) * 100 for t, eq in stock_equities.items()}
    sorted_perf = sorted(stock_perf.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_perf[:5]
    bottom5 = sorted_perf[-5:] if len(sorted_perf) > 5 else sorted_perf

    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_runtime = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    # === Multi-stock Performance Graph ===
    if stock_equities:
        plt.figure(figsize=(10, 6))

        tickers = list(stock_equities.keys())
        total_returns = [(eq - 1) * 100 for eq in stock_equities.values()]

        plt.bar(tickers, total_returns, color="skyblue", edgecolor="black")
        plt.title(f"Overall Performance by Stock ({args.strategy})")
        plt.ylabel("Total Return (%)")
        plt.xlabel("Ticker")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        strategy_perf_plot = os.path.join(strategy_dir, "overall_stock_performance.png")
        plt.savefig(strategy_perf_plot, dpi=150)
        plt.close()
        print(f"📈 Saved multi-stock performance chart: {strategy_perf_plot}")

    if yearly_returns:
        equity = [1.0]
        for r in yearly_returns:
            equity.append(equity[-1] * (1 + r / 100))

        plt.figure(figsize=(10, 5))
        if len(years) == len(equity[1:]):
            plt.plot(years, equity[1:], marker="o", linewidth=2)
        else:
            # Only plot the available data
            truncated_years = years[:len(equity[1:])]
            plt.plot(truncated_years, equity[1:], marker="o", linewidth=2)
            print(f"[WARN] Adjusted equity plot to match available years ({len(truncated_years)} years).")

        plt.title(f"Portfolio Equity Growth ({args.strategy})")
        plt.xlabel("Year")
        plt.ylabel("Equity (Starting = 1.0)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        portfolio_equity_plot = os.path.join(strategy_dir, "portfolio_equity_growth.png")
        plt.savefig(portfolio_equity_plot, dpi=150)
        plt.close()
        print(f"📈 Saved portfolio equity growth chart: {portfolio_equity_plot}")


    
    # --- Write global summary ---
    # Extract readable strategy details (uses selected_strategy from YAML)
    strategy_cfg = selected_strategy or {}
    strategy_title = args.strategy or "Unknown Strategy"
    description = strategy_cfg.get("description", "No description provided")
    risk = strategy_cfg.get("risk", {})
    stop_loss = risk.get("stop_loss_pct", 5)
    take_profit = risk.get("take_profit_pct", 10)
    take_profit_str = "None" if take_profit is None else f"{take_profit}%"

    # 🧠 Include parameter info (from optimize_params_for_year)
    param_info = ""
    try:
        if hasattr(params, "__dict__"):
            params_dict = vars(params)
            param_pairs = [f"{k}={v}" for k, v in params_dict.items()]
            param_info = " | ".join(param_pairs)
        else:
            param_info = str(params)
    except Exception as e:
        param_info = f"[Could not extract params: {e}]"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FSDStocks - Global Backtest Summary\n")
        f.write("====================================\n")
        f.write(f"Strategy: {strategy_title} ({description})\n")
        f.write(f"Stop-loss: {stop_loss}% | Take-profit: {take_profit_str}\n\n")
        if param_info:
            f.write(f"Parameters Used: {param_info}\n")
        f.write("\n")
        f.write(f"Stocks Tested: {tickers_tested}\n")
        f.write(f"Years Covered: {len(yearly_returns)}\n\n")
        f.write(f"Total Compounded Return: {total_return_pct:.2f}%\n")
        f.write(f"Average Annual Return: {avg_annual_return:.2f}%\n")
        f.write(f"Win Rate (profitable years): {win_rate:.2f}%\n")
        f.write(f"Average Trades per Stock: {avg_trades_per_stock:.1f}\n\n")

        f.write("Yearly Portfolio Returns (%):\n")
        for yr, ret in zip(years, yearly_returns):
            f.write(f"  {yr}: {ret:.2f}%\n")

        f.write("\nTop 5 Performing Stocks:\n")
        for t, r in top5:
            f.write(f"  {t:<10} {r:>8.2f}%\n")

        f.write("\nBottom 5 Performing Stocks:\n")
        for t, r in bottom5:
            f.write(f"  {t:<10} {r:>8.2f}%\n")

        f.write(f"\nTotal Runtime: {formatted_runtime}\n")

    # --- Write runtime log (appending, not overwriting) ---
    with open(runtime_log_path, "a", encoding="utf-8") as log:
        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        log.write(
            f"[{end_str}] Completed backtest on {tickers_tested} stocks "
            f"over {len(yearly_returns)} years | Runtime: {formatted_runtime}\n"
        )

    print("\n✅ Global summary written to:", {summary_path})
    print(f"📁 Strategy results saved under: {strategy_dir}")
    print(f"Total Compounded Return: {total_return_pct:.2f}%")
    print(f"Average Annual Return: {avg_annual_return:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")

    print("\n🏆 Top 5 performers:")
    for t, r in top5:
        print(f"  {t}: {r:.2f}%")

    print("\n💀 Bottom 5 performers:")
    for t, r in bottom5:
        print(f"  {t}: {r:.2f}%")

    
    print(f"\n⏱️ Total runtime: {formatted_runtime}")
    print(f"🧾 Runtime log updated: {runtime_log_path}")

    # === Strategy Scoped performance CSV ===

    # The strategy "name" comes from the YAML block; fall back to args.strategy
    strategy_title = selected_strategy.get("name", args.strategy)
    strategy_dir = os.path.join(OUTPUT_DIR, strategy_title)
    os.makedirs(strategy_dir, exist_ok=True)
    performance_csv = os.path.join(strategy_dir, "strategy_performance.csv")

    import csv
    from datetime import datetime

    n_years = len(yearly_returns) if yearly_returns else 0
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Create / Load CSV safely ---
    headers = ["Strategy", "Ticker", "TotalReturn(%)", "AvgAnnual(%)", "WinRate(%)","Timestamp"]
    rows = []

    # Load existing file if it exists
    if os.path.exists(performance_csv):
        try:
            existing = pd.read_csv(performance_csv)
        except Exception:
            existing = pd.DataFrame(columns=headers)
    else:
        existing = pd.DataFrame(columns=headers)

    # --- Prepare new results for this run ---
    new_data = []
    for ticker, eq in stock_equities.items():
        total_return_pct_t = (eq - 1.0) * 100.0
        avg_annual_pct_t = ((eq ** (1.0 / n_years) - 1.0) * 100.0) if n_years > 0 else 0.0
        win_rate_t = (ticker_wins.get(ticker, 0) / n_years * 100.0) if n_years > 0 else 0.0

        new_data.append({
            "Strategy": strategy_title,
            "Ticker": ticker,
            "TotalReturn(%)": round(total_return_pct_t, 2),
            "AvgAnnual(%)": round(avg_annual_pct_t, 2),
            "WinRate(%)": round(win_rate_t, 2),
            "Timestamp": now_ts
        })

    # --- Add portfolio summary row ---
    portfolio_row = {
        "Strategy": strategy_title,
        "Ticker": "PORTFOLIO_AVG",
        "TotalReturn(%)": round(total_return_pct, 2),
        "AvgAnnual(%)": round(avg_annual_return, 2),
        "WinRate(%)": round(win_rate, 2),
        "Timestamp": now_ts
    }
    new_data.append(portfolio_row)

    new_df = pd.DataFrame(new_data, columns=headers)

    # --- Drop old entries for same (Strategy, Ticker) and append new ones ---
    merged = (
        pd.concat([existing, new_df], ignore_index=True)
        .sort_values("Timestamp")
        .drop_duplicates(subset=["Strategy", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # --- Ensure PORTFOLIO_AVG row stays on top ---
    merged = pd.concat(
        [
            merged[merged["Ticker"] == "PORTFOLIO_AVG"],
            merged[merged["Ticker"] != "PORTFOLIO_AVG"]
        ],
        ignore_index=True
    )

    merged.to_csv(performance_csv, index=False)
    print(f"📊 Updated strategy performance CSV: {performance_csv}")

    # --- Optionally append all runs to history file ---
    history_path = os.path.join(strategy_dir, "strategy_performance_history.csv")
    new_df.to_csv(
        history_path,
        mode="a",
        header=not os.path.exists(history_path),
        index=False
    )


if __name__ == "__main__":
    main()