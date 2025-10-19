import argparse
import os
import numpy as np
import pandas as pd
import time
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
    args = parser.parse_args()

    # --- Limit ticker list if requested ---
    all_csvs = args.csvs[:args.limit] if args.limit else args.csvs
    print(f"\n📊 Loaded {len(all_csvs)} stock files for testing")

    # Deduplicate and sort years
    years = list(dict.fromkeys(args.years))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary_results.txt")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # --- Train and optimize strategy parameters ---
    print(f"Optimizing parameters for year {TRAIN_YEAR}...")
    all_df = load_csvs(all_csvs)
    params = optimize_params_for_year(all_df, TRAIN_YEAR)
    print("Best parameters:", params)

    # --- Global trackers ---
    total_trades = 0
    tickers_tested = len(all_csvs)
    yearly_returns = []
    win_years = 0
    stock_equities = {}

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
                _, _, metrics = backtest_year(stock_df, yr, params, tested_years=years)
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

    # --- Debug yearly returns used for global compounding ---
    clean_returns = [float(r) for r in yearly_returns]
    print(f"\n[DEBUG] Yearly portfolio returns used for compounding: {clean_returns}")

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

    # --- Write global summary ---
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FSDStocks - Global Backtest Summary\n")
        f.write("====================================\n")
        f.write("Strategy: MA crossover + RSI momentum + volatility filter\n")
        f.write("Stop-loss: 5% | Take-profit: 10%\n\n")
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
    runtime_log_path = os.path.join(OUTPUT_DIR, "runtime_log.txt")
    with open(runtime_log_path, "a", encoding="utf-8") as log:
        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        log.write(
            f"[{end_str}] Completed backtest on {tickers_tested} stocks "
            f"over {len(yearly_returns)} years | Runtime: {formatted_runtime}\n"
        )

    print("\n✅ Global summary written to:", summary_path)
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


if __name__ == "__main__":
    main()