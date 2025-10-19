import argparse
import os
import numpy as np
import pandas as pd
from fsdstocks.data_loader import load_csvs
from fsdstocks.strategy import optimize_params_for_year
from fsdstocks.backtest import backtest_year
from fsdstocks.utils import DEFAULT_TEST_YEARS, TRAIN_YEAR, OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description="FSDStocks backtester")
    parser.add_argument("csvs", nargs="+", help="Paths to CSV files")
    parser.add_argument("--years", nargs="*", type=int, default=DEFAULT_TEST_YEARS, help="Years to backtest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stocks to test (e.g., 25 or 50)")
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

    # --- Train and optimize strategy parameters
    print(f"Optimizing parameters for year {TRAIN_YEAR}...")
    all_df = load_csvs(all_csvs)
    params = optimize_params_for_year(all_df, TRAIN_YEAR)
    print("Best parameters: ", params)

    # --- Global trackers ---
    total_equity = 1.0
    total_trades = 0
    tickers_tested = len(all_csvs)
    yearly_returns = []
    win_years = 0
    stock_equities = {}


    # --- Run backtests year by year (aggregated portfolio approach) ---
    for yr in years:
        print(f"\n=== Running backtests for year {yr} across all stocks ===")
        year_equity = 1.0
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
                
                # --- Update per-stock compounded equity ---
                if ticker not in stock_equities:
                    stock_equities[ticker] = 1.0
                stock_equities[ticker] *= (1 + yr_ret / 100)


                year_equity *= (1 + yr_ret / 100)
                total_stocks += 1
                if yr_ret > 0:
                    profitable_stocks += 1
                print(
                    f"  {ticker}: AvgRet={metrics['avg_return_pct']:.2f}% | "
                    f"Trades={metrics['num_trades']} | YearTotal={yr_ret:.2f}%"
                )
            except Exception as e:
                print(f"  {ticker}: backtest failed ({e})")

        # Compute average across tickers for that year
        if total_stocks > 0:
            avg_yearly_return = ((year_equity - 1) * 100)
            yearly_returns.append(avg_yearly_return)
            total_equity *= (1 + avg_yearly_return / 100)
            total_trades += trades_this_year
            if profitable_stocks / total_stocks > 0.5:
                win_years += 1

        print(f"→ Year {yr} portfolio return: {avg_yearly_return:.2f}%\n")

    
    # --- Compute overall performance ---
    years_covered = len(years)
    total_return_pct = (total_equity - 1) * 100
    avg_annual_return = (total_equity ** (1 / years_covered) - 1) * 100 if years_covered > 0 else 0
    win_rate = (win_years / years_covered * 100) if years_covered > 0 else 0
    avg_trades_per_stock = total_trades / tickers_tested if tickers_tested > 0 else 0

    # --- Rank top/bottom stocks ---
    stock_perf = {t: (eq - 1) * 100 for t, eq in stock_equities.items()}
    sorted_perf = sorted(stock_perf.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_perf[:5]
    bottom5 = sorted_perf[-5:] if len(sorted_perf) > 5 else sorted_perf

    # --- Write global summary ---
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FSDStocks - Global Backtest Summary\n")
        f.write("====================================\n")
        f.write("Strategy: MA crossover + RSI momentum + volatility filter\n")
        f.write("Stop-loss: 5% | Take-profit: 10%\n\n")
        f.write(f"Stocks Tested: {tickers_tested}\n")
        f.write(f"Years Covered: {years_covered}\n\n")
        f.write(f"Total Compounded Return: {total_return_pct:.2f}%\n")
        f.write(f"Average Annual Return: {avg_annual_return:.2f}%\n")
        f.write(f"Win Rate (profitable years): {win_rate:.2f}%\n")
        f.write(f"Average Trades per Stock: {avg_trades_per_stock:.1f}\n")

        # --- Append top/bottom performers ---
        f.write("Top 5 Performing Stocks:\n")
        for t, r in top5:
            f.write(f"  {t:<10} {r:>8.2f}%\n")

        f.write("\nBottom 5 Performing Stocks:\n")
        for t, r in bottom5:
            f.write(f"  {t:<10} {r:>8.2f}%\n")

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


if __name__ == "__main__":
    main()