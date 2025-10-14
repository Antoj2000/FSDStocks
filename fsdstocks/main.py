import argparse
from fsdstocks.data_loader import load_csvs
from fsdstocks.strategy import optimize_params_for_year
from fsdstocks.backtest import backtest_year
from fsdstocks.utils import DEFAULT_TEST_YEARS, TRAIN_YEAR 

def main():
    parser = argparse.ArgumentParser(description="FSDStocks backtester")
    parser.add_argument("csvs", nargs="+", help="Paths to CSV files")
    parser.add_argument("--years", nargs="*", type=int, default=DEFAULT_TEST_YEARS, help="Years to backtest")
    args = parser.parse_args()

    all_df = load_csvs(args.csvs)

    print(f"Optimizing parameters for year {TRAIN_YEAR}...")
    params = optimize_params_for_year(all_df, TRAIN_YEAR)
    print("Best parameters: ", params)

    for yr in args.years:
        print(f"Running backtest for year {yr}...")
        try:
            trades_df, equity, metrics = backtest_year(all_df, yr, params)
            print(f"Year {yr}: AvgRet={metrics['avg_return_pct']:.2f}% | AvgHold={metrics['avg_hold_days']:.1f}d | Trades={metrics['num_trades']} | EstTotal={metrics['est_total_return']:.2f}%")
        except Exception as e:
            print(f"Year {yr} backtest failed: {e}")

if __name__ == "__main__":
    main()