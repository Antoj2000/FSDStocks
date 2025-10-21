from dataclasses import dataclass
import numpy as np
import pandas as pd
from fsdstocks.indicators import compute_indicators
from fsdstocks.utils import TRAIN_YEAR, run_backtest_equity_only, score_equity 

@dataclass
class StrategyParams:
    ma_fast: int = 50
    ma_slow: int = 200
    rsi_buy: float = 55.0
    rsi_sell: float = 45.0
    rsi_window: int = 14

# ============================================================
# ================ SIGNAL GENERATION CORE ====================
# ============================================================

def generate_signals(df, params: StrategyParams, strategy_cfg=None):
    # --- Commpute trading signals for a single ticker ---

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.columns = df.columns.str.replace(" ", "_")

    # 🧹 QUIET MODE: global toggle for verbosity
    import sys, os
    verbose = not (
        "--quiet" in sys.argv or 
        os.getenv("QUIET", "false").lower() == "true"
    )
    # === CASE 1: YAML-driven strategy config ===
    if strategy_cfg:
        if verbose:
            print(f"[DEBUG] Using YAML strategy: {strategy_cfg.get('name', '?')}")
        try: 
            indicators = strategy_cfg.get("indicators", [])
            rules = strategy_cfg.get("rules", {})

            ind_kwargs = {}

            #=== Compute all declared indicators === 
            for ind in indicators:
                t = ind["type"].lower()
                p = ind.get("params", {})

                # --- Moving Average ---
                if t == "ma":
                    fast_param = p.get("fast", params.ma_fast)
                    slow_param = p.get("slow", params.ma_slow)
                     # 🩵 FIX: enforce numeric and distinct MAs
                    ind_kwargs["ma_fast"] = int(params.ma_fast or fast_param)
                    ind_kwargs["ma_slow"] = int(params.ma_slow or slow_param)
                    if ind_kwargs["ma_fast"] == ind_kwargs["ma_slow"]:
                        ind_kwargs["ma_slow"] = ind_kwargs["ma_fast"] * 10

                # --- RSI ---
                elif t == "rsi":
                    ind_kwargs["rsi_window"] = p.get("window", 14)
                    df["rsi_buy"] = p.get("buy", params.rsi_buy)
                    df["rsi_sell"] = p.get("sell", params.rsi_sell)

                # --- Bollinger --- 
                elif t == "bollinger":
                    window = p.get("window", 20)
                    std = p.get("stddev", p.get("num_std", 2))
                    price = df["Adj_Close"]
                    ma = price.rolling(window).mean()
                    stddev = price.rolling(window).std()
                    df["Middle_Band"] = ma
                    df["Upper_Band"] = ma + std * stddev
                    df["Lower_Band"] = ma - std * stddev

                # --- Volatility (rolling range) ---
                elif t == "volatility":
                    # Handled directly, since not part of compute_indicators()
                    w = p.get("window", 14)
                    mult = p.get("threshold_mult", 0.5)
                    df["Volatility"] = df["High"].rolling(w).max() - df["Low"].rolling(w).min()
                    df["VolThresh"] = df["Volatility"].rolling(50).mean() * mult
                    df["VolOK"] = df["Volatility"] > df["VolThresh"]

                # --- High/Low breakout ---
                elif t == "highlow":
                    w = p.get("window", 20)
                    df[f"High_{w}"] = df["High"].rolling(w, min_periods=1).max()
                    df[f"Low_{w}"] = df["Low"].rolling(w, min_periods=1).min()
                    df[f"High_{w}_prev"] = df[f"High_{w}"].shift(1)

                else:
                    raise ValueError(f"Unknown indicator type: {t}")
                
            # 🧹 CLEANUP: Only show compute message in verbose mode
            if any(k in ind_kwargs for k in ["ma_fast", "ma_slow", "rsi_window"]):
                if verbose:
                    print(f"[INFO] Calling compute_indicators with {ind_kwargs}")
                df = compute_indicators(df, **ind_kwargs)

            # --- Safe rule evaluation ---
            df["signal"] = 0
            buy_rule = rules.get("buy", "")
            sell_rule = rules.get("sell", "")

            
            fast_col = f"MA{ind_kwargs['ma_fast']}"
            slow_col = f"MA{ind_kwargs['ma_slow']}"
            if fast_col in df.columns:
                df["MA_fast"] = df[fast_col].copy()
            if slow_col in df.columns:
                df["MA_slow"] = df[slow_col].copy()

             # 🧹 CLEANUP: remove column dumps unless verbose
            if verbose:
                print(f"[INFO] Aliases → MA_fast={fast_col}, MA_slow={slow_col}")
                print(df[[fast_col, slow_col]].head(5))

            # Inject safe context (columns + params)
            safe_context = {c: df[c] for c in df.columns if c not in ["signal"]}
            safe_context.update({
                "RSI": df.get("RSI"),
                "Adj_Close": df.get("Adj_Close"),
                "rsi_buy": float(params.rsi_buy),
                "rsi_sell": float(params.rsi_sell),
            })

             # auto-detect breakout column
            for c in df.columns:
                if "High_" in c and "_prev" in c:
                    safe_context["breakout_col"] = df[c]
                    break
   
            def safe_eval(rule: str):
                if not rule:
                    return pd.Series(False, index=df.index)
                try:
                    return eval(rule, {}, safe_context).fillna(False)
                except Exception as e:
                    if verbose:
                        print(f"[YAML STRATEGY ERROR] {e}")
                    return pd.Series(False, index=df.index)

            buy_mask = safe_eval(buy_rule)
            sell_mask = safe_eval(sell_rule)

            df.loc[buy_mask, "signal"] = 1
            # Only reset sell signals where already in position
            df.loc[sell_mask & (df["signal"] == 1), "signal"] = 0

            # --- Debug how many buys/sells were triggered ---
            num_buys = (buy_mask).sum()
            num_sells = (sell_mask).sum()
            # 🧹 CLEANUP: concise yearly log
            ticker = df.get("Ticker", ["?"])[0]
            if verbose:
                print(f"  {ticker}: {num_buys} buys | {num_sells} sells | {len(df)} rows")

            # --- Dynamically detect available breakout columns ---
            high_cols = [c for c in df.columns if "High_" in c and "_prev" in c]
            last_high = high_cols[0] if high_cols else None

            # 🧹 REMOVE noisy prints: tail tables, diagnostics
            # keep only when verbose
            if verbose:
                print(f"Sample MA/RSI preview:\n{df[['Date','Adj_Close','MA_fast','MA_slow','RSI']].tail(3)}")

            return df
        except Exception as e:
            if verbose:
                print(f"[YAML STRATEGY ERROR] {e}")
            raise

    # === Case 2: default hardcoded logic ----
    d = compute_indicators(df, params.ma_fast, params.ma_slow, params.rsi_window)
    fast, slow = d[f"MA{params.ma_fast}"], d[f"MA{params.ma_slow}"]

    # --- Normalize column names to underscores ---
    d.columns = d.columns.str.replace(" ", "_")

    # --- Trend Filter (long-term direction) ---
    d["MA200"] = d["Adj_Close"].rolling(200, min_periods=1).mean()
    uptrend = d["Adj_Close"] > (d["MA200"] * 0.95)

    # --- Volatility Filter (ATR proxy using rolling range) --- 
    d["Volatility"] = d["High"].rolling(14, min_periods=1).max() - d["Low"].rolling(14, min_periods=1).min()
    vol_thresh = d["Volatility"].rolling(50, min_periods=1).mean() * 0.5
    vol_ok = d["Volatility"] > vol_thresh

    # --- RSI Momentum Confirmation ---
    rsi = d["RSI"]
    
    # --- Core Buy/Sell Conditions ---
    buy_condition = (
        (fast > slow) & 
        uptrend & 
        (rsi > params.rsi_buy - 15) &
        vol_ok
    )

    sell_condition = (
        (fast < slow) | 
        (rsi < params.rsi_sell) |
        (~uptrend)
    )

    d["signal"] = 0
    d.loc[buy_condition, "signal"] = 1
    d.loc[sell_condition, "signal"] = 0

    return d

# === Paramter Optimization Logic ===

def optimize_params_for_year(all_df, year=TRAIN_YEAR, strategy_name=None):
    # --- pick training years for robustness (use year-1 if present) ---
    print(f"⚙️ optimize_params_for_year() running from: {__file__}")
    available_years = sorted(all_df["Date"].dt.year.unique().tolist())
    train_years = [y for y in [year - 1, year] if y in available_years]
    if not train_years:
        train_years = [year]

    rng = np.random.RandomState(42)  # reproducible

    def evaluate_params(p: StrategyParams) -> float:
        scores = []
        for y in train_years:
            year_df = all_df[all_df["Date"].dt.year == y]
            if year_df.empty:
                continue
            eq = run_backtest_equity_only(year_df, p)
            scores.append(score_equity(eq))
        return float(np.mean(scores)) if scores else -np.inf

    best_score = -np.inf
    best_params = StrategyParams()


     # === Strategy-specific search spaces ===
    if strategy_name == "bollinger_rsi":
        fast_list, slow_list = [10, 20], [50, 100]
        buy_vals, sell_vals = [30, 35, 40, 45], [55, 60, 65, 70]
        # simple grid (unchanged)
        for f in fast_list:
            for s in slow_list:
                if f >= s: 
                    continue
                for rb in buy_vals:
                    for rs in sell_vals:
                        p = StrategyParams(ma_fast=f, ma_slow=s, rsi_buy=rb, rsi_sell=rs)
                        score = evaluate_params(p)
                        if score > best_score:
                            best_score, best_params = score, p

    elif strategy_name == "mean_reversion_rsi":
        fast_list, slow_list = [10], [50]
        buy_vals, sell_vals = [25, 30, 35], [60, 65, 70]
        for f in fast_list:
            for s in slow_list:
                if f >= s: 
                    continue
                for rb in buy_vals:
                    for rs in sell_vals:
                        p = StrategyParams(ma_fast=f, ma_slow=s, rsi_buy=rb, rsi_sell=rs)
                        score = evaluate_params(p)
                        if score > best_score:
                            best_score, best_params = score, p

    elif strategy_name == "momentum_breakout":
        fast_list, slow_list = [10, 20, 30], [50, 100]
        buy_vals, sell_vals = [55, 60, 65], [35, 40, 45]
        for f in fast_list:
            for s in slow_list:
                if f >= s: 
                    continue
                for rb in buy_vals:
                    for rs in sell_vals:
                        p = StrategyParams(ma_fast=f, ma_slow=s, rsi_buy=rb, rsi_sell=rs)
                        score = evaluate_params(p)
                        if score > best_score:
                            best_score, best_params = score, p

    elif strategy_name == "momentum_rsi_breakout":
        print("🔍 Enhanced single-year parallel optimizer for momentum_rsi_breakout")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        rng = np.random.default_rng(42)

        # --- Search spaces ---
        ma_fast_choices = np.arange(3, 25, 2)
        ma_slow_choices = np.arange(50, 200, 10)
        rsi_buy_choices = np.arange(48, 70, 2)
        rsi_sell_choices = np.arange(25, 45, 2)

        # --- Restrict to target training year ---
        year_df = all_df[all_df["Date"].dt.year == year].copy()
        if year_df.empty:
            print(f"[ERROR] No data found for {year}")
            return StrategyParams()

        def evaluate_params(p: StrategyParams) -> float:
            eq = run_backtest_equity_only(year_df, p)
            # skip empty/flat equity curves
            if eq is None or len(eq) < 2 or eq.diff().abs().sum() < 1e-3:
                return -9999.0
            score = score_equity(eq)
            if np.isnan(score):
                score = -9999.0
            return float(score)

        # === Round 1: broad parallel exploration ===
        print("🧭 Exploring parameter space (parallel random search 150)...")

        # generate random candidates
        candidates = []
        for _ in range(150):
            f = rng.choice(ma_fast_choices)
            s = rng.choice(ma_slow_choices)
            rb = rng.choice(rsi_buy_choices)
            rs = rng.choice(rsi_sell_choices)
            if f >= s or rb <= rs + 3:
                continue
            candidates.append(StrategyParams(f, s, rb, rs))

        samples = []
        with ThreadPoolExecutor() as ex:
            futures = {ex.submit(evaluate_params, p): p for p in candidates}
            for fut in as_completed(futures):
                sc = fut.result()
                if sc > -9999:
                    samples.append((sc, futures[fut]))

        if not samples:
            print("[WARN] All samples flat or invalid.")
            return StrategyParams()

        samples.sort(key=lambda x: x[0], reverse=True)
        top5 = samples[:5]

        print("🏁 Top 5 (Round 1):")
        for i, (sc, pr) in enumerate(top5):
            print(f"  {i+1}. {pr} → {sc:.4f}")

        best_score, best_params = top5[0]

        # === Round 2: parallel refinement around top 5 ===
        print("🔬 Refining around best candidates (parallel)...")
        refined_candidates = []
        for _, base in top5:
            for df in [-2, 0, 2]:
                for ds in [-10, 0, 10]:
                    for drb in [-2, 0, 2]:
                        for drs in [-2, 0, 2]:
                            f = max(3, base.ma_fast + df)
                            s = min(200, base.ma_slow + ds)
                            rb = np.clip(base.rsi_buy + drb, 48, 70)
                            rs = np.clip(base.rsi_sell + drs, 25, 45)
                            if f >= s or rb <= rs + 3:
                                continue
                            refined_candidates.append(StrategyParams(f, s, rb, rs))

        refined = []
        with ThreadPoolExecutor() as ex:
            futures = {ex.submit(evaluate_params, p): p for p in refined_candidates}
            for fut in as_completed(futures):
                sc = fut.result()
                if sc > -9999:
                    refined.append((sc, futures[fut]))

        refined.sort(key=lambda x: x[0], reverse=True)
        if refined and refined[0][0] > best_score:
            best_score, best_params = refined[0]

        print(f"🔧 Best Params → {best_params} | Score={best_score:.4f}")


    else:
        # --- default baseline (original grid) ---
        buy_vals = [45, 50, 55]
        sell_vals = [35, 40, 45]
        fast_list, slow_list = [5, 10, 20, 30, 50], [50, 100, 150, 200]
        for f in fast_list:
            for s in slow_list:
                if f >= s:
                    continue
                for rb in buy_vals:
                    for rs in sell_vals:
                        p = StrategyParams(ma_fast=f, ma_slow=s, rsi_buy=rb, rsi_sell=rs)
                        score = evaluate_params(p)
                        if score > best_score:
                            best_score, best_params = score, p

    return best_params