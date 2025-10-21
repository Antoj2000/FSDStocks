import os
import pandas as pd

def load_csvs(paths):
    frames = []
    for p in paths: 
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")
        df = pd.read_csv(p)
         # --- Clean & normalize column names ---
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        required = {"Date", "Adj_Close", "Close", "High", "Low", "Open", "Volume"}

        if not required.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns in {p}. Required: {required}")
        if "Ticker" not in df.columns:
            df["Ticker"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    
    for c in ["Adj_Close", "Close", "High", "Low", "Open", "Volume"]:
        all_df[c] = pd.to_numeric(all_df[c], errors='coerce')
    all_df.dropna(subset=["Adj_Close"], inplace=True)
    all_df.sort_values(["Date", "Ticker"], inplace=True)
    return all_df.reset_index(drop=True)