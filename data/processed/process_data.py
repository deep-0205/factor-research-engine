import pandas as pd
import os 
import yaml
from logger import get_logger

logger = get_logger("process_data")
def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna(subset=["Close"])
    df = df.ffill(limit=3)
    return df

def build_close_matrix(raw_path: str, processed_path: str) -> pd.DataFrame:
    os.makedirs(processed_path, exist_ok=True)
    all_close = {}

    files = [f for f in os.listdir(raw_path) if f.endswith(".csv") and f != "failed_tickers.csv"]

    for file in files:
        ticker = file.replace(".csv", "")
        path = os.path.join(raw_path, file)

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = clean_ohlcv(df)
            all_close[ticker] = df["Close"]
        except Exception as e:
            logger.warning(f"Could not process {ticker}: {e}")

    close_matrix = pd.DataFrame(all_close).sort_index()

    threshold = 0.80
    close_matrix = close_matrix.dropna(thresh=int(threshold * len(close_matrix)), axis=1)

    save_path = os.path.join(processed_path, "close_prices.csv")
    close_matrix.to_csv(save_path)
    logger.info(f"Close matrix saved: {close_matrix.shape} → {save_path}")

    return close_matrix

def compute_returns(close_matrix: pd.DataFrame, processed_path: str) -> pd.DataFrame:
    import numpy as np
    returns = np.log(close_matrix / close_matrix.shift(1)).dropna(how="all")

    save_path = os.path.join(processed_path, "daily_returns.csv")
    returns.to_csv(save_path)
    logger.info(f"Returns matrix saved: {returns.shape}")
    return returns
