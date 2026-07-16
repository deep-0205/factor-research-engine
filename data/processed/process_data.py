import pandas as pd
import numpy as np
import os
from logger import get_logger

logger = get_logger("process_data")

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Close"])
    df = df.ffill(limit=3)
    return df

def build_close_matrix(raw_path: str, processed_path: str) -> pd.DataFrame:
    os.makedirs(processed_path, exist_ok=True)
    all_close = {}

    files = [f for f in os.listdir(raw_path) if f.endswith(".csv") and f != "failed_tickers.csv"]

    for file in files:
        ticker = file.replace(".csv", "")
        path   = os.path.join(raw_path, file)

        try:
            df = pd.read_csv(path, index_col=0)

            if df.index[0] in ["Ticker", "Price"]:
                df = df.iloc[1:]

            df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            df = df[df.index.notna()]

            df.columns = [str(c).strip() for c in df.columns]

            close_col = None
            for candidate in ["Close", "close", "Adj Close", "adjclose"]:
                if candidate in df.columns:
                    close_col = candidate
                    break

            if close_col is None:
                logger.warning(f"No Close column for {ticker}: "
                               f"{df.columns.tolist()}")
                continue

            series = pd.to_numeric(df[close_col], errors="coerce")
            series = series.dropna()

            if series.empty:
                logger.warning(f"Empty close series: {ticker}")
                continue

            all_close[ticker] = series

        except Exception as e:
            logger.warning(f"Could not process {ticker}: {e}")

    if not all_close:
        logger.error("No data loaded — check data/raw/")
        return pd.DataFrame()

    close_matrix = pd.DataFrame(all_close).sort_index()

    threshold    = 0.80
    close_matrix = close_matrix.dropna(
        thresh=int(threshold * len(close_matrix)), axis=1
    )

    save_path = os.path.join(processed_path, "close_prices.csv")
    close_matrix.to_csv(save_path)
    logger.info(
        f"Close matrix saved: {close_matrix.shape} → {save_path}"
    )
    return close_matrix


def compute_returns(close_matrix: pd.DataFrame, processed_path: str) -> pd.DataFrame:

    returns = np.log(
        close_matrix / close_matrix.shift(1)
    ).dropna(how="all")

    save_path = os.path.join(processed_path, "daily_returns.csv")
    returns.to_csv(save_path)
    logger.info(f"Returns matrix saved: {returns.shape}")
    return returns
