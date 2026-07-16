import yfinance as yf
import pandas as pd
import os
from logger import get_logger

logger = get_logger("download_data")


def download_ohlcv(tickers: list, start: str, end: str, raw_path: str):

    os.makedirs(raw_path, exist_ok=True)
    failed = []

    for i, ticker in enumerate(tickers):
        save_path = os.path.join(raw_path, f"{ticker}.csv")

        if os.path.exists(save_path):
            logger.info(f"[{i+1}/{len(tickers)}] Already exists: {ticker}")
            continue

        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                logger.warning(f"No data for: {ticker}")
                failed.append(ticker)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(save_path)
            logger.info(f"[{i+1}/{len(tickers)}] Saved: {ticker}")

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            failed.append(ticker)

    logger.info(f"Download Complete. Failed: {len(failed)} tickers.")
    if failed:
        pd.Series(failed).to_csv(
            os.path.join(raw_path, "failed_tickers.csv"), index=False
        )