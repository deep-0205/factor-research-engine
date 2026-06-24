import pandas as pd
import yfinance as yf
import os
import yaml
from logger import get_logger
from data.fetch_universe import get_nifty500_tickers

logger = get_logger("download_data")

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
def download_data(tickers: list, start: str, end: str, raw_path: str):
    os.makedirs(raw_path, exist_ok=True)
    failed = []
    for i, ticker in enumerate(tickers):
        save_path = os.path.join(raw_path, f"{ticker}.csv")
        if os.path.exists(save_path):
            logger.info(f"[{i+1}/{len(tickers)}] Already exists: {ticker}")
            continue
        try: 
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
            if df.empty:
                logger.warning(f"No data for: {ticker}")
                failed.append(ticker)
                continue
            df.to_csv(save_path)
            logger.info(f"[{i+1}/{len(tickers)}] Saved: {ticker}")
        except Exception as e:
            logger.error(f"Error ownloading data for {ticker}: {e}")
            failed.append(ticker)
    logger.info(F"Download Complete. Failed: {len(failed)} tickers.")
    if failed:
        pd.Series(failed).to_csv(os.path.join(raw_path, "failed_tickers.csv"), index=False)

