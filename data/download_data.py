import yfinance as yf
import pandas as pd
import os
from logger import get_logger

logger = get_logger("download_data")


def download_ohlcv(tickers: list, start: str, end: str, raw_path: str):
    """
    Download OHLCV data for a list of tickers.
    
    CRITICAL:
    - Validates date range
    - Skips already-downloaded files to avoid redundant work
    - Saves failed tickers for later investigation
    - Handles multi-index columns from yfinance
    
    Args:
        tickers: List of stock tickers
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        raw_path: Directory to save CSV files
    """
    
    if not tickers:
        logger.error("Empty ticker list provided")
        return
    
    os.makedirs(raw_path, exist_ok=True)
    
    # Validate date range
    try:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        if start_dt >= end_dt:
            logger.error(f"Invalid date range: {start} >= {end}")
            return
    except Exception as e:
        logger.error(f"Invalid date format: {e}")
        return
    
    logger.info(
        f"Starting download: {len(tickers)} tickers | "
        f"Period: {start} to {end}"
    )
    
    failed = []
    skipped = 0

    for i, ticker in enumerate(tickers):
        save_path = os.path.join(raw_path, f"{ticker}.csv")

        # Skip if already exists
        if os.path.exists(save_path):
            logger.debug(f"[{i+1}/{len(tickers)}] Already exists: {ticker}")
            skipped += 1
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
                logger.warning(f"[{i+1}/{len(tickers)}] No data: {ticker}")
                failed.append(ticker)
                continue

            # Handle multi-index columns (when downloading multiple tickers at once)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Validate data before saving
            if "Close" not in df.columns:
                logger.warning(f"[{i+1}/{len(tickers)}] No Close column: {ticker}")
                failed.append(ticker)
                continue
            
            # Check for valid prices
            if (df["Close"] <= 0).any():
                logger.warning(f"[{i+1}/{len(tickers)}] Invalid prices: {ticker}")
                failed.append(ticker)
                continue

            df.to_csv(save_path)
            logger.info(f"[{i+1}/{len(tickers)}] Downloaded: {ticker} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"[{i+1}/{len(tickers)}] Error: {ticker}: {e}")
            failed.append(ticker)

    logger.info(
        f"Download complete | "
        f"Success: {len(tickers) - len(failed) - skipped} | "
        f"Skipped: {skipped} | "
        f"Failed: {len(failed)}"
    )
    
    if failed:
        failed_path = os.path.join(raw_path, "failed_tickers.csv")
        pd.Series(failed).to_csv(failed_path, index=False)
        logger.warning(f"Failed tickers saved to {failed_path}")