import pandas as pd
import numpy as np
import os
from logger import get_logger

logger = get_logger("process_data")

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean individual stock OHLCV data.
    
    Operations:
    1. Remove duplicate dates (keep first)
    2. Require Close price for each row
    3. Forward-fill missing values up to 3 days max
    
    Args:
        df: OHLCV DataFrame with datetime index
        
    Returns:
        Cleaned DataFrame
    """
    
    if df.empty:
        logger.warning("Empty DataFrame provided to clean_ohlcv")
        return pd.DataFrame()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]
    
    # Remove rows with no Close price
    initial_rows = len(df)
    df = df.dropna(subset=["Close"])
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        logger.debug(f"Removed {removed_rows} rows with missing Close price")
    
    # Forward-fill missing values (max 3 days to avoid stale data)
    df = df.ffill(limit=3)
    
    # Check for excessive forward-fill usage
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.debug(f"Remaining NaNs after forward-fill: {nan_count}")
    
    return df


def build_close_matrix(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Build matrix of close prices from individual stock files.
    
    Process:
    1. Load all CSV files from raw_path
    2. Clean each stock's data
    3. Combine into single DataFrame
    4. Apply quality filters:
       - Keep only stocks with 80%+ of data
       - Remove dates before 2001
    5. Validate index and alignment
    
    Args:
        raw_path: Directory with individual stock CSVs
        processed_path: Directory to save processed data
        
    Returns:
        Close matrix (dates × tickers)
    """
    
    os.makedirs(processed_path, exist_ok=True)
    all_close = {}

    files = [f for f in os.listdir(raw_path) 
             if f.endswith(".csv") and f != "failed_tickers.csv"]

    if not files:
        logger.error(f"No CSV files found in {raw_path}")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} stock files from {raw_path}")

    for file_idx, file in enumerate(files):
        ticker = file.replace(".csv", "")
        path = os.path.join(raw_path, file)

        try:
            df = pd.read_csv(path, index_col=0)

            if df.empty:
                logger.warning(f"Empty file: {ticker}")
                continue

            # Handle header rows sometimes included in data
            if len(df) > 0 and df.index[0] in ["Ticker", "Price"]:
                df = df.iloc[1:]

            # Parse dates
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            df = df[df.index.notna()]
            
            if df.empty:
                logger.warning(f"No valid dates for {ticker}")
                continue

            # Standardize column names
            df.columns = [str(c).strip() for c in df.columns]

            # Find Close column (case-insensitive)
            close_col = None
            for candidate in ["Close", "close", "Adj Close", "adjclose"]:
                if candidate in df.columns:
                    close_col = candidate
                    break

            if close_col is None:
                logger.warning(f"No Close column for {ticker}: {df.columns.tolist()}")
                continue

            # Extract and validate close prices
            series = pd.to_numeric(df[close_col], errors="coerce")
            
            # Remove NaNs
            initial_len = len(series)
            series = series.dropna()
            
            if len(series) == 0:
                logger.warning(f"No valid close prices: {ticker}")
                continue
            
            # Validate prices are positive
            if (series <= 0).any():
                logger.warning(f"Non-positive prices in {ticker}: {(series <= 0).sum()} values")
                series = series[series > 0]
            
            if series.empty:
                logger.warning(f"All prices invalid after filtering: {ticker}")
                continue
            
            # Sort by date
            series = series.sort_index()
            
            all_close[ticker] = series

            if (file_idx + 1) % 50 == 0:
                logger.info(f"  Processed {file_idx + 1}/{len(files)} files")

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")
            continue

    if not all_close:
        logger.error("No data loaded from raw files — check data/raw/")
        return pd.DataFrame()

    logger.info(f"Successfully loaded {len(all_close)} stocks")

    # Build matrix
    close_matrix = pd.DataFrame(all_close).sort_index()
    
    initial_shape = close_matrix.shape
    logger.info(f"Initial matrix shape: {initial_shape}")

    # CRITICAL: Validate index is sorted and unique
    if not close_matrix.index.is_unique:
        logger.warning(f"Duplicate dates in index: {close_matrix.index.duplicated().sum()}")
        close_matrix = close_matrix[~close_matrix.index.duplicated(keep="first")]
    
    if not close_matrix.index.is_monotonic_increasing:
        close_matrix = close_matrix.sort_index()

    # Quality filter: keep only stocks with sufficient data (80% of observations)
    threshold = 0.80
    min_valid_count = int(threshold * len(close_matrix))
    valid_count = close_matrix.notna().sum()
    keep_stocks = valid_count[valid_count >= min_valid_count].index.tolist()
    
    removed_stocks = len(close_matrix.columns) - len(keep_stocks)
    if removed_stocks > 0:
        logger.info(f"Data quality filter: removed {removed_stocks} stocks with <{threshold*100:.0f}% data")
    
    close_matrix = close_matrix[keep_stocks]

    # Ensure datetime index
    close_matrix.index = pd.to_datetime(close_matrix.index, errors="coerce")
    close_matrix = close_matrix[close_matrix.index.notna()]
    
    if close_matrix.empty:
        logger.error("Matrix empty after date filtering")
        return pd.DataFrame()

    # Filter to years > 2000 (exclude old/unreliable data)
    close_matrix = close_matrix[close_matrix.index.year > 2000]

    final_shape = close_matrix.shape
    logger.info(
        f"Close matrix final shape: {final_shape} | "
        f"Removed {initial_shape[1] - final_shape[1]} stocks | "
        f"Date range: {close_matrix.index[0].date()} to {close_matrix.index[-1].date()}"
    )

    # Validate no NaNs in critical columns
    nan_pct = close_matrix.isna().sum().sum() / close_matrix.size * 100
    logger.info(f"Missing data after processing: {nan_pct:.2f}%")

    # Save
    save_path = os.path.join(processed_path, "close_prices.csv")
    close_matrix.to_csv(save_path)
    logger.info(f"Close matrix saved: {close_matrix.shape} → {save_path}")
    
    return close_matrix


def compute_returns(close_matrix: pd.DataFrame, processed_path: str) -> pd.DataFrame:
    """
    Compute log returns from close prices.
    
    CRITICAL:
    - Uses log returns: ln(P_t / P_{t-1})
    - First observation is NaN (no prior price)
    - Validates no infinities or extreme values
    - Checks for NaN propagation
    
    Args:
        close_matrix: Close prices matrix
        processed_path: Directory to save returns
        
    Returns:
        Returns matrix (aligned with close_matrix index)
    """
    
    if close_matrix.empty:
        logger.error("Empty close_matrix provided to compute_returns")
        return pd.DataFrame()
    
    logger.info(f"Computing log returns from {close_matrix.shape} price matrix")
    
    # Log returns: ln(P_t / P_{t-1})
    returns = np.log(
        close_matrix / close_matrix.shift(1)
    )
    
    # Remove rows where all returns are NaN (first row)
    returns = returns.dropna(how="all")
    
    # Validate returns
    # Check for infinities (can occur if price = 0, which shouldn't happen but check anyway)
    inf_count = np.isinf(returns.values).sum()
    if inf_count > 0:
        logger.warning(f"Infinities detected in returns: {inf_count} values")
        returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Check for extreme values (>100% daily return is suspicious)
    extreme_returns = (returns.abs() > 1.0).sum().sum()  # 100% = ln(2) ≈ 0.693
    if extreme_returns > 0:
        logger.warning(f"Extreme returns (>100%): {extreme_returns} values")
    
    # Statistics
    nan_count = returns.isna().sum().sum()
    nan_pct = nan_count / returns.size * 100
    mean_return = returns.values[~np.isnan(returns.values)].mean()
    std_return = returns.values[~np.isnan(returns.values)].std()
    
    logger.info(
        f"Returns matrix: {returns.shape} | "
        f"NaN: {nan_pct:.2f}% | "
        f"Mean daily return: {mean_return*100:.4f}% | "
        f"Std dev: {std_return*100:.4f}%"
    )
    
    # Save
    save_path = os.path.join(processed_path, "daily_returns.csv")
    returns.to_csv(save_path)
    logger.info(f"Returns matrix saved: {returns.shape} → {save_path}")
    
    return returns
