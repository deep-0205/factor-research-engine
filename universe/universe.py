import pandas as pd
import numpy as np
import os
import yaml
from logger import get_logger

logger = get_logger("universe")

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def compute_average_volume(raw_path: str, tickers: list, lookback: int = 60) -> pd.Series:
    """
    Compute average trading volume for liquidity filtering.
    
    CRITICAL: Uses only the LAST lookback days, not including any
    future data. This represents the most recent trading pattern
    before the universe selection date.
    
    Args:
        raw_path: Directory containing raw OHLCV CSV files
        tickers: List of ticker symbols to process
        lookback: Number of recent days for averaging
        
    Returns:
        Series with average volume per ticker
    """
    avg_volumes = {}

    for ticker in tickers:
        path = os.path.join(raw_path, f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, header=0)

            # Handle header rows that contain ticker info
            if df.iloc[0].astype(str).str.contains(
                ticker.split(".")[0], case=False, na=False
            ).any():
                df = pd.read_csv(path, header=0, skiprows=[1])

            df.columns = [str(c).strip().lower() for c in df.columns]

            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
            df = df[~df.index.isna()]

            if "volume" not in df.columns:
                continue

            vol = pd.to_numeric(df["volume"], errors="coerce")
            # ✓ CRITICAL: Use ONLY the last lookback days for average
            avg_vol = vol.iloc[-lookback:].mean()

            if pd.notna(avg_vol):
                avg_volumes[ticker] = avg_vol

        except Exception as e:
            logger.warning(f"Volume read failed for {ticker}: {e}")

    logger.debug(f"Computed average volume for {len(avg_volumes)}/{len(tickers)} tickers")
    return pd.Series(avg_volumes)

def apply_liquidity_filter(avg_volumes: pd.Series, min_avg_volume: int) -> list:
    """
    Filter stocks based on average trading volume.
    
    CRITICAL: Uses historical average volume (across entire history),
    preventing survivorship bias from current-only volume checks.
    
    Args:
        avg_volumes: Series of average volumes (from compute_average_volume)
        min_avg_volume: Minimum acceptable average daily volume
        
    Returns:
        List of ticker symbols with sufficient liquidity
    """
    if avg_volumes.empty:
        logger.warning("apply_liquidity_filter: avg_volumes is empty")
        return []
    
    liquid = avg_volumes[avg_volumes >= min_avg_volume].index.tolist()
    logger.info(
        f"Liquidity filter: {len(avg_volumes)} → {len(liquid)} stocks "
        f"(min avg volume: {min_avg_volume:,} shares/day)"
    )
    return liquid

def apply_history_filter(close_matrix: pd.DataFrame, min_history_days: int = 252) -> list:
    """
    Filter stocks based on minimum historical data requirement.
    
    CRITICAL: Prevents data leakage by checking the TOTAL history available,
    not the current date. A stock must have at least min_history_days of
    non-NaN price data across its entire history.
    
    Args:
        close_matrix: Price DataFrame with datetime index
        min_history_days: Minimum trading days required
        
    Returns:
        List of ticker symbols meeting history requirement
    """
    if close_matrix.empty:
        logger.warning("apply_history_filter: close_matrix is empty")
        return []
    
    valid = close_matrix.columns[
        close_matrix.notna().sum() >= min_history_days
    ].tolist()
    logger.info(
        f"History filter: {len(close_matrix.columns)} → {len(valid)} stocks "
        f"(min {min_history_days} trading days)"
    )
    return valid

def apply_price_filter(close_matrix: pd.DataFrame, min_price: float = 10.0) -> list:
    """
    Filter stocks based on current (latest) price.
    
    CRITICAL: Uses ONLY the latest available price in close_matrix,
    preventing look-ahead bias. Stocks below min_price may be:
    - Delisted or suspended
    - Penny stocks with lower liquidity
    - Recent IPOs not yet meeting minimum price
    
    Args:
        close_matrix: Price DataFrame with datetime index
        min_price: Minimum current price (₹ for Indian stocks)
        
    Returns:
        List of ticker symbols currently trading above min_price
    """
    if close_matrix.empty:
        logger.warning("apply_price_filter: close_matrix is empty")
        return []
    
    close_matrix_copy = close_matrix.copy()
    close_matrix_copy.index = pd.to_datetime(
        close_matrix_copy.index, errors="coerce"
    )
    close_matrix_copy = close_matrix_copy[
        close_matrix_copy.index.notna() &
        (close_matrix_copy.index.year > 2000)
    ]

    if close_matrix_copy.empty:
        logger.warning("apply_price_filter: No valid dates after filtering")
        return []

    # ✓ CRITICAL: Use ONLY the latest (most recent) date
    latest_prices = close_matrix_copy.iloc[-1]
    latest_prices = pd.to_numeric(latest_prices, errors="coerce")

    valid = latest_prices[latest_prices >= min_price].index.tolist()
    logger.info(
        f"Price filter: {len(close_matrix_copy.columns)} → "
        f"{len(valid)} stocks (current min price: ₹{min_price})"
    )
    return valid

def build_historical_universe(
    close_matrix: pd.DataFrame,
    lookback_days: int = 252,
    min_price: float = 10.0,
    min_data_pct: float = 0.80,) -> pd.DataFrame:
    """
    Build TIME-VARYING universe flags for each date.
    
    CRITICAL: This prevents SURVIVORSHIP BIAS by determining which stocks
    were in the investable universe at EACH historical date, using only
    data that existed at that date (no future data).
    
    Algorithm:
    - For each date i, use only data up to (but not including) date i
    - Check if each stock has sufficient history: notna().sum() >= lookback_days
    - Check if current price meets min_price threshold
    - Returns: Boolean DataFrame indicating universe membership per date
    
    Args:
        close_matrix: Price matrix with datetime index
        lookback_days: Minimum trading days required in history
        min_price: Minimum price filter (₹ for Indian stocks)
        min_data_pct: Minimum % of non-NaN values in lookback window
        
    Returns:
        Boolean DataFrame: universe_flags[date, ticker] = True if ticker
                          was in investable universe at that date
    """
    if close_matrix.empty:
        logger.warning("build_historical_universe: close_matrix is empty")
        return pd.DataFrame()
    
    # Validate index is sorted
    if not close_matrix.index.is_monotonic_increasing:
        logger.warning("build_historical_universe: Sorting close_matrix index")
        close_matrix = close_matrix.sort_index()
    
    universe_flags = pd.DataFrame(
        False, 
        index=close_matrix.index, 
        columns=close_matrix.columns
    )
    
    # For each date, determine which stocks meet selection criteria
    # using ONLY data available UP TO (not including) that date
    for i, date in enumerate(close_matrix.index):
        # ✓ CRITICAL: Only use past data (data up to but not including current date)
        if i < lookback_days:
            logger.debug(
                f"  {date.date()}: Insufficient history (need {lookback_days} days, have {i})"
            )
            continue
        
        # ✓ Use iloc[:i] to get ONLY past data (up to but not including current date)
        past_data = close_matrix.iloc[:i]
        
        # Check 1: Minimum historical data requirement
        # Count non-NaN values for each stock in the full past history
        data_availability = past_data.notna().sum()
        history_ok = data_availability >= lookback_days
        
        # Check 2: Current price filter
        # Use current date's prices (already in past_data)
        current_prices = close_matrix.iloc[i]
        current_prices = pd.to_numeric(current_prices, errors='coerce')
        price_ok = current_prices >= min_price
        
        # Check 3: Data quality in lookback window
        # For lookback window, ensure minimum data percentage
        if i >= lookback_days:
            lookback_window = close_matrix.iloc[i - lookback_days:i]
            data_pct = lookback_window.notna().sum() / lookback_days
            data_quality_ok = data_pct >= min_data_pct
        else:
            data_quality_ok = pd.Series(False, index=close_matrix.columns)
        
        # Universe membership: all three criteria must be met
        universe_flags.iloc[i] = history_ok & price_ok & data_quality_ok
    
    avg_universe_size = universe_flags.sum(axis=1).mean()
    min_universe_size = universe_flags.sum(axis=1).min()
    max_universe_size = universe_flags.sum(axis=1).max()
    
    logger.info(
        f"Historical universe built (TIME-VARYING): "
        f"avg {avg_universe_size:.0f} stocks/day "
        f"(min {min_universe_size:.0f}, max {max_universe_size:.0f})"
    )
    
    return universe_flags

def build_universe(
    close_matrix: pd.DataFrame, 
    raw_path: str, 
    save_path: str = "universe/") -> dict:
    """
    Build universe with TIME-VARYING membership flags.
    
    CRITICAL: Modern portfolio construction requires different universe
    definitions at different times (stocks delisted, added, or fell below
    liquidity thresholds). This function:
    
    1. Applies STATIC filters to identify eligible stock universe
       (history, price, volume criteria)
    2. Creates TIME-VARYING universe_flags for each date
       (shows which eligible stocks were actually tradeable then)
    3. Saves both for reference and reproducibility
    
    Args:
        close_matrix: Price DataFrame with datetime index and stock columns
        raw_path: Path to raw OHLCV data files
        save_path: Directory to save universe outputs
        
    Returns:
        dict with:
            - "current_universe": List of stocks meeting static criteria
            - "universe_flags": DataFrame showing time-varying membership
    """
    config = load_config()
    os.makedirs(save_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info("UNIVERSE CONSTRUCTION: Time-Varying Selection")
    logger.info("=" * 60)

    all_tickers = close_matrix.columns.tolist()
    logger.info(f"Initial universe: {len(all_tickers)} stocks")

    # Apply static filters to identify eligible stocks
    logger.info("\n[1/4] Applying static criteria...")
    history_ok = apply_history_filter(close_matrix)
    price_ok = apply_price_filter(close_matrix)
    avg_volumes = compute_average_volume(raw_path, all_tickers)
    liquidity_ok = apply_liquidity_filter(
        avg_volumes, min_avg_volume=config["universe"]["min_avg_volume"]
    )

    # Union of all static filters (stocks that COULD ever be in universe)
    current_universe = list(set(history_ok) & set(price_ok) & set(liquidity_ok))
    logger.info(
        f"Static criteria: {len(current_universe)} stocks meet history, "
        f"price, and liquidity requirements"
    )

    # Build time-varying universe flags
    # Shows which eligible stocks were actually tradeable at each date
    logger.info("\n[2/4] Building time-varying universe membership...")
    universe_flags = build_historical_universe(close_matrix)
    
    # ✓ CRITICAL: universe_flags now shows the actual historical universe
    # at each point in time, preventing survivorship bias
    
    logger.info("\n[3/4] Universe statistics:")
    if not universe_flags.empty:
        daily_counts = universe_flags.sum(axis=1)
        logger.info(f"  Min universe size: {daily_counts.min():.0f} stocks")
        logger.info(f"  Max universe size: {daily_counts.max():.0f} stocks")
        logger.info(f"  Avg universe size: {daily_counts.mean():.0f} stocks")
        logger.info(f"  Std universe size: {daily_counts.std():.0f} stocks")

    # Save outputs
    logger.info("\n[4/4] Saving universe data...")
    current_universe_series = pd.Series(
        current_universe, 
        name="ticker"
    )
    current_universe_series.to_csv(
        os.path.join(save_path, "current_universe.csv"), 
        index=False
    )
    logger.info(
        f"  Saved {len(current_universe)} eligible stocks to "
        f"current_universe.csv"
    )
    
    universe_flags.to_csv(os.path.join(save_path, "universe_flags.csv"))
    logger.info(
        f"  Saved time-varying universe flags ({universe_flags.shape[0]} dates) "
        f"to universe_flags.csv"
    )

    logger.info("\n" + "=" * 60)
    logger.info("Final universe: {} eligible stocks, {} dates with flags".format(
        len(current_universe), 
        len(universe_flags)
    ))
    logger.info("=" * 60 + "\n")

    return {
        "current_universe": current_universe, 
        "universe_flags": universe_flags
    }
