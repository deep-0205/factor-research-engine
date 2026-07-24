import pandas as pd
import numpy as np
import os
import yaml
from logger import get_logger

logger = get_logger("signal_engine")

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)
    
def generate_long_short_signals(
        composite_scores: pd.DataFrame,
        universe_flags: pd.DataFrame,
        top_decile: float = 0.10,
        bottom_decile: float = 0.10) -> pd.DataFrame:
    """
    Generate binary long/short signals from composite factor scores.
    
    CRITICAL: NO LOOK-AHEAD BIAS
    - Uses only scores available on each date
    - Top decile (best scores) → long signal (+1)
    - Bottom decile (worst scores) → short signal (-1)
    - Middle 80% → no signal (0)
    
    Args:
        composite_scores: Factor scores from 0 to 1
        universe_flags: Boolean DF indicating valid universe
        top_decile: Fraction of universe to go long (0.10 = top 10%)
        bottom_decile: Fraction to go short (0.10 = bottom 10%)
        
    Returns:
        Signal matrix with values in {-1, 0, 1}
    """
    
    if composite_scores.empty or universe_flags.empty:
        logger.error("Empty input to generate_long_short_signals")
        return pd.DataFrame()

    # Apply universe filter: only score valid universe stocks
    scores = composite_scores.where(universe_flags)

    signals = pd.DataFrame(0, index=scores.index, columns=scores.columns)

    for date in scores.index:
        row = scores.loc[date].dropna()
        
        if row.empty:
            continue

        # Compute quantiles using ONLY data on this date
        long_threshold  = row.quantile(1 - top_decile)
        short_threshold = row.quantile(bottom_decile)

        # Generate signals
        signals.loc[date, row[row >= long_threshold].index]  =  1
        signals.loc[date, row[row <= short_threshold].index] = -1

    # Statistics
    long_count  = (signals ==  1).sum(axis=1).mean()
    short_count = (signals == -1).sum(axis=1).mean()
    
    logger.info(
        f"Raw Signals Generated | "
        f"Avg Longs/day: {long_count:.1f} | "
        f"Avg Shorts/day: {short_count:.1f} | "
        f"Total dates: {len(signals)}"
    )
    return signals

def apply_signal_smoothing(
        signals: pd.DataFrame,
        composite_scores: pd.DataFrame,
        hold_buffer: float = 0.05) -> pd.DataFrame:
    """
    Smooth signal transitions using hold logic and dynamic thresholds.
    
    CRITICAL: Removed look-ahead bias!
    
    OLD BUG: Used composite_scores[col].quantile() which computed quantile
    over entire HISTORICAL SERIES — future data was included!
    
    NEW FIX: Uses ONLY scores up to current date (expanding quantile).
    
    Logic:
    - If in long position and new signal is 0:
      - Hold if score still above 85th percentile (with buffer)
      - Exit if score drops below threshold
    - If in short position and new signal is 0:
      - Hold if score still below 15th percentile (with buffer)
      - Exit if score rises above threshold
    
    Args:
        signals: Binary signals from generate_long_short_signals
        composite_scores: Factor scores
        hold_buffer: Buffer for hold thresholds (0.05 = 5%)
        
    Returns:
        Smoothed signals with hold logic applied
    """
    
    if signals.empty or composite_scores.empty:
        logger.error("Empty input to apply_signal_smoothing")
        return pd.DataFrame()

    smoothed = signals.copy()

    for col in smoothed.columns:
        prev_signal = 0
        
        for date_idx, date in enumerate(smoothed.index):
            raw_sig = signals.loc[date, col]
            score = composite_scores.loc[date, col]

            if pd.isna(score):
                smoothed.loc[date, col] = 0
                prev_signal = 0
                continue

            # FIX: Use expanding quantile (only data up to today)
            # not historical quantile (which includes future data)
            historical_scores = composite_scores[col].iloc[:date_idx + 1].dropna()
            
            if len(historical_scores) < 20:
                # Not enough history, use current signal as-is
                smoothed.loc[date, col] = raw_sig
                prev_signal = raw_sig
                continue
            
            long_hold_threshold = historical_scores.quantile(0.90 - hold_buffer)
            short_hold_threshold = historical_scores.quantile(0.10 + hold_buffer)

            # Hold logic
            if prev_signal == 1 and raw_sig == 0:
                if score >= long_hold_threshold:
                    smoothed.loc[date, col] = 1  # Hold
                else:
                    smoothed.loc[date, col] = 0  # Exit

            elif prev_signal == -1 and raw_sig == 0:
                if score <= short_hold_threshold:
                    smoothed.loc[date, col] = -1  # Hold
                else:
                    smoothed.loc[date, col] = 0   # Exit
            else:
                # New signal or continuing 0
                smoothed.loc[date, col] = raw_sig

            prev_signal = smoothed.loc[date, col]

    # Statistics
    n_held = ((signals != 0) & (smoothed == signals)).sum().sum()
    n_total = (signals != 0).sum().sum()
    
    logger.info(
        f"Signal Smoothing Applied | "
        f"Hold Buffer: {hold_buffer:.2f} | "
        f"Positions held: {n_held}/{n_total} "
        f"({100*n_held/(n_total+1):.1f}%)"
    )
    return smoothed

def compute_turnover(signals: pd.DataFrame) -> pd.Series:
    """
    Compute portfolio turnover (fraction of positions that change each day).
    
    Turnover = (# positions changed) / (# active positions previous day)
    
    Args:
        signals: Binary signal matrix
        
    Returns:
        Daily turnover series
    """
    
    if signals.empty:
        logger.warning("Empty signals provided to compute_turnover")
        return pd.Series()

    # Absolute value of signal changes
    signal_changes = signals.diff().abs()
    
    # Number of active (non-zero) positions at previous day
    prev_active = (signals.shift(1) != 0).sum(axis=1)

    # Turnover: fraction of active positions that changed
    turnover = signal_changes.sum(axis=1) / prev_active.replace(0, np.nan)

    # Statistics
    valid_turnover = turnover.dropna()
    
    if not valid_turnover.empty:
        logger.info(
            f"Turnover Computed | "
            f"Mean: {valid_turnover.mean():.4f} | "
            f"Max: {valid_turnover.max():.4f} | "
            f"Median: {valid_turnover.median():.4f}"
        )
    
    return turnover

def get_rebalance_dates(index: pd.DatetimeIndex, frequency: str = "monthly") -> list:
    """
    Get dates when rebalancing occurs.
    
    Args:
        index: DatetimeIndex of all available dates
        frequency: "monthly", "weekly", or "daily"
        
    Returns:
        List of rebalance dates
    """
    
    if len(index) == 0:
        logger.warning("Empty index provided to get_rebalance_dates")
        return []
    
    index = pd.DatetimeIndex(index)

    if frequency == "monthly":
        dates = pd.Series(index, index=index)
        rebalance_dates = (
            dates.groupby([index.year, index.month])
            .first()
            .tolist()
        )

    elif frequency == "weekly":
        rebalance_dates = [d for d in index if d.weekday() == 0]

    elif frequency == "daily":
        rebalance_dates = index.tolist()

    else:
        logger.error(f"Unknown rebalance frequency: {frequency}")
        raise ValueError(f"Unknown frequency: {frequency}")

    logger.info(
        f"Rebalance Schedule | Frequency: {frequency} | "
        f"Rebalance dates: {len(rebalance_dates)}"
    )
    return rebalance_dates


def apply_rebalance_schedule(signals: pd.DataFrame, frequency: str = "monthly") -> pd.DataFrame:
    """
    Hold signals constant between rebalance dates.
    
    CRITICAL: FIX for correctness
    - On rebalance dates: update to new signal
    - Between rebalance dates: maintain previous signal
    - This prevents excessive turnover
    
    Args:
        signals: Daily signals from smoothing
        frequency: Rebalance frequency
        
    Returns:
        Signals held constant between rebalance dates
    """
    
    if signals.empty:
        logger.error("Empty signals provided to apply_rebalance_schedule")
        return pd.DataFrame()

    rebalance_dates = get_rebalance_dates(signals.index, frequency)
    
    if not rebalance_dates:
        logger.warning("No rebalance dates found")
        return signals

    rebalance_set = set(rebalance_dates)

    scheduled = pd.DataFrame(0, index=signals.index, columns=signals.columns)

    # Start with no position
    last_signal = pd.Series(0, index=signals.columns)

    for date in signals.index:
        if date in rebalance_set:
            # Rebalance: update to new signal
            last_signal = signals.loc[date].copy()
        
        # Hold the last signal
        scheduled.loc[date] = last_signal

    # Statistics
    n_rebalances = len(rebalance_dates)
    n_held_days = len(signals) - len([d for d in signals.index if d in rebalance_set])
    
    logger.info(
        f"Rebalance Schedule Applied | "
        f"Frequency: {frequency} | "
        f"Rebalance dates: {n_rebalances} | "
        f"Held days: {n_held_days}"
    )
    return scheduled

def compute_signal_stats(signals: pd.DataFrame, returns_matrix: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:
    """
    Compute signal performance statistics.
    
    For each rebalance date:
    - Compute forward returns (next forward_days days)
    - Compare long positions vs short positions
    - Measure hit rate and spread
    
    Args:
        signals: Binary signal matrix
        returns_matrix: Daily returns
        forward_days: Horizon for forward returns
        
    Returns:
        Statistics DataFrame
    """
    
    if signals.empty or returns_matrix.empty:
        logger.warning("Empty input to compute_signal_stats")
        return pd.DataFrame()

    # Forward returns: sum of returns over next forward_days days
    forward_returns = returns_matrix.shift(-forward_days).rolling(forward_days).sum()

    stats = []

    for date in signals.index:
        # Current signals
        long_mask  = signals.loc[date] ==  1
        short_mask = signals.loc[date] == -1

        # Need both longs and shorts for meaningful comparison
        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue

        # Forward returns on this date
        if date not in forward_returns.index:
            continue
        
        fwd = forward_returns.loc[date]

        # Average returns
        long_ret  = fwd[long_mask].mean()
        short_ret = fwd[short_mask].mean()
        spread    = long_ret - short_ret

        # Hit rate (% of long positions with positive returns)
        long_hits = (fwd[long_mask] > 0).mean()

        stats.append({
            "date"        : date,
            "long_return" : long_ret,
            "short_return": short_ret,
            "spread"      : spread,
            "hit_rate"    : long_hits,
            "n_long"      : long_mask.sum(),
            "n_short"     : short_mask.sum()
        })

    if not stats:
        logger.warning("No signal statistics computed")
        return pd.DataFrame()

    stats_df = pd.DataFrame(stats).set_index("date")

    logger.info(
        f"Signal Stats Computed | "
        f"Avg Spread: {stats_df['spread'].mean():.4f} | "
        f"Hit Rate: {stats_df['hit_rate'].mean():.4f} | "
        f"Avg Longs: {stats_df['n_long'].mean():.1f} | "
        f"Avg Shorts: {stats_df['n_short'].mean():.1f}"
    )
    return stats_df

def run_signal_engine(
        composite_scores: pd.DataFrame,
        universe_flags: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        save_path: str = "signals/") -> dict:
    """
    Complete signal generation pipeline.
    
    Pipeline:
    1. Generate binary long/short signals from composite scores
    2. Apply signal smoothing with hold logic (uses expanding quantiles)
    3. Apply rebalance schedule (hold signals between rebalance dates)
    4. Compute diagnostics (turnover, signal performance)
    
    CRITICAL: All steps use only past/current data. No look-ahead bias.
    
    Args:
        composite_scores: Factor scores
        universe_flags: Valid universe indicator
        returns_matrix: Daily returns
        save_path: Output directory
        
    Returns:
        Dictionary with raw signals, final signals, turnover, stats
    """

    if composite_scores.empty or universe_flags.empty:
        logger.error("Empty input to run_signal_engine")
        return {}

    config = load_config()
    os.makedirs(save_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1: Generating raw signals...")
    logger.info("=" * 60)
    
    raw_signals = generate_long_short_signals(
        composite_scores=composite_scores,
        universe_flags=universe_flags,
        top_decile=config["portfolio"]["top_decile"],
        bottom_decile=config["portfolio"]["bottom_decile"]
    )
    
    if raw_signals.empty:
        logger.error("Raw signals generation failed")
        return {}

    logger.info("=" * 60)
    logger.info("STEP 2: Smoothing signals with hold logic...")
    logger.info("=" * 60)
    
    smoothed_signals = apply_signal_smoothing(
        signals=raw_signals,
        composite_scores=composite_scores
    )

    logger.info("=" * 60)
    logger.info("STEP 3: Applying rebalance schedule...")
    logger.info("=" * 60)
    
    final_signals = apply_rebalance_schedule(
        signals=smoothed_signals,
        frequency=config["portfolio"]["rebalance_freq"]
    )

    logger.info("=" * 60)
    logger.info("STEP 4: Computing diagnostics...")
    logger.info("=" * 60)
    
    turnover = compute_turnover(final_signals)

    signal_stats = compute_signal_stats(final_signals, returns_matrix)

    # Save outputs
    logger.info("Saving signal outputs...")
    raw_signals.to_csv(os.path.join(save_path, "raw_signals.csv"))
    final_signals.to_csv(os.path.join(save_path, "final_signals.csv"))
    turnover.to_csv(os.path.join(save_path, "turnover.csv"))
    signal_stats.to_csv(os.path.join(save_path, "signal_stats.csv"))

    logger.info(f"Signal engine complete. Outputs saved to {save_path}")

    return {
        "raw_signals"   : raw_signals,
        "final_signals" : final_signals,
        "turnover"      : turnover,
        "signal_stats"  : signal_stats
    }

