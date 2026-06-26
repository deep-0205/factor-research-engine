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

    scores = composite_scores.where(universe_flags)

    signals = pd.DataFrame(0, index=scores.index, columns=scores.columns)

    for date in scores.index:
        row = scores.loc[date].dropna()
        if row.empty:
            continue

        long_threshold  = row.quantile(1 - top_decile)
        short_threshold = row.quantile(bottom_decile)

        signals.loc[date, row[row >= long_threshold].index]  =  1
        signals.loc[date, row[row <= short_threshold].index] = -1

    long_count  = (signals ==  1).sum(axis=1).mean()
    short_count = (signals == -1).sum(axis=1).mean()

    logger.info(
        f"Signals generated | Avg longs/day: {long_count:.1f} | "
        f"Avg shorts/day: {short_count:.1f}"
    )
    return signals

def apply_signal_smoothing(
        signals: pd.DataFrame,
        composite_scores: pd.DataFrame,
        hold_buffer: float = 0.05) -> pd.DataFrame:

    smoothed = signals.copy()

    for col in smoothed.columns:
        prev_signal = 0
        for date in smoothed.index:
            raw_sig = signals.loc[date, col]
            score   = composite_scores.loc[date, col]

            if pd.isna(score):
                smoothed.loc[date, col] = 0
                prev_signal = 0
                continue

            if prev_signal == 1 and raw_sig == 0:
                if score >= (composite_scores[col].quantile(
                        0.90 - hold_buffer)):
                    smoothed.loc[date, col] = 1  # Hold
                else:
                    smoothed.loc[date, col] = 0  # Exit

            elif prev_signal == -1 and raw_sig == 0:
                if score <= (composite_scores[col].quantile(
                        0.10 + hold_buffer)):
                    smoothed.loc[date, col] = -1  # Hold
                else:
                    smoothed.loc[date, col] = 0   # Exit

            prev_signal = smoothed.loc[date, col]

    logger.info(f"Signal smoothing applied: buffer={hold_buffer}")
    return smoothed

def compute_turnover(signals: pd.DataFrame) -> pd.Series:

    signal_changes = signals.diff().abs()
    prev_active    = (signals.shift(1) != 0).sum(axis=1)

    turnover = signal_changes.sum(axis=1) / prev_active.replace(0, np.nan)

    logger.info(
        f"Turnover | Mean: {turnover.mean():.4f} | "
        f"Max: {turnover.max():.4f}"
    )
    return turnover

def get_rebalance_dates(index: pd.DatetimeIndex, frequency: str = "monthly") -> list:

    if frequency == "monthly":
        rebalance_dates = (
            pd.Series(index, index=index)
            .groupby([index.year, index.month])
            .first()
            .tolist()
        )

    elif frequency == "weekly":
        rebalance_dates = [d for d in index if d.weekday() == 0]

    elif frequency == "daily":
        rebalance_dates = index.tolist()

    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    logger.info(
        f"Rebalance schedule: {frequency} | "
        f"{len(rebalance_dates)} rebalance dates"
    )
    return rebalance_dates


def apply_rebalance_schedule(signals: pd.DataFrame, frequency: str = "monthly") -> pd.DataFrame:

    rebalance_dates = get_rebalance_dates(signals.index, frequency)
    rebalance_set   = set(rebalance_dates)

    scheduled = pd.DataFrame(0, index=signals.index, columns=signals.columns)

    last_signal = pd.Series(0, index=signals.columns)

    for date in signals.index:
        if date in rebalance_set:
            last_signal = signals.loc[date]   
        scheduled.loc[date] = last_signal     

    logger.info(f"Rebalance schedule applied: {frequency}")
    return scheduled

def compute_signal_stats(signals: pd.DataFrame, returns_matrix: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:

    forward_returns = returns_matrix.shift(-forward_days).rolling(forward_days).sum()

    stats = []

    for date in signals.index:
        long_mask  = signals.loc[date] ==  1
        short_mask = signals.loc[date] == -1

        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue

        fwd = forward_returns.loc[date]

        long_ret  = fwd[long_mask].mean()
        short_ret = fwd[short_mask].mean()
        spread    = long_ret - short_ret

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

    stats_df = pd.DataFrame(stats).set_index("date")

    logger.info(
        f"Signal Stats | "
        f"Avg Spread: {stats_df['spread'].mean():.4f} | "
        f"Hit Rate: {stats_df['hit_rate'].mean():.4f} | "
        f"Avg Longs: {stats_df['n_long'].mean():.1f}"
    )
    return stats_df

def run_signal_engine(
        composite_scores: pd.DataFrame,
        universe_flags: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        save_path: str = "signals/") -> dict:

    config = load_config()
    os.makedirs(save_path, exist_ok=True)

    logger.info("Generating raw signals...")
    raw_signals = generate_long_short_signals(
        composite_scores=composite_scores,
        universe_flags=universe_flags,
        top_decile=config["portfolio"]["top_decile"],
        bottom_decile=config["portfolio"]["bottom_decile"]
    )

    logger.info("Smoothing signals...")
    smoothed_signals = apply_signal_smoothing(
        signals=raw_signals,
        composite_scores=composite_scores
    )

    logger.info("Applying rebalance schedule...")
    final_signals = apply_rebalance_schedule(
        signals=smoothed_signals,
        frequency=config["portfolio"]["rebalance_freq"]
    )

    turnover = compute_turnover(final_signals)

    logger.info("Computing signal diagnostics...")
    signal_stats = compute_signal_stats(final_signals, returns_matrix)

    raw_signals.to_csv(os.path.join(save_path, "raw_signals.csv"))
    final_signals.to_csv(os.path.join(save_path, "final_signals.csv"))
    turnover.to_csv(os.path.join(save_path, "turnover.csv"))
    signal_stats.to_csv(os.path.join(save_path, "signal_stats.csv"))

    logger.info("Signal engine complete. All outputs saved to signals/")

    return {
        "raw_signals"   : raw_signals,
        "final_signals" : final_signals,
        "turnover"      : turnover,
        "signal_stats"  : signal_stats
    }

