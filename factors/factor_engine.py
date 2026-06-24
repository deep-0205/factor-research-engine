import pandas as pd
import numpy as np
import os
from logger import get_logger

logger = get_logger("factor_engine")

def compute_momentum(close_matrix: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    # Price `lookback` days ago
    past_price = close_matrix.shift(lookback)

    # Price `skip` days ago (exclude most recent month)
    recent_price = close_matrix.shift(skip)

    # Momentum = cumulative return over the window
    momentum = (recent_price - past_price) / past_price

    logger.info(f"Momentum factor computed: window={lookback}d, skip={skip}d")
    return momentum

def compute_reversal(close_matrix: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    short_return = close_matrix.pct_change(periods=window)

    # Negate: large negative return → high positive reversal score
    reversal = -short_return

    logger.info(f"Reversal factor computed: window={window}d")
    return reversal

def compute_volatility(returns_matrix: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    rolling_vol = returns_matrix.rolling(window=window).std() * np.sqrt(252)

    # Negate: low vol → high score → preferred
    volatility_factor = -rolling_vol

    logger.info(f"Volatility factor computed: window={window}d, annualized")
    return volatility_factor

def compute_quality_proxy(returns_matrix: pd.DataFrame, close_matrix: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    # --- Component 1: Return Consistency (Sharpe-like) ---
    rolling_mean = returns_matrix.rolling(window).mean()
    rolling_std  = returns_matrix.rolling(window).std()
    consistency  = rolling_mean / (rolling_std + 1e-8)

    # --- Component 2: Trend Stability (R-squared) ---
    def rolling_r2(series, w):
        """Computes rolling R² of price vs time index."""
        r2_values = [np.nan] * len(series)
        x = np.arange(w)

        for i in range(w, len(series)):
            y = series.iloc[i - w: i].values
            if np.isnan(y).any():
                continue
            # Linear regression manually
            x_mean = x.mean()
            y_mean = y.mean()
            ss_tot = ((y - y_mean) ** 2).sum()
            ss_res = ((y - (np.polyval(np.polyfit(x, y, 1), x))) ** 2).sum()
            r2_values[i] = 1 - ss_res / (ss_tot + 1e-8)

        return pd.Series(r2_values, index=series.index)

    trend_stability = close_matrix.apply(
        lambda col: rolling_r2(col, window), axis=0
    )

    # --- Combine ---
    quality = 0.5 * consistency + 0.5 * trend_stability

    logger.info(f"Quality proxy factor computed: window={window}d")
    return quality

def cross_sectional_zscore(factor: pd.DataFrame, universe_flags: pd.DataFrame = None) -> pd.DataFrame:
    if universe_flags is not None:
        # Only normalize stocks that are in the universe on each date
        factor = factor.where(universe_flags)

    # Row-wise (cross-sectional) mean and std
    row_mean = factor.mean(axis=1)
    row_std  = factor.std(axis=1)

    # Z-score
    zscore = factor.sub(row_mean, axis=0).div(row_std + 1e-8, axis=0)

    # Winsorize: clip extreme values to ±3
    zscore = zscore.clip(lower=-3, upper=3)

    return zscore

def cross_sectional_rank(factor: pd.DataFrame, universe_flags: pd.DataFrame = None) -> pd.DataFrame:
    if universe_flags is not None:
        factor = factor.where(universe_flags)

    # pct=True gives percentile rank between 0 and 1
    ranks = factor.rank(axis=1, pct=True, na_option="keep")

    return ranks

def compute_composite_factor(momentum_rank: pd.DataFrame, reversal_rank: pd.DataFrame, volatility_rank: pd.DataFrame, quality_rank: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    if weights is None:
        weights = {
            "momentum"  : 0.40,
            "reversal"  : 0.20,
            "volatility": 0.20,
            "quality"   : 0.20
        }

    composite = (
        weights["momentum"]   * momentum_rank   +
        weights["reversal"]   * reversal_rank   +
        weights["volatility"] * volatility_rank +
        weights["quality"]    * quality_rank
    )

    logger.info(
        f"Composite factor computed with weights: {weights}"
    )
    return composite

def compute_ic(factor_scores: pd.DataFrame, returns_matrix: pd.DataFrame, forward_days: int = 21) -> pd.Series:
    from scipy.stats import spearmanr

    forward_returns = returns_matrix.shift(-forward_days).rolling(
        forward_days
    ).sum()

    ic_series = {}

    for date in factor_scores.index:
        scores  = factor_scores.loc[date].dropna()
        fwd_ret = forward_returns.loc[date].dropna()

        common = scores.index.intersection(fwd_ret.index)
        if len(common) < 20:
            continue

        ic, _ = spearmanr(scores[common], fwd_ret[common])
        ic_series[date] = ic

    ic = pd.Series(ic_series)
    logger.info(
        f"IC Summary → Mean: {ic.mean():.4f}, "
        f"Std: {ic.std():.4f}, "
        f"IR (IC/Std): {ic.mean()/ic.std():.4f}"
    )
    return ic

def run_factor_engine(close_matrix: pd.DataFrame, returns_matrix: pd.DataFrame, universe_flags: pd.DataFrame, save_path: str = "factors/") -> dict:
    os.makedirs(save_path, exist_ok=True)

    # --- Step 1: Raw Factors ---
    logger.info("Computing raw factors...")
    momentum_raw   = compute_momentum(close_matrix)
    reversal_raw   = compute_reversal(close_matrix)
    volatility_raw = compute_volatility(returns_matrix)
    quality_raw    = compute_quality_proxy(returns_matrix, close_matrix)

    # --- Step 2: Z-Score Normalization ---
    logger.info("Normalizing factors...")
    momentum_z   = cross_sectional_zscore(momentum_raw,   universe_flags)
    reversal_z   = cross_sectional_zscore(reversal_raw,   universe_flags)
    volatility_z = cross_sectional_zscore(volatility_raw, universe_flags)
    quality_z    = cross_sectional_zscore(quality_raw,    universe_flags)

    # --- Step 3: Percentile Ranking ---
    logger.info("Ranking factors...")
    momentum_rank   = cross_sectional_rank(momentum_z,   universe_flags)
    reversal_rank   = cross_sectional_rank(reversal_z,   universe_flags)
    volatility_rank = cross_sectional_rank(volatility_z, universe_flags)
    quality_rank    = cross_sectional_rank(quality_z,    universe_flags)

    # --- Step 4: Composite Score ---
    composite = compute_composite_factor(
        momentum_rank, reversal_rank, volatility_rank, quality_rank
    )

    # --- Step 5: IC Diagnostics ---
    logger.info("Computing IC diagnostics...")
    ic_series = compute_ic(composite, returns_matrix)

    # --- Save All Outputs ---
    composite.to_csv(os.path.join(save_path, "composite_scores.csv"))
    momentum_rank.to_csv(os.path.join(save_path, "momentum_rank.csv"))
    reversal_rank.to_csv(os.path.join(save_path, "reversal_rank.csv"))
    volatility_rank.to_csv(os.path.join(save_path, "volatility_rank.csv"))
    quality_rank.to_csv(os.path.join(save_path, "quality_rank.csv"))
    ic_series.to_csv(os.path.join(save_path, "ic_series.csv"))

    logger.info("Factor engine complete. All outputs saved to factors/")

    return {
        "composite"      : composite,
        "momentum_rank"  : momentum_rank,
        "reversal_rank"  : reversal_rank,
        "volatility_rank": volatility_rank,
        "quality_rank"   : quality_rank,
        "ic_series"      : ic_series
    }

