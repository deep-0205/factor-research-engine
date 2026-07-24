import pandas as pd
import numpy as np
import os
from logger import get_logger

logger = get_logger("factor_engine")

def compute_momentum(close_matrix: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    Compute momentum factor with NO LOOK-AHEAD BIAS.
    
    CRITICAL:
    - Looks at return from (lookback + skip) days ago to skip days ago
    - Excludes most recent 'skip' days (typically 21) to avoid recent noise
    - All data is past data at each observation point
    
    Timing:
      Date t:
      - past_price = close[t - lookback - skip]  (oldest)
      - recent_price = close[t - skip]            (most recent available)
      - momentum = (recent_price - past_price) / past_price
      - NOT using close[t] or close[t-1] (future data at time of factor formation)
    
    Args:
        close_matrix: Historical close prices
        lookback: Number of days to look back (252 = 1 year)
        skip: Number of days to skip (21 = 1 month, avoids recent returns)
        
    Returns:
        Momentum factor scores
    """
    if close_matrix.empty:
        logger.warning("Empty close_matrix provided to compute_momentum")
        return pd.DataFrame()
    
    # CORRECTED: Use lookback + skip to get the baseline (oldest price)
    past_price = close_matrix.shift(lookback + skip)

    # Most recent price (skip days ago, not today)
    recent_price = close_matrix.shift(skip)

    # Momentum = cumulative return over the window
    # This is the return from (lookback + skip) days ago to skip days ago
    momentum = (recent_price - past_price) / past_price

    n_valid = momentum.notna().sum().sum()
    logger.info(
        f"Momentum factor computed: lookback={lookback}d, skip={skip}d, "
        f"effective_window={lookback}d | Valid values: {n_valid}"
    )
    return momentum

def compute_reversal(close_matrix: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute reversal factor (short-term mean reversion).
    
    CRITICAL: Uses only past data
    - Computes return over past 'window' days
    - Negates it (negative recent return → positive reversal signal)
    - No future information leaks
    
    Args:
        close_matrix: Historical close prices
        window: Number of days for reversal window (5 = 1 week)
        
    Returns:
        Reversal factor scores (negated short-term returns)
    """
    if close_matrix.empty:
        logger.warning("Empty close_matrix provided to compute_reversal")
        return pd.DataFrame()
    
    # Recent short-term return (past 'window' days)
    # Specify fill_method=None to avoid FutureWarning
    short_return = close_matrix.pct_change(periods=window, fill_method=None)

    # Negate: large negative return → high positive reversal score
    # This captures mean-reversion tendency
    reversal = -short_return

    n_valid = reversal.notna().sum().sum()
    logger.info(f"Reversal factor computed: window={window}d | Valid values: {n_valid}")
    return reversal

def compute_volatility(returns_matrix: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute volatility factor (low volatility preference).
    
    CRITICAL: Uses only past data
    - Rolling volatility over past 'window' days
    - Annualized to 252 trading days
    - Negated: low vol → high score (preferred)
    - No future information
    
    Args:
        returns_matrix: Daily returns
        window: Rolling window for vol (20 = ~1 month)
        
    Returns:
        Volatility factor (negated annualized vol)
    """
    if returns_matrix.empty:
        logger.warning("Empty returns_matrix provided to compute_volatility")
        return pd.DataFrame()
    
    rolling_vol = returns_matrix.rolling(window=window).std() * np.sqrt(252)

    # Negate: low vol → high score → preferred
    volatility_factor = -rolling_vol

    n_valid = volatility_factor.notna().sum().sum()
    logger.info(
        f"Volatility factor computed: window={window}d, annualized | "
        f"Valid values: {n_valid}"
    )
    return volatility_factor

def compute_quality_proxy(returns_matrix: pd.DataFrame, close_matrix: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute quality proxy factor combining consistency and trend stability.
    
    Components:
    1. Return Consistency: Rolling Sharpe-like measure (mean return / std return)
    2. Trend Stability: Rolling R² of price vs time (how smooth the trend)
    
    Both computed on past data only, no look-ahead bias.
    
    Args:
        returns_matrix: Daily returns
        close_matrix: Close prices (for trend R²)
        window: Rolling window (60 = ~3 months)
        
    Returns:
        Quality factor (50% consistency + 50% trend)
    """
    if returns_matrix.empty or close_matrix.empty:
        logger.warning("Empty input provided to compute_quality_proxy")
        return pd.DataFrame()
    
    # --- Component 1: Return Consistency (Sharpe-like) ---
    rolling_mean = returns_matrix.rolling(window).mean()
    rolling_std  = returns_matrix.rolling(window).std()
    consistency  = rolling_mean / (rolling_std + 1e-8)

    # --- Component 2: Trend Stability (R-squared) ---
    def rolling_r2(series, w):
        """
        Computes rolling R² of price vs time index.
        Measures how well a linear trend explains price movements.
        Uses only past data (lookback window).
        """
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

    # --- Combine components ---
    quality = 0.5 * consistency + 0.5 * trend_stability

    n_valid = quality.notna().sum().sum()
    logger.info(
        f"Quality proxy factor computed: window={window}d | Valid values: {n_valid}"
    )
    return quality

def cross_sectional_zscore(factor: pd.DataFrame, universe_flags: pd.DataFrame = None) -> pd.DataFrame:
    """
    Normalize factors to zero-mean, unit-variance cross-sectionally.
    
    CRITICAL:
    - Only normalizes within stocks in universe on each date
    - Prevents look-ahead bias through universe_flags filtering
    - Clips extreme values to ±3 to handle outliers
    
    Args:
        factor: Raw factor values
        universe_flags: Boolean DF indicating which stocks are valid each date
        
    Returns:
        Z-scored factor values
    """
    if factor.empty:
        logger.warning("Empty factor provided to cross_sectional_zscore")
        return pd.DataFrame()
    
    factor_copy = factor.copy()
    
    # Only include stocks in the defined universe for normalization
    if universe_flags is not None:
        if universe_flags.shape != factor_copy.shape:
            logger.warning(
                f"Universe flags shape {universe_flags.shape} != "
                f"factor shape {factor_copy.shape}"
            )
        else:
            factor_copy = factor_copy.where(universe_flags)

    # Row-wise (cross-sectional) mean and std
    row_mean = factor_copy.mean(axis=1)
    row_std  = factor_copy.std(axis=1)

    # Z-score normalization
    zscore = factor_copy.sub(row_mean, axis=0).div(row_std + 1e-8, axis=0)

    # Winsorize: clip extreme values to ±3
    zscore = zscore.clip(lower=-3, upper=3)

    n_clipped = ((zscore == 3) | (zscore == -3)).sum().sum()
    logger.info(
        f"Cross-sectional Z-score | "
        f"Clipped to ±3: {n_clipped} values | "
        f"Mean: {zscore.values.mean():.4f}, Std: {zscore.values.std():.4f}"
    )
    return zscore

def cross_sectional_rank(factor: pd.DataFrame, universe_flags: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert factor values to percentile ranks within each date.
    
    CRITICAL:
    - Only ranks stocks in the universe on each date
    - Percentile ranks from 0 to 1
    - NA values remain NA (not ranked)
    
    Args:
        factor: Factor values
        universe_flags: Boolean DF indicating valid universe
        
    Returns:
        Percentile ranks for each factor
    """
    if factor.empty:
        logger.warning("Empty factor provided to cross_sectional_rank")
        return pd.DataFrame()
    
    factor_copy = factor.copy()
    
    # Only rank stocks in the universe
    if universe_flags is not None:
        if universe_flags.shape != factor_copy.shape:
            logger.warning(
                f"Universe flags shape {universe_flags.shape} != "
                f"factor shape {factor_copy.shape}"
            )
        else:
            factor_copy = factor_copy.where(universe_flags)

    # pct=True gives percentile rank between 0 and 1
    # na_option="keep" preserves NaNs (out-of-universe stocks)
    ranks = factor_copy.rank(axis=1, pct=True, na_option="keep")

    n_ranked = ranks.notna().sum().sum()
    logger.info(
        f"Cross-sectional Rank | Total values: {ranks.size}, "
        f"Ranked: {n_ranked}, NA: {ranks.isna().sum().sum()}"
    )
    return ranks

def compute_composite_factor(momentum_rank: pd.DataFrame, reversal_rank: pd.DataFrame, volatility_rank: pd.DataFrame, quality_rank: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Combine individual factor ranks into a composite score.
    
    Each input is already a percentile rank (0 to 1), so they're comparable.
    Weights must sum to 1.0 and reflect relative importance.
    
    Args:
        momentum_rank: Percentile ranks for momentum
        reversal_rank: Percentile ranks for reversal
        volatility_rank: Percentile ranks for low volatility
        quality_rank: Percentile ranks for quality
        weights: Dictionary of factor weights (default 40/20/20/20)
        
    Returns:
        Composite factor scores (0 to 1 percentile)
    """
    if momentum_rank.empty or reversal_rank.empty or volatility_rank.empty or quality_rank.empty:
        logger.error("Empty factor input to compute_composite_factor")
        return pd.DataFrame()
    
    if weights is None:
        weights = {
            "momentum"  : 0.40,
            "reversal"  : 0.20,
            "volatility": 0.20,
            "quality"   : 0.20
        }
    
    # Validate weights
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        logger.warning(
            f"Factor weights don't sum to 1.0: {weight_sum}. Normalizing..."
        )
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Verify all inputs have same shape and index
    shapes = {
        "momentum": momentum_rank.shape,
        "reversal": reversal_rank.shape,
        "volatility": volatility_rank.shape,
        "quality": quality_rank.shape
    }
    if len(set(str(v) for v in shapes.values())) > 1:
        logger.warning(f"Input shapes differ: {shapes}")

    composite = (
        weights["momentum"]   * momentum_rank   +
        weights["reversal"]   * reversal_rank   +
        weights["volatility"] * volatility_rank +
        weights["quality"]    * quality_rank
    )

    # Validate composite is in [0, 1] range (should be after weighting percentile ranks)
    n_out_of_range = ((composite < 0) | (composite > 1)).sum().sum()
    if n_out_of_range > 0:
        logger.debug(
            f"Composite factor: {n_out_of_range} values outside [0,1] "
            f"(likely due to NaNs). This is acceptable."
        )

    logger.info(
        f"Composite factor computed | Weights: {weights} | "
        f"Mean: {composite.values.mean():.4f}, "
        f"Std: {composite.values.std():.4f}"
    )
    return composite

def compute_ic(factor_scores: pd.DataFrame, returns_matrix: pd.DataFrame, forward_days: int = 21) -> pd.Series:
    """
    Compute Information Coefficient (factor predictiveness).
    
    CRITICAL: NO LOOK-AHEAD BIAS
    - Factor scores at date t
    - Forward returns = cumulative returns from t+1 to t+forward_days
    - Correlation tells us if factor at t predicts returns in future
    - Uses only data available at time of scoring
    
    Args:
        factor_scores: Factor values at each date
        returns_matrix: Daily returns
        forward_days: Number of days to look forward for return (21 = ~1 month)
        
    Returns:
        Time series of Information Coefficients
    """
    from scipy.stats import spearmanr
    
    if factor_scores.empty or returns_matrix.empty:
        logger.warning("Empty factor or returns provided to compute_ic")
        return pd.Series()

    # Ensure indices are DatetimeIndex for proper alignment
    factor_scores.index = pd.to_datetime(factor_scores.index)
    returns_matrix.index = pd.to_datetime(returns_matrix.index)
    
    # CRITICAL: Forward returns = returns from tomorrow through forward_days ahead
    # shift(-forward_days) moves dates back, rolling().sum() compounds them
    forward_returns = returns_matrix.shift(-forward_days).rolling(
        forward_days
    ).sum()

    ic_series = {}
    valid_count = 0
    
    for date in factor_scores.index:
        
        if date not in forward_returns.index:
            continue

        # Factor scores today
        scores  = factor_scores.loc[date].dropna()
        
        # Forward returns (from tomorrow)
        fwd_ret = forward_returns.loc[date].dropna()

        # Common stocks
        common = scores.index.intersection(fwd_ret.index)
        
        if len(common) < 20:  # Need minimum 20 stocks for meaningful correlation
            continue

        # Spearman rank correlation between factor and forward returns
        try:
            ic, p_value = spearmanr(scores[common], fwd_ret[common])
            ic_series[date] = ic
            valid_count += 1
        except Exception as e:
            logger.debug(f"IC computation failed for {date}: {e}")
            continue

    ic = pd.Series(ic_series)

    if ic.empty:
        logger.warning(
            "IC series is empty — insufficient forward data available"
        )
        return ic

    # Calculate statistics
    mean_ic = ic.mean()
    std_ic = ic.std()
    ir = mean_ic / std_ic if std_ic > 0 else 0
    
    # Count periods with significant IC
    positive_ic = (ic > 0).sum()
    pct_positive = positive_ic / len(ic) * 100 if len(ic) > 0 else 0

    logger.info(
        f"IC Analysis | Periods: {len(ic)} | Valid obs: {valid_count} | "
        f"Mean IC: {mean_ic:.4f} | Std: {std_ic:.4f} | "
        f"IR (IC/Std): {ir:.4f} | Positive: {pct_positive:.1f}%"
    )
    return ic

def run_factor_engine(close_matrix: pd.DataFrame, returns_matrix: pd.DataFrame, universe_flags: pd.DataFrame, save_path: str = "factors/") -> dict:
    """
    Complete factor calculation pipeline with validation.
    
    Pipeline:
    1. Compute raw factors (momentum, reversal, volatility, quality)
    2. Z-score normalize each factor cross-sectionally
    3. Convert to percentile ranks
    4. Combine into composite score
    5. Compute Information Coefficient for validation
    
    CRITICAL: All operations use only past data, no look-ahead bias.
    
    Args:
        close_matrix: Historical close prices
        returns_matrix: Daily returns
        universe_flags: Boolean DF indicating valid universe each date
        save_path: Directory to save factor outputs
        
    Returns:
        Dictionary with all factor data
    """
    
    if close_matrix.empty or returns_matrix.empty or universe_flags.empty:
        logger.error("Empty input to run_factor_engine")
        return {}
    
    os.makedirs(save_path, exist_ok=True)

    # --- Step 1: Raw Factors ---
    logger.info("=" * 60)
    logger.info("STEP 1: Computing raw factors...")
    logger.info("=" * 60)
    
    momentum_raw   = compute_momentum(close_matrix)
    reversal_raw   = compute_reversal(close_matrix)
    volatility_raw = compute_volatility(returns_matrix)
    quality_raw    = compute_quality_proxy(returns_matrix, close_matrix)
    
    # Validate raw factors
    raw_factors = {
        "momentum": momentum_raw,
        "reversal": reversal_raw,
        "volatility": volatility_raw,
        "quality": quality_raw
    }
    for name, factor in raw_factors.items():
        nan_pct = factor.isna().sum().sum() / factor.size * 100
        logger.info(f"  {name}: shape={factor.shape}, NaN={nan_pct:.1f}%")

    # --- Step 2: Z-Score Normalization ---
    logger.info("=" * 60)
    logger.info("STEP 2: Normalizing factors cross-sectionally...")
    logger.info("=" * 60)
    
    momentum_z   = cross_sectional_zscore(momentum_raw,   universe_flags)
    reversal_z   = cross_sectional_zscore(reversal_raw,   universe_flags)
    volatility_z = cross_sectional_zscore(volatility_raw, universe_flags)
    quality_z    = cross_sectional_zscore(quality_raw,    universe_flags)

    # --- Step 3: Percentile Ranking ---
    logger.info("=" * 60)
    logger.info("STEP 3: Converting factors to percentile ranks...")
    logger.info("=" * 60)
    
    momentum_rank   = cross_sectional_rank(momentum_z,   universe_flags)
    reversal_rank   = cross_sectional_rank(reversal_z,   universe_flags)
    volatility_rank = cross_sectional_rank(volatility_z, universe_flags)
    quality_rank    = cross_sectional_rank(quality_z,    universe_flags)

    # --- Step 4: Composite Score ---
    logger.info("=" * 60)
    logger.info("STEP 4: Computing composite factor...")
    logger.info("=" * 60)
    
    composite = compute_composite_factor(
        momentum_rank, reversal_rank, volatility_rank, quality_rank
    )
    
    if composite.empty:
        logger.error("Composite factor is empty")
        return {}

    # --- Step 5: IC Diagnostics ---
    logger.info("=" * 60)
    logger.info("STEP 5: Computing Information Coefficient...")
    logger.info("=" * 60)
    
    ic_series = compute_ic(composite, returns_matrix)

    # --- Save All Outputs ---
    logger.info("=" * 60)
    logger.info("Saving factor outputs...")
    logger.info("=" * 60)
    
    composite.to_csv(os.path.join(save_path, "composite_scores.csv"))
    momentum_rank.to_csv(os.path.join(save_path, "momentum_rank.csv"))
    reversal_rank.to_csv(os.path.join(save_path, "reversal_rank.csv"))
    volatility_rank.to_csv(os.path.join(save_path, "volatility_rank.csv"))
    quality_rank.to_csv(os.path.join(save_path, "quality_rank.csv"))
    ic_series.to_csv(os.path.join(save_path, "ic_series.csv"))

    logger.info(f"Factor engine complete. Outputs saved to {save_path}")

    return {
        "composite"      : composite,
        "momentum_rank"  : momentum_rank,
        "reversal_rank"  : reversal_rank,
        "volatility_rank": volatility_rank,
        "quality_rank"   : quality_rank,
        "ic_series"      : ic_series
    }

