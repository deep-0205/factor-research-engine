import pandas as pd
import numpy as np
import os
import yaml
from logger import get_logger

logger = get_logger("portfolio_construction")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def equal_weight(signals: pd.Series) -> pd.Series:
    """
    Equal-weight positions: 1/n_long for longs, -1/n_short for shorts.
    
    Args:
        signals: Binary signal Series (1, 0, -1)
        
    Returns:
        Weight Series
    """
    
    if signals.empty:
        logger.warning("Empty signals provided to equal_weight")
        return pd.Series()

    weights = pd.Series(0.0, index=signals.index)

    long_mask  = signals ==  1
    short_mask = signals == -1

    n_long  = long_mask.sum()
    n_short = short_mask.sum()

    if n_long > 0:
        weights[long_mask]  =  1.0 / n_long

    if n_short > 0:
        weights[short_mask] = -1.0 / n_short

    # Validate weights sum to at most 1 long + 1 short
    assert abs(weights[weights > 0].sum() - 1.0) < 1e-6 or n_long == 0, \
        f"Long weights don't sum to 1: {weights[weights > 0].sum()}"
    assert abs(weights[weights < 0].sum() + 1.0) < 1e-6 or n_short == 0, \
        f"Short weights don't sum to -1: {weights[weights < 0].sum()}"
    
    return weights


def volatility_weight(signals: pd.Series, vol_forecasts: pd.Series) -> pd.Series:
    """
    Volatility-weighted positions: weight inversely by forecasted vol.
    
    Lower vol positions get higher weight (risk parity within signal group).
    
    Args:
        signals: Binary signal Series
        vol_forecasts: Volatility forecasts (aligned to signals)
        
    Returns:
        Volatility-weighted positions
    """
    
    if signals.empty or vol_forecasts.empty:
        logger.warning("Empty signals or vol_forecasts provided to volatility_weight")
        return pd.Series()

    weights = pd.Series(0.0, index=signals.index)

    active = signals[signals != 0]
    if active.empty:
        return weights

    # Align volatility forecasts to active positions
    vols = vol_forecasts.reindex(active.index)
    
    # Fill missing vols with median (shouldn't happen but safety check)
    if vols.isna().any():
        median_vol = vol_forecasts.median()
        vols = vols.fillna(median_vol)
    
    # Replace zero volatility with median (to avoid division by zero)
    if (vols == 0).any():
        median_vol = vols[vols > 0].median()
        if pd.isna(median_vol):
            logger.warning("No valid volatilities found")
            return equal_weight(signals)
        vols = vols.replace(0, median_vol)

    # Inverse volatility weighting
    raw = active / vols

    # Separate longs and shorts
    long_raw  = raw[raw > 0]
    short_raw = raw[raw < 0]

    # Normalize long weights to sum to 1
    if not long_raw.empty:
        weights[long_raw.index]  = long_raw  / long_raw.sum()

    # Normalize short weights to sum to -1
    # Note: short_raw is already negative, so divide by abs().sum() then multiply by -1
    if not short_raw.empty:
        weights[short_raw.index] = -(short_raw.abs() / short_raw.abs().sum())

    return weights


def risk_parity_weight(signals: pd.Series, vol_forecasts: pd.Series, corr_matrix: pd.DataFrame = None) -> pd.Series:
    """
    Risk parity: positions inversely weighted by volatility.
    
    Each unit of risk gets equal allocation (inverse vol weighting).
    
    Args:
        signals: Binary signal Series
        vol_forecasts: Volatility forecasts
        corr_matrix: Optional correlation matrix (not used currently)
        
    Returns:
        Risk parity weights
    """
    
    if signals.empty or vol_forecasts.empty:
        logger.warning("Empty signals or vol_forecasts provided to risk_parity_weight")
        return pd.Series()

    weights = pd.Series(0.0, index=signals.index)

    active = signals[signals != 0]
    if active.empty:
        return weights

    # Align and validate vols
    vols = vol_forecasts.reindex(active.index).fillna(vol_forecasts.median())
    vols = vols.replace(0, vol_forecasts.median())

    # Inverse volatility
    inv_vol = 1.0 / vols

    long_mask  = active > 0
    short_mask = active < 0

    long_inv  = inv_vol[long_mask]
    short_inv = inv_vol[short_mask]

    # Normalize
    if not long_inv.empty:
        weights[long_inv.index] = long_inv / long_inv.sum()

    if not short_inv.empty:
        weights[short_inv.index] = -(short_inv / short_inv.sum())

    logger.debug(
        f"Risk parity | Longs: {long_mask.sum()} | "
        f"Shorts: {short_mask.sum()}"
    )
    return weights


def apply_exposure_controls(
        weights: pd.Series,
        max_gross: float = 1.0,
        max_net: float = 0.20,
        max_single_position: float = 0.05) -> pd.Series:
    """
    Apply exposure constraints to portfolio weights.
    
    CRITICAL ORDER OF OPERATIONS:
    1. Clip individual positions to max size
    2. Scale to max gross exposure
    3. Scale to max net exposure (delta-neutral bias)
    
    Args:
        weights: Raw weights (unconstrained)
        max_gross: Max |long| + |short| (e.g., 1.0 = 100% capital)
        max_net: Max |long - short| (e.g., 0.20 = 20% bias)
        max_single_position: Max single position size
        
    Returns:
        Constrained weights
    """
    
    if weights.empty:
        logger.warning("Empty weights provided to apply_exposure_controls")
        return pd.Series()

    w = weights.copy()

    # Step 1: Clip individual positions
    w = w.clip(lower=-max_single_position, upper=max_single_position)

    # Step 2: Scale to max gross exposure
    gross = w.abs().sum()
    if gross > max_gross + 1e-8:  # Small epsilon for numerical stability
        w = w * (max_gross / gross)

    # Step 3: Scale to max net exposure
    net = w.sum()
    if abs(net) > max_net + 1e-8:
        if net > max_net:
            # Positive net: reduce longs
            excess = net - max_net
            long_sum = w[w > 0].sum()
            if long_sum > 1e-8:
                w[w > 0] *= (1 - excess / long_sum)
        else:
            # Negative net: reduce shorts (make less negative)
            excess = abs(net) - max_net
            short_sum = w[w < 0].abs().sum()
            if short_sum > 1e-8:
                w[w < 0] *= (1 - excess / short_sum)

    # Validate final constraints
    final_gross = w.abs().sum()
    final_net = w.sum()
    
    if final_gross > max_gross + 1e-6:
        logger.warning(
            f"Gross exposure constraint violated: {final_gross:.4f} > {max_gross:.4f}"
        )
    if abs(final_net) > max_net + 1e-6:
        logger.warning(
            f"Net exposure constraint violated: {final_net:.4f} > ±{max_net:.4f}"
        )

    logger.debug(
        f"Exposure Controls | "
        f"Gross: {final_gross:.4f} ({max_gross:.4f} max) | "
        f"Net: {final_net:.4f} (±{max_net:.4f} max) | "
        f"Max pos: {w.abs().max():.4f}"
    )
    return w


def apply_sector_neutrality(weights: pd.Series, sector_map: dict, max_sector_net: float = 0.10) -> pd.Series:
    """
    Apply sector-level exposure constraints.
    
    Limits sector-level net exposure (long - short) to max_sector_net.
    
    Args:
        weights: Portfolio weights
        sector_map: Dict mapping ticker → sector
        max_sector_net: Max net exposure per sector
        
    Returns:
        Sector-constrained weights
    """
    
    if weights.empty or not sector_map:
        return weights

    w = weights.copy()

    # Build sector groupings
    sector_to_tickers = {}
    for ticker, sector in sector_map.items():
        if ticker not in w.index:
            continue
        sector_to_tickers.setdefault(sector, []).append(ticker)

    # Apply constraints per sector
    for sector, tickers in sector_to_tickers.items():
        sector_weights = w[tickers]
        sector_net     = sector_weights.sum()

        # Already within limit
        if abs(sector_net) <= max_sector_net + 1e-8:
            continue  

        # Reduce longs if net is too positive
        if sector_net > max_sector_net:
            excess = sector_net - max_sector_net
            long_tickers = [t for t in tickers if w[t] > 0]
            long_sum = w[long_tickers].sum()
            if long_sum > 1e-8:
                for t in long_tickers:
                    w[t] *= (1 - excess / long_sum)
        
        # Reduce shorts if net is too negative
        else:
            excess = abs(sector_net) - max_sector_net
            short_tickers = [t for t in tickers if w[t] < 0]
            short_sum = w[short_tickers].abs().sum()
            if short_sum > 1e-8:
                for t in short_tickers:
                    w[t] *= (1 - excess / short_sum)

        logger.debug(
            f"Sector {sector}: net {sector_net:.4f} → {w[tickers].sum():.4f}"
        )

    return w


def build_weight_matrix(
        signals_df: pd.DataFrame,
        vol_matrix: pd.DataFrame,
        method: str = "volatility",
        sector_map: dict = None) -> pd.DataFrame:
    """
    Build weight matrix for entire backtest period.
    
    CRITICAL:
    - Validates signals_df and vol_matrix have compatible indices
    - For each date, constructs weights using that day's signals and vol forecasts
    - Applies exposure and sector constraints
    
    Args:
        signals_df: Daily signals (dates × tickers)
        vol_matrix: Volatility forecasts (dates × tickers)
        method: "equal", "volatility", or "risk_parity"
        sector_map: Optional sector mapping
        
    Returns:
        Weight matrix (dates × tickers)
    """
    
    if signals_df.empty or vol_matrix.empty:
        logger.error("Empty signals_df or vol_matrix")
        return pd.DataFrame()
    
    # Validate indices
    signals_df.index = pd.to_datetime(signals_df.index)
    vol_matrix.index = pd.to_datetime(vol_matrix.index)
    
    common_dates = signals_df.index.intersection(vol_matrix.index)
    if len(common_dates) == 0:
        logger.error(
            f"No common dates between signals ({len(signals_df)}) "
            f"and vol_matrix ({len(vol_matrix)})"
        )
        return pd.DataFrame()
    
    logger.info(
        f"Building weight matrix | Method: {method} | "
        f"Dates: {len(signals_df)}, Vol dates: {len(vol_matrix)}, "
        f"Common: {len(common_dates)}"
    )

    config = load_config()
    weight_matrix = pd.DataFrame(0.0, index=signals_df.index, columns=signals_df.columns)

    for date_idx, date in enumerate(signals_df.index):
        signals = signals_df.loc[date]

        # Skip if no active signals
        if (signals != 0).sum() == 0:
            continue

        # Get volatility forecasts for this date
        if date in vol_matrix.index:
            vol_forecasts = vol_matrix.loc[date]
        else:
            # Fallback: use median vol across all dates
            vol_forecasts = pd.Series(vol_matrix.values[~np.isnan(vol_matrix.values)].mean(), 
                                     index=signals.index)
            logger.debug(f"No vol forecast for {date}, using default")

        # Compute weights
        if method == "equal":
            weights = equal_weight(signals)
        elif method == "volatility":
            weights = volatility_weight(signals, vol_forecasts)
        elif method == "risk_parity":
            weights = risk_parity_weight(signals, vol_forecasts)
        else:
            logger.error(f"Unknown method: {method}")
            raise ValueError(f"Unknown method: {method}")

        # Apply constraints
        weights = apply_exposure_controls(
            weights,
            max_gross=config["portfolio"]["max_gross_exposure"],
            max_net=config["portfolio"]["max_net_exposure"]
        )

        if sector_map:
            weights = apply_sector_neutrality(weights, sector_map)

        weight_matrix.loc[date] = weights
        
        if (date_idx + 1) % 50 == 0:
            logger.debug(f"Processed {date_idx + 1}/{len(signals_df)} dates")

    logger.info(
        f"Weight matrix built | Method: {method} | "
        f"Shape: {weight_matrix.shape} | "
        f"Avg gross: {weight_matrix.abs().sum(axis=1).mean():.4f} | "
        f"Avg net: {weight_matrix.sum(axis=1).mean():.4f}"
    )
    return weight_matrix


def compute_portfolio_stats(weight_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portfolio-level statistics for each day.
    
    Metrics:
    - gross_exposure: |long| + |short|
    - net_exposure: long - short
    - n_long, n_short: position counts
    - max_position: largest single position
    - concentration: Herfindahl index of long positions
    
    Args:
        weight_matrix: Weight matrix
        
    Returns:
        Stats DataFrame (dates × metrics)
    """
    
    if weight_matrix.empty:
        logger.warning("Empty weight_matrix provided to compute_portfolio_stats")
        return pd.DataFrame()

    stats = pd.DataFrame(index=weight_matrix.index)

    stats["gross_exposure"] = weight_matrix.abs().sum(axis=1)
    stats["net_exposure"]   = weight_matrix.sum(axis=1)
    stats["n_long"]         = (weight_matrix > 0).sum(axis=1)
    stats["n_short"]        = (weight_matrix < 0).sum(axis=1)
    stats["max_position"]   = weight_matrix.abs().max(axis=1)

    # Concentration: Herfindahl index of long positions
    long_weights = weight_matrix.clip(lower=0)
    long_sum = long_weights.sum(axis=1).replace(0, np.nan)
    normalized = long_weights.div(long_sum, axis=0)
    stats["concentration"] = (normalized ** 2).sum(axis=1)

    # Remove NaNs from concentration (when no long positions)
    stats["concentration"] = stats["concentration"].fillna(0)

    logger.info(
        f"Portfolio Stats | "
        f"Avg gross: {stats['gross_exposure'].mean():.4f} | "
        f"Avg net: {stats['net_exposure'].mean():.4f} | "
        f"Avg longs: {stats['n_long'].mean():.1f} | "
        f"Avg shorts: {stats['n_short'].mean():.1f} | "
        f"Avg concentration: {stats['concentration'].mean():.4f}"
    )
    return stats


def run_portfolio_construction(
        signals_df: pd.DataFrame,
        vol_matrix: pd.DataFrame,
        sector_map: dict = None,
        save_path: str = "portfolio/") -> dict:
    """
    Complete portfolio construction pipeline.
    
    Builds three weight schemes:
    1. Equal-weight (baseline)
    2. Volatility-weighted (risk parity within signals)
    3. Risk parity (full inverse vol weighting)
    
    Args:
        signals_df: Daily signals
        vol_matrix: Volatility forecasts
        sector_map: Optional sector mapping
        save_path: Output directory
        
    Returns:
        Dictionary with weight matrices and stats
    """
    
    if signals_df.empty or vol_matrix.empty:
        logger.error("Empty signals_df or vol_matrix provided to run_portfolio_construction")
        return {}

    os.makedirs(save_path, exist_ok=True)

    methods = ["equal", "volatility", "risk_parity"]
    weight_matrices = {}

    logger.info("=" * 60)
    logger.info("Building portfolio weight matrices...")
    logger.info("=" * 60)

    for method in methods:
        logger.info(f"Building {method} portfolio...")
        try:
            w = build_weight_matrix(
                signals_df=signals_df,
                vol_matrix=vol_matrix,
                method=method,
                sector_map=sector_map
            )
            
            if w.empty:
                logger.error(f"Empty weight matrix for method: {method}")
                continue
            
            weight_matrices[method] = w
            w.to_csv(os.path.join(save_path, f"weights_{method}.csv"))
            logger.info(f"  {method} portfolio saved")
        except Exception as e:
            logger.error(f"Error building {method} portfolio: {e}")
            continue

    if not weight_matrices:
        logger.error("No weight matrices built successfully")
        return {}

    # Use volatility-weighted as primary
    primary_weights = weight_matrices.get("volatility", weight_matrices[methods[0]])

    stats = compute_portfolio_stats(primary_weights)
    stats.to_csv(os.path.join(save_path, "portfolio_stats.csv"))

    logger.info("Portfolio construction complete. Outputs saved.")

    return {
        "weights"         : primary_weights,
        "weights_equal"   : weight_matrices.get("equal", pd.DataFrame()),
        "weights_rp"      : weight_matrices.get("risk_parity", pd.DataFrame()),
        "portfolio_stats" : stats
    }
