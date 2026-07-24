"""
Validation layer for quantitative research pipeline.

CRITICAL: This module provides comprehensive input/output validation
and assertions for all pipeline stages. Used to enforce data quality
and catch bugs early before they propagate.

All validators:
1. Check critical invariants
2. Report detailed diagnostics
3. Raise exceptions on validation failure (fail-fast)
4. Log validation results for audit trail
"""

import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger("validators")


# ============================================================================
# MATRIX VALIDATORS
# ============================================================================

def validate_price_matrix(close_matrix: pd.DataFrame, name: str = "close_matrix") -> bool:
    """
    Validate price matrix structure and content.
    
    Args:
        close_matrix: DataFrame with datetime index and ticker columns
        name: Name for logging
        
    Raises:
        ValueError if validation fails
    """
    if close_matrix is None or close_matrix.empty:
        raise ValueError(f"{name}: Matrix is None or empty")
    
    # Check index
    if not isinstance(close_matrix.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: Index must be DatetimeIndex, got {type(close_matrix.index)}")
    
    if not close_matrix.index.is_monotonic_increasing:
        raise ValueError(f"{name}: Index is not sorted (required for walk-forward)")
    
    # Check columns
    if len(close_matrix.columns) == 0:
        raise ValueError(f"{name}: No ticker columns")
    
    # Check for duplicates
    if close_matrix.index.duplicated().any():
        raise ValueError(f"{name}: Duplicate dates found: {close_matrix.index[close_matrix.index.duplicated()].tolist()}")
    
    # Check data types
    numeric_cols = pd.to_numeric(close_matrix.iloc[0], errors='coerce')
    if numeric_cols.isna().all():
        raise ValueError(f"{name}: All values are non-numeric")
    
    # Check for infinities
    if np.isinf(close_matrix.values).any():
        raise ValueError(f"{name}: Contains infinity values")
    
    # Data quality report
    nan_pct = close_matrix.isna().sum().sum() / (close_matrix.shape[0] * close_matrix.shape[1]) * 100
    logger.info(
        f"✓ {name} valid: {close_matrix.shape[0]} dates × {close_matrix.shape[1]} stocks, "
        f"{nan_pct:.1f}% NaN"
    )
    
    return True


def validate_returns_matrix(returns_matrix: pd.DataFrame, name: str = "returns_matrix") -> bool:
    """
    Validate returns matrix (daily returns data).
    
    Args:
        returns_matrix: DataFrame with datetime index and ticker columns
        name: Name for logging
        
    Raises:
        ValueError if validation fails
    """
    if returns_matrix is None or returns_matrix.empty:
        raise ValueError(f"{name}: Matrix is None or empty")
    
    # Check index
    if not isinstance(returns_matrix.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: Index must be DatetimeIndex")
    
    if not returns_matrix.index.is_monotonic_increasing:
        raise ValueError(f"{name}: Index is not sorted")
    
    # Check for duplicates
    if returns_matrix.index.duplicated().any():
        raise ValueError(f"{name}: Duplicate dates")
    
    # Check for infinities
    if np.isinf(returns_matrix.values).any():
        raise ValueError(f"{name}: Contains infinity values (data processing error)")
    
    # Check for extreme values (>100% daily moves are suspicious)
    max_daily_return = returns_matrix.abs().max().max()
    if max_daily_return > 1.0:
        extreme_count = (returns_matrix.abs() > 1.0).sum().sum()
        logger.warning(f"{name}: Found {extreme_count} returns >100% (max: {max_daily_return:.1%})")
    
    # Data quality
    nan_pct = returns_matrix.isna().sum().sum() / (returns_matrix.shape[0] * returns_matrix.shape[1]) * 100
    logger.info(
        f"✓ {name} valid: {returns_matrix.shape[0]} dates × {returns_matrix.shape[1]} stocks, "
        f"{nan_pct:.1f}% NaN"
    )
    
    return True


def validate_universe_flags(universe_flags: pd.DataFrame, close_matrix: pd.DataFrame) -> bool:
    """
    Validate universe membership flags.
    
    Args:
        universe_flags: Boolean DataFrame indicating universe membership
        close_matrix: Reference price matrix for shape/index checking
        
    Raises:
        ValueError if validation fails
    """
    if universe_flags is None or universe_flags.empty:
        raise ValueError("universe_flags: Matrix is None or empty")
    
    # Check shape alignment
    if universe_flags.shape != close_matrix.shape:
        raise ValueError(
            f"universe_flags shape {universe_flags.shape} != "
            f"close_matrix shape {close_matrix.shape}"
        )
    
    # Check index alignment
    if not universe_flags.index.equals(close_matrix.index):
        raise ValueError("universe_flags index != close_matrix index")
    
    if not universe_flags.columns.equals(close_matrix.columns):
        raise ValueError("universe_flags columns != close_matrix columns")
    
    # Check all values are boolean
    unique_vals = universe_flags.values[~pd.isna(universe_flags.values)]
    if not all(v in [True, False, 0, 1] for v in unique_vals):
        raise ValueError(f"universe_flags: Contains non-boolean values")
    
    # Universe size statistics
    daily_counts = universe_flags.sum(axis=1)
    logger.info(
        f"✓ universe_flags valid: min={daily_counts.min():.0f} / "
        f"avg={daily_counts.mean():.0f} / max={daily_counts.max():.0f} stocks/day"
    )
    
    return True


def validate_signal_matrix(signals: pd.DataFrame, universe_flags: pd.DataFrame) -> bool:
    """
    Validate signal matrix (long/short/neutral positions).
    
    Args:
        signals: DataFrame with datetime index, values in {-1, 0, 1}
        universe_flags: Reference universe for validation
        
    Raises:
        ValueError if validation fails
    """
    if signals is None or signals.empty:
        raise ValueError("signals: Matrix is None or empty")
    
    # Check index alignment
    if not signals.index.equals(universe_flags.index):
        raise ValueError("signals index != universe_flags index")
    
    # Check values are in valid set
    unique_signals = set(signals.values[~pd.isna(signals.values)].flatten())
    valid_signals = {-1.0, 0.0, 1.0, -1, 0, 1}
    if not unique_signals.issubset(valid_signals):
        raise ValueError(f"signals: Contains invalid values {unique_signals - valid_signals}")
    
    # Signals should respect universe membership
    # (Can't have signal for stock not in universe)
    # This is a soft check (warn, don't fail)
    signal_mask = signals != 0
    outside_universe = signal_mask & ~universe_flags
    if outside_universe.any().any():
        count = outside_universe.sum().sum()
        logger.warning(f"signals: Found {count} signals outside universe (should be 0)")
    
    # Statistics
    longs = (signals == 1).sum().sum()
    shorts = (signals == -1).sum().sum()
    neutrals = (signals == 0).sum().sum()
    logger.info(
        f"✓ signals valid: {longs} longs, {shorts} shorts, {neutrals} neutrals"
    )
    
    return True


def validate_weight_matrix(weights: pd.DataFrame, signals: pd.DataFrame) -> bool:
    """
    Validate portfolio weight matrix.
    
    CRITICAL: Weights must satisfy:
    - All values in [-1, 1]
    - Sum to net exposure in [-1, 1]
    - Gross exposure in [0, 2] (for long/short)
    - No NaN
    
    Args:
        weights: DataFrame with datetime index and ticker columns
        signals: Reference signal matrix
        
    Raises:
        ValueError if validation fails
    """
    if weights is None or weights.empty:
        raise ValueError("weights: Matrix is None or empty")
    
    # Check for NaN
    if weights.isna().any().any():
        nan_count = weights.isna().sum().sum()
        raise ValueError(f"weights: Contains {nan_count} NaN values")
    
    # Check bounds: each position in [-1, 1]
    if (weights.abs() > 1.0 + 1e-8).any().any():
        max_pos = weights.abs().max().max()
        raise ValueError(f"weights: Position {max_pos:.4f} exceeds ±1.0 bound")
    
    # Check net exposure (sum across positions)
    net_exposures = weights.sum(axis=1)
    if (net_exposures.abs() > 1.0 + 1e-8).any():
        max_net = net_exposures.abs().max()
        raise ValueError(f"weights: Net exposure {max_net:.4f} exceeds ±1.0 bound")
    
    # Check gross exposure (sum of absolute positions)
    gross_exposures = weights.abs().sum(axis=1)
    if (gross_exposures > 2.0 + 1e-8).any():
        max_gross = gross_exposures.max()
        raise ValueError(f"weights: Gross exposure {max_gross:.4f} exceeds 2.0 bound")
    
    # Statistics
    avg_net = net_exposures.mean()
    avg_gross = gross_exposures.mean()
    logger.info(
        f"✓ weights valid: avg net={avg_net:.4f}, avg gross={avg_gross:.4f}, "
        f"leverage={avg_gross:.2%}"
    )
    
    return True


# ============================================================================
# RETURNS VALIDATORS
# ============================================================================

def validate_backtest_returns(oos_returns: pd.Series) -> bool:
    """
    Validate out-of-sample backtest returns.
    
    Args:
        oos_returns: Series of daily returns from backtest
        
    Raises:
        ValueError if validation fails
    """
    if oos_returns is None or oos_returns.empty:
        raise ValueError("oos_returns: Series is None or empty")
    
    # Check for NaN (first few values may be NaN during initial lookback)
    nan_count = oos_returns.isna().sum()
    if nan_count > len(oos_returns) * 0.3:  # Allow up to 30% NaN
        raise ValueError(
            f"oos_returns: Too many NaN values ({nan_count}/{len(oos_returns)})"
        )
    
    # Check for infinities
    if np.isinf(oos_returns).any():
        raise ValueError("oos_returns: Contains infinity values")
    
    # Statistics
    valid_returns = oos_returns[~oos_returns.isna()]
    if len(valid_returns) > 0:
        mean_ret = valid_returns.mean()
        vol = valid_returns.std()
        sharpe = mean_ret / vol * np.sqrt(252) if vol > 0 else 0
        logger.info(
            f"✓ oos_returns valid: {len(valid_returns)} obs, "
            f"mean={mean_ret:.4f}, vol={vol:.4f}, Sharpe={sharpe:.3f}"
        )
    
    return True


# ============================================================================
# RISK METRICS VALIDATORS
# ============================================================================

def validate_risk_metrics(risk_report: dict) -> bool:
    """
    Validate risk calculation outputs.
    
    Args:
        risk_report: Dictionary with VaR, CVaR, and other metrics
        
    Raises:
        ValueError if validation fails
    """
    if not isinstance(risk_report, dict):
        raise ValueError("risk_report: Not a dictionary")
    
    # Check for required metrics
    required_metrics = ['hist_var_95', 'hist_var_99', 'cvar_95', 'cvar_99']
    for metric in required_metrics:
        if metric not in risk_report:
            raise ValueError(f"risk_report: Missing {metric}")
    
    # Check values are reasonable
    var_95 = risk_report['hist_var_95']
    var_99 = risk_report['hist_var_99']
    cvar_95 = risk_report['cvar_95']
    
    if pd.isna(var_95) or pd.isna(var_99) or pd.isna(cvar_95):
        raise ValueError("risk_report: Contains NaN values")
    
    # CVaR should be more extreme than VaR
    if abs(cvar_95) < abs(var_95) - 1e-4:
        logger.warning(f"risk_report: CVaR95 {cvar_95:.4f} < VaR95 {var_95:.4f}")
    
    # VaR99 should be more extreme than VaR95
    if abs(var_99) < abs(var_95) - 1e-4:
        logger.warning(f"risk_report: VaR99 {var_99:.4f} < VaR95 {var_95:.4f}")
    
    logger.info(f"✓ risk_report valid: VaR95={var_95:.4f}, CVaR95={cvar_95:.4f}")
    
    return True


# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def generate_data_quality_report(
    close_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    universe_flags: pd.DataFrame,
    stage_name: str = "pipeline"
) -> dict:
    """
    Generate comprehensive data quality report.
    
    Args:
        close_matrix: Price matrix
        returns_matrix: Returns matrix
        universe_flags: Universe flags
        stage_name: Name of pipeline stage for logging
        
    Returns:
        dict with quality metrics
    """
    report = {
        "stage": stage_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "price_matrix": {
            "shape": close_matrix.shape,
            "nan_pct": close_matrix.isna().sum().sum() / (close_matrix.shape[0] * close_matrix.shape[1]) * 100,
            "date_range": f"{close_matrix.index[0]} to {close_matrix.index[-1]}",
        },
        "returns_matrix": {
            "shape": returns_matrix.shape,
            "nan_pct": returns_matrix.isna().sum().sum() / (returns_matrix.shape[0] * returns_matrix.shape[1]) * 100,
            "mean": returns_matrix.mean().mean(),
            "std": returns_matrix.std().std(),
        },
        "universe": {
            "daily_min": universe_flags.sum(axis=1).min(),
            "daily_max": universe_flags.sum(axis=1).max(),
            "daily_mean": universe_flags.sum(axis=1).mean(),
        }
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA QUALITY REPORT: {stage_name}")
    logger.info(f"{'='*60}")
    for key, val in report.items():
        if key not in ["timestamp", "stage"]:
            logger.info(f"{key}: {val}")
    logger.info(f"{'='*60}\n")
    
    return report
