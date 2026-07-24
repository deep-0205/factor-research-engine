import pandas as pd
import numpy as np
import os
import warnings
import yaml
from arch import arch_model
from logger import get_logger

logger = get_logger("volatility_model")
warnings.filterwarnings("ignore")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def fit_garch(returns_series: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    Fit GARCH(1,1) model to returns and forecast next-day volatility.
    
    CRITICAL:
    - Uses only past data in returns_series
    - Forecast is 1-day ahead (next day's predicted vol)
    - Scales from daily to annual (× √252)
    - Requires minimum 100 observations
    
    Args:
        returns_series: Daily returns (past data only)
        p, q: GARCH order (default 1,1)
        
    Returns:
        Dictionary with forecast_vol and model parameters
    """

    # Remove NaNs and scale to percentage for GARCH fitting
    scaled = returns_series.dropna() * 100

    if len(scaled) < 100:
        logger.debug(f"Insufficient data for GARCH: {len(scaled)} < 100")
        return {"success": False, "forecast_vol": np.nan}

    try:
        model = arch_model(
            scaled,
            vol="Garch",
            p=p,
            q=q,
            mean="Constant",
            dist="normal"
        )

        result = model.fit(
            disp="off",
            options={"maxiter": 200}
        )

        # Forecast next day's variance
        forecast = result.forecast(horizon=1, reindex=False)
        forecast_variance = forecast.variance.iloc[-1, 0]

        # Convert to annualized volatility
        forecast_vol = np.sqrt(forecast_variance) / 100 * np.sqrt(252)

        params = result.params

        return {
            "success"      : True,
            "forecast_vol" : forecast_vol,
            "omega"        : params.get("omega", np.nan),
            "alpha"        : params.get("alpha[1]", np.nan),
            "beta"         : params.get("beta[1]", np.nan),
            "persistence"  : params.get("alpha[1]", 0) + params.get("beta[1]", 0),
            "aic"          : result.aic,
            "bic"          : result.bic
        }

    except Exception as e:
        logger.debug(f"GARCH fit failed: {e}")
        return {"success": False, "forecast_vol": np.nan}


def compute_garch_forecasts(returns_matrix: pd.DataFrame, current_universe: list) -> pd.Series:
    """
    Compute GARCH volatility forecasts for current universe.
    
    For each stock, fits GARCH on its full available history
    and produces forecast. Falls back to realized vol if GARCH fails.
    
    Args:
        returns_matrix: Full returns matrix
        current_universe: List of tickers to forecast
        
    Returns:
        Series of volatility forecasts (one per ticker)
    """

    if returns_matrix.empty or not current_universe:
        logger.error("Empty returns or universe")
        return pd.Series()

    forecasts = {}
    failed = []

    for i, ticker in enumerate(current_universe):
        if ticker not in returns_matrix.columns:
            logger.debug(f"Ticker not in returns: {ticker}")
            continue

        series = returns_matrix[ticker].dropna()
        
        if series.empty:
            logger.debug(f"No returns data: {ticker}")
            continue

        result = fit_garch(series)

        if result["success"]:
            forecasts[ticker] = result["forecast_vol"]
        else:
            # Fallback: recent realized volatility (last 20 days)
            fallback_vol = series.iloc[-20:].std() * np.sqrt(252)
            forecasts[ticker] = fallback_vol
            failed.append(ticker)

        if (i + 1) % 50 == 0:
            logger.debug(f"Processed {i + 1}/{len(current_universe)} forecasts")

    logger.info(
        f"GARCH Forecasts | "
        f"Success: {len(forecasts) - len(failed)} | "
        f"Fallback: {len(failed)}"
    )
    return pd.Series(forecasts, name="forecast_vol")


def build_rolling_vol_forecasts(
        returns_matrix: pd.DataFrame,
        universe_flags: pd.DataFrame,
        estimation_window: int = 252,
        step: int = 21) -> pd.DataFrame:
    """
    Build rolling vol forecasts using walk-forward GARCH.
    
    CRITICAL: NO LOOK-AHEAD BIAS
    - For each estimation date d:
      - Use data from [d - estimation_window] to [d]
      - Fit GARCH only on this window (past data)
      - Forecast is for date d (using only information available up to d)
    - Forward-fill between estimation dates
    
    Args:
        returns_matrix: Daily returns
        universe_flags: Universe indicator DF
        estimation_window: Days to use for fitting (252 = 1 year)
        step: Days between estimation dates (21 = monthly)
        
    Returns:
        Vol forecast matrix (dates × tickers)
    """

    dates = returns_matrix.index
    tickers = returns_matrix.columns
    
    # Initialize forecast matrix
    vol_matrix = pd.DataFrame(np.nan, index=dates, columns=tickers)

    # Estimation dates: skip first estimation_window days, then every 'step' days
    estimation_dates = dates[estimation_window::step]

    if len(estimation_dates) == 0:
        logger.error(f"No estimation dates: {len(dates)} data points, {estimation_window}d window, {step}d step")
        return vol_matrix

    logger.info(
        f"Building rolling GARCH forecasts | "
        f"{len(estimation_dates)} estimation dates | "
        f"{len(tickers)} tickers"
    )

    for i, date in enumerate(estimation_dates):
        date_idx = dates.get_loc(date)
        
        # CRITICAL: Use only past data [d - window : d]
        window_start_idx = max(0, date_idx - estimation_window)
        window = returns_matrix.iloc[window_start_idx:date_idx]
        
        if window.empty:
            logger.warning(f"Empty window for date {date}")
            continue

        # Get universe for this date
        if date in universe_flags.index:
            active = universe_flags.loc[date]
            active_tickers = active[active].index.tolist()
        else:
            active_tickers = tickers.tolist()

        date_forecasts = {}

        for ticker in active_tickers:
            if ticker not in window.columns:
                continue
            
            series = window[ticker].dropna()
            
            if len(series) < 30:  # Need minimum data for meaningful GARCH
                continue
            
            result = fit_garch(series)

            if result["success"]:
                date_forecasts[ticker] = result["forecast_vol"]
            else:
                # Fallback to realized vol if GARCH fails
                date_forecasts[ticker] = series.std() * np.sqrt(252)

        # Record forecasts for this estimation date
        for ticker, vol in date_forecasts.items():
            vol_matrix.loc[date, ticker] = vol

        if (i + 1) % max(1, len(estimation_dates) // 10) == 0:
            logger.info(f"  Progress: {i+1}/{len(estimation_dates)} dates")

    # Forward-fill between estimation dates
    vol_matrix = vol_matrix.ffill()
    
    # For dates before first estimation, backfill if needed
    vol_matrix = vol_matrix.bfill()

    # Validate no NaNs if we have data
    nan_count = vol_matrix.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Remaining NaNs in vol matrix: {nan_count}/{vol_matrix.size}")

    logger.info("Rolling GARCH forecast matrix complete")
    return vol_matrix


def fit_index_garch(index_returns: pd.Series) -> dict:
    """
    Fit GARCH model to market index returns.
    
    Classifies market regime based on volatility forecast.
    
    Args:
        index_returns: Daily index returns (past data)
        
    Returns:
        Dictionary with forecast vol and market regime
    """

    result = fit_garch(index_returns)

    if not result["success"]:
        logger.warning("Index GARCH fit failed")
        return {"forecast_vol": np.nan, "regime": "UNKNOWN"}

    vol = result["forecast_vol"]

    # Classify regime
    if vol < 0.12:
        regime = "LOW"
    elif vol < 0.20:
        regime = "MEDIUM"
    else:
        regime = "HIGH"

    logger.info(
        f"Index GARCH | Forecast Vol: {vol:.2%} | Regime: {regime}"
    )

    return {
        "forecast_vol": vol,
        "regime"      : regime,
        "alpha"       : result.get("alpha", np.nan),
        "beta"        : result.get("beta", np.nan),
        "persistence" : result.get("persistence", np.nan)
    }


def compute_vol_scalars(
        vol_forecasts: pd.Series,
        target_vol: float = 0.20,
        min_scalar: float = 0.25,
        max_scalar: float = 2.00) -> pd.Series:
    """
    Compute position scalars based on vol forecasts.
    
    Scales positions inversely to vol: low vol → high scalar (more capital).
    
    Args:
        vol_forecasts: Forecasted volatilities
        target_vol: Target portfolio volatility (0.20 = 20%)
        min_scalar: Minimum position scale
        max_scalar: Maximum position scale
        
    Returns:
        Scalars for position sizing
    """

    if vol_forecasts.empty:
        logger.warning("Empty vol_forecasts")
        return pd.Series()

    # Inverse vol scaling
    scalars = target_vol / vol_forecasts.replace(0, np.nan)

    # Clip to bounds
    scalars = scalars.clip(lower=min_scalar, upper=max_scalar)

    logger.info(
        f"Vol Scalars | "
        f"Mean: {scalars.mean():.3f} | "
        f"Min: {scalars.min():.3f} | "
        f"Max: {scalars.max():.3f} | "
        f"Valid: {scalars.notna().sum()}"
    )
    return scalars


def run_garch_diagnostics(
        returns_matrix: pd.DataFrame,
        sample_tickers: list = None,
        n_sample: int = 20) -> pd.DataFrame:
    """
    Run GARCH diagnostics on sample of tickers.
    
    Args:
        returns_matrix: Returns data
        sample_tickers: Specific tickers to diagnose
        n_sample: Number of random tickers if sample_tickers not provided
        
    Returns:
        Diagnostics DataFrame
    """

    if sample_tickers is None:
        cols = returns_matrix.columns.tolist()
        sample_tickers = np.random.choice(
            cols, size=min(n_sample, len(cols)), replace=False
        )

    diagnostics = []

    for ticker in sample_tickers:
        if ticker not in returns_matrix.columns:
            continue
        
        series = returns_matrix[ticker].dropna()
        result = fit_garch(series)

        if result["success"]:
            diagnostics.append({
                "ticker"     : ticker,
                "forecast_vol": result["forecast_vol"],
                "omega"      : result["omega"],
                "alpha"      : result["alpha"],
                "beta"       : result["beta"],
                "persistence": result["persistence"],
                "aic"        : result["aic"],
                "bic"        : result["bic"],
                "n_obs"      : len(series)
            })

    if not diagnostics:
        logger.warning("No diagnostics computed")
        return pd.DataFrame()

    diag_df = pd.DataFrame(diagnostics).set_index("ticker")

    logger.info(
        f"GARCH Diagnostics | "
        f"Models fit: {len(diag_df)} | "
        f"Avg persistence: {diag_df['persistence'].mean():.4f} | "
        f"Avg forecast vol: {diag_df['forecast_vol'].mean():.4f}"
    )
    return diag_df


def run_volatility_model(
        returns_matrix: pd.DataFrame,
        universe_flags: pd.DataFrame,
        current_universe: list,
        save_path: str = "risk/") -> dict:
    """
    Complete volatility modeling pipeline.
    
    CRITICAL: All forecasts use only past data.
    
    Pipeline:
    1. GARCH forecasts for current universe
    2. Rolling GARCH vol matrix for backtest
    3. Index GARCH and regime classification
    4. Position scalars based on vol
    5. Diagnostics on sample of stocks
    
    Args:
        returns_matrix: Daily returns
        universe_flags: Universe indicators
        current_universe: Current stocks to forecast
        save_path: Output directory
        
    Returns:
        Dictionary with all vol model outputs
    """

    if returns_matrix.empty or universe_flags.empty:
        logger.error("Empty returns or universe_flags")
        return {}

    os.makedirs(save_path, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1: Computing GARCH forecasts for current universe...")
    logger.info("=" * 60)
    
    current_forecasts = compute_garch_forecasts(
        returns_matrix, current_universe
    )

    if current_forecasts.empty:
        logger.error("Failed to compute current forecasts")
        return {}

    logger.info("=" * 60)
    logger.info("STEP 2: Building rolling historical vol forecasts...")
    logger.info("=" * 60)
    
    vol_matrix = build_rolling_vol_forecasts(
        returns_matrix, universe_flags
    )

    logger.info("=" * 60)
    logger.info("STEP 3: Fitting GARCH on market index...")
    logger.info("=" * 60)
    
    try:
        # Use equal-weighted index returns
        nifty = returns_matrix.mean(axis=1)
        index_result = fit_index_garch(nifty)
    except Exception as e:
        logger.error(f"Index GARCH failed: {e}")
        index_result = {"forecast_vol": np.nan, "regime": "UNKNOWN"}

    logger.info("=" * 60)
    logger.info("STEP 4: Computing position scalars...")
    logger.info("=" * 60)
    
    vol_scalars = compute_vol_scalars(current_forecasts)

    logger.info("=" * 60)
    logger.info("STEP 5: Running GARCH diagnostics...")
    logger.info("=" * 60)
    
    diagnostics = run_garch_diagnostics(returns_matrix)

    # Save outputs
    logger.info("Saving volatility model outputs...")
    current_forecasts.to_csv(
        os.path.join(save_path, "current_vol_forecasts.csv")
    )
    vol_matrix.to_csv(
        os.path.join(save_path, "rolling_vol_matrix.csv")
    )
    vol_scalars.to_csv(
        os.path.join(save_path, "vol_scalars.csv")
    )
    diagnostics.to_csv(
        os.path.join(save_path, "garch_diagnostics.csv")
    )
    pd.Series(index_result).to_csv(
        os.path.join(save_path, "index_garch.csv")
    )

    logger.info(f"Volatility model complete. Outputs saved to {save_path}")

    return {
        "current_forecasts": current_forecasts,
        "vol_matrix"       : vol_matrix,
        "index_result"     : index_result,
        "vol_scalars"      : vol_scalars,
        "diagnostics"      : diagnostics
    }
