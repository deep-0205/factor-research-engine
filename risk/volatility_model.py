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

    scaled = returns_series.dropna() * 100

    if len(scaled) < 100:
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

        forecast = result.forecast(horizon=1, reindex=False)
        forecast_variance = forecast.variance.iloc[-1, 0]

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
        logger.warning(f"GARCH fit failed: {e}")
        return {"success": False, "forecast_vol": np.nan}
    
def compute_garch_forecasts(returns_matrix: pd.DataFrame, current_universe: list) -> pd.DataFrame:

    forecasts = {}
    failed    = []

    for i, ticker in enumerate(current_universe):
        if ticker not in returns_matrix.columns:
            continue

        series = returns_matrix[ticker].dropna()
        result = fit_garch(series)

        if result["success"]:
            forecasts[ticker] = result["forecast_vol"]
        else:
            fallback_vol = series.iloc[-20:].std() * np.sqrt(252)
            forecasts[ticker] = fallback_vol
            failed.append(ticker)

    logger.info(
        f"GARCH forecasts complete | "
        f"Success: {len(forecasts) - len(failed)} | "
        f"Fallback: {len(failed)}"
    )
    return pd.Series(forecasts, name="forecast_vol")

def build_rolling_vol_forecasts(
        returns_matrix: pd.DataFrame,
        universe_flags: pd.DataFrame,
        estimation_window: int = 252,
        step: int = 21) -> pd.DataFrame:
 
    dates   = returns_matrix.index
    tickers = returns_matrix.columns
    vol_matrix = pd.DataFrame(np.nan, index=dates, columns=tickers)

    estimation_dates = dates[estimation_window::step]

    logger.info(
        f"Building rolling GARCH forecasts | "
        f"{len(estimation_dates)} estimation dates | "
        f"{len(tickers)} tickers"
    )

    for i, date in enumerate(estimation_dates):
        date_idx = dates.get_loc(date)
        window   = returns_matrix.iloc[
            date_idx - estimation_window: date_idx
        ]

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
            result = fit_garch(series)

            if result["success"]:
                date_forecasts[ticker] = result["forecast_vol"]
            else:
                date_forecasts[ticker] = (
                    series.std() * np.sqrt(252)
                )

        for ticker, vol in date_forecasts.items():
            vol_matrix.loc[date, ticker] = vol

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(estimation_dates)} dates"
            )

    vol_matrix = vol_matrix.ffill()

    logger.info("Rolling GARCH forecast matrix complete")
    return vol_matrix

def fit_index_garch(index_returns: pd.Series) -> dict:

    result = fit_garch(index_returns)

    if not result["success"]:
        return {"forecast_vol": np.nan, "regime": "UNKNOWN"}

    vol = result["forecast_vol"]

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

    scalars = target_vol / vol_forecasts.replace(0, np.nan)

    scalars = scalars.clip(lower=min_scalar, upper=max_scalar)

    logger.info(
        f"Vol scalars | "
        f"Mean: {scalars.mean():.3f} | "
        f"Min: {scalars.min():.3f} | "
        f"Max: {scalars.max():.3f}"
    )
    return scalars

def run_garch_diagnostics(
        returns_matrix: pd.DataFrame,
        sample_tickers: list = None,
        n_sample: int = 20) -> pd.DataFrame:

    if sample_tickers is None:
        cols = returns_matrix.columns.tolist()
        sample_tickers = np.random.choice(
            cols, size=min(n_sample, len(cols)), replace=False
        )

    diagnostics = []

    for ticker in sample_tickers:
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

    diag_df = pd.DataFrame(diagnostics).set_index("ticker")

    logger.info(
        f"GARCH Diagnostics | "
        f"Avg persistence: {diag_df['persistence'].mean():.4f} | "
        f"Avg forecast vol: {diag_df['forecast_vol'].mean():.4f}"
    )
    return diag_df

def run_volatility_model(
        returns_matrix: pd.DataFrame,
        universe_flags: pd.DataFrame,
        current_universe: list,
        save_path: str = "risk/") -> dict:

    os.makedirs(save_path, exist_ok=True)

    logger.info("Computing GARCH forecasts for current universe...")
    current_forecasts = compute_garch_forecasts(
        returns_matrix, current_universe
    )

    logger.info("Building rolling historical vol forecasts...")
    vol_matrix = build_rolling_vol_forecasts(
        returns_matrix, universe_flags
    )

    logger.info("Fitting GARCH on market index...")
    try:
        nifty = returns_matrix.mean(axis=1)   
        index_result = fit_index_garch(nifty)
    except Exception as e:
        logger.warning(f"Index GARCH failed: {e}")
        index_result = {"forecast_vol": np.nan, "regime": "UNKNOWN"}

    logger.info("Computing position scalars...")
    vol_scalars = compute_vol_scalars(current_forecasts)

    logger.info("Running GARCH diagnostics...")
    diagnostics = run_garch_diagnostics(returns_matrix)

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

    logger.info("Volatility model complete. All outputs saved to risk/")

    return {
        "current_forecasts": current_forecasts,
        "vol_matrix"       : vol_matrix,
        "index_result"     : index_result,
        "vol_scalars"      : vol_scalars,
        "diagnostics"      : diagnostics
    }

