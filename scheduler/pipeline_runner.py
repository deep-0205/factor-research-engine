import pandas as pd
import numpy as np
import os
import yaml
import traceback
from datetime import datetime, time
from logger import get_logger

logger = get_logger("pipeline_runner")


def initialize_reproducibility(seed: int = 42) -> None:
    """
    Initialize random seeds for full reproducibility.
    
    CRITICAL: All quantitative results must be reproducible across runs.
    This function ensures consistent behavior from:
    - NumPy random operations (array generation, sampling)
    - Python random module
    - Pandas operations with randomization
    
    Must be called at pipeline start, BEFORE any random operations.
    
    Args:
        seed: Random seed value (default 42)
    """
    np.random.seed(seed)
    np.random.RandomState(seed)
    
    import random
    random.seed(seed)
    
    # Pandas operations with seed
    pd.np.random.seed(seed) if hasattr(pd, 'np') else None
    
    logger.info(f"Reproducibility initialized: seed={seed}")


def load_config():
    """Load and validate configuration file."""
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        
        # ✓ Validate critical config keys exist
        required_keys = ["project", "data", "universe", "backtest", "portfolio"]
        for key in required_keys:
            if key not in config:
                logger.warning(f"Missing config key: {key}")
        
        logger.debug(f"Config loaded: {config.get('project', {}).get('name', 'UNKNOWN')}")
        return config
    
    except FileNotFoundError:
        logger.error("config.yaml not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Config YAML error: {e}")
        raise

def step_update_data(config: dict) -> bool:

    try:
        from data.download_data import download_ohlcv
        from data.fetch_universe import get_nifty500_tickers

        logger.info("Step 1: Updating market data...")

        tickers = get_nifty500_tickers()

        end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
        start_date = (
            pd.Timestamp.today() - pd.Timedelta(days=5)
        ).strftime("%Y-%m-%d")

        download_ohlcv(
            tickers=tickers,
            start=start_date,
            end=end_date,
            raw_path=config["data"]["raw_path"]
        )

        logger.info("Step 1 complete: Data updated")
        return True

    except Exception as e:
        logger.error(f"Step 1 FAILED: {e}\n{traceback.format_exc()}")
        return False


def step_process_data(config: dict) -> tuple:

    try:
        from data.processed.process_data import build_close_matrix, compute_returns

        logger.info("Step 2: Processing data...")

        close = build_close_matrix(
            raw_path=config["data"]["raw_path"],
            processed_path=config["data"]["processed_path"]
        )
        returns = compute_returns(close, config["data"]["processed_path"])

        logger.info(
            f"Step 2 complete: {close.shape[1]} stocks, "
            f"{len(close)} days"
        )
        return close, returns

    except Exception as e:
        logger.error(f"Step 2 FAILED: {e}\n{traceback.format_exc()}")
        return None, None


def step_build_universe(config: dict, close: pd.DataFrame) -> dict:

    try:
        from universe.universe import build_universe

        logger.info("Step 3: Building universe...")

        universe_data = build_universe(
            close_matrix=close,
            raw_path=config["data"]["raw_path"]
        )

        logger.info(
            f"Step 3 complete: "
            f"{len(universe_data['current_universe'])} stocks"
        )
        return universe_data

    except Exception as e:
        logger.error(f"Step 3 FAILED: {e}\n{traceback.format_exc()}")
        return None


def step_compute_factors(returns: pd.DataFrame, close: pd.DataFrame, universe_flags: pd.DataFrame) -> dict:

    try:
        from factors.factor_engine import run_factor_engine

        logger.info("Step 4: Computing factors...")

        factor_data = run_factor_engine(
            close_matrix=close,
            returns_matrix=returns,
            universe_flags=universe_flags
        )

        logger.info("Step 4 complete: Factors computed")
        return factor_data

    except Exception as e:
        logger.error(f"Step 4 FAILED: {e}\n{traceback.format_exc()}")
        return None


def step_generate_signals(composite_scores: pd.DataFrame, universe_flags: pd.DataFrame, returns: pd.DataFrame) -> dict:

    try:
        from signals.signal_engine import run_signal_engine

        logger.info("Step 5: Generating signals...")

        signal_data = run_signal_engine(
            composite_scores=composite_scores,
            universe_flags=universe_flags,
            returns_matrix=returns
        )

        final = signal_data["final_signals"]
        today_signals = final.iloc[-1]

        logger.info(
            f"Step 5 complete: "
            f"Longs={( today_signals == 1).sum()} | "
            f"Shorts={(today_signals == -1).sum()}"
        )
        return signal_data

    except Exception as e:
        logger.error(f"Step 5 FAILED: {e}\n{traceback.format_exc()}")
        return None


def step_update_volatility(returns: pd.DataFrame, universe_flags: pd.DataFrame, current_universe: list) -> dict:

    try:
        from risk.volatility_model import run_volatility_model

        logger.info("Step 6: Updating volatility forecasts...")

        vol_data = run_volatility_model(
            returns_matrix=returns,
            universe_flags=universe_flags,
            current_universe=current_universe
        )

        logger.info(
            f"Step 6 complete: "
            f"Market regime={vol_data['index_result']['regime']}"
        )
        return vol_data

    except Exception as e:
        logger.error(f"Step 6 FAILED: {e}\n{traceback.format_exc()}")
        return None


def step_construct_portfolio(signals: pd.DataFrame, vol_matrix: pd.DataFrame) -> dict:

    try:
        from portfolio.portfolio_construction import (
            run_portfolio_construction
        )

        logger.info("Step 7: Constructing portfolio...")

        portfolio_data = run_portfolio_construction(
            signals_df=signals,
            vol_matrix=vol_matrix
        )

        today_weights = portfolio_data["weights"].iloc[-1]
        gross = today_weights.abs().sum()
        net   = today_weights.sum()

        logger.info(
            f"Step 7 complete: "
            f"Gross={gross:.4f} | Net={net:.4f}"
        )
        return portfolio_data

    except Exception as e:
        logger.error(f"Step 7 FAILED: {e}\n{traceback.format_exc()}")
        return None


def step_compute_risk(oos_returns: pd.Series, weights: pd.DataFrame, returns: pd.DataFrame) -> dict:
   
    try:
        from risk.risk_engine import run_risk_engine

        logger.info("Step 8: Computing risk metrics...")

        risk_data = run_risk_engine(
            oos_returns=oos_returns,
            weight_matrix=weights,
            returns_matrix=returns
        )

        logger.info(
            f"Step 8 complete: "
            f"VaR95={risk_data['risk_report']['hist_var_95']:.4f}"
        )
        return risk_data

    except Exception as e:
        logger.error(f"Step 8 FAILED: {e}\n{traceback.format_exc()}")
        return None
    
def save_pipeline_state(state: dict, path: str = "logs/pipeline_state.yaml"):

    import yaml

    state["last_run"] = datetime.now().isoformat()
    os.makedirs("logs", exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(state, f, default_flow_style=False)

    logger.info(f"Pipeline state saved → {path}")


def load_pipeline_state(path: str = "logs/pipeline_state.yaml") -> dict:

    import yaml

    if not os.path.exists(path):
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}

def step_run_backtest(weights: pd.DataFrame, returns: pd.DataFrame) -> dict:
    try:
        from backtest.backtest_engine import run_backtest
        logger.info("Step 8: Running backtest...")

        results = run_backtest(
            weight_matrix=weights,
            returns_matrix=returns,
            benchmark_returns=None
        )

        if results:
            logger.info(
                f"Step 8 complete: "
                f"CAGR={results['overall_metrics'].get('cagr',0):.2%} | "
                f"Sharpe={results['overall_metrics'].get('sharpe',0):.3f}"
            )
        return results

    except Exception as e:
        logger.error(f"Step 8 FAILED: {e}\n{traceback.format_exc()}")
        return {}
    
def run_daily_pipeline() -> dict:
    """
    Execute complete daily quantitative research pipeline.
    
    CRITICAL: This function orchestrates all pipeline steps with:
    - Reproducibility initialization
    - Step-by-step error handling
    - Fallback strategies for partial failures
    - Comprehensive pipeline state tracking
    - Validation at each stage
    
    Returns:
        dict with status, results, and step-by-step execution log
    """
    
    # ✓ CRITICAL: Initialize reproducibility at pipeline start
    initialize_reproducibility(seed=42)

    config  = load_config()
    results = {}
    step_statuses = {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY PIPELINE START: {run_date}")
    logger.info(f"{'='*60}")

    ok = step_update_data(config)
    step_statuses["data_update"] = "OK" if ok else "FAILED"

    close, returns = step_process_data(config)
    step_statuses["data_process"] = "OK" if close is not None else "FAILED"

    if close is None:
        logger.error("Cannot continue — data processing failed")
        save_pipeline_state({
            "status"      : "FAILED",
            "step_results": step_statuses,
            "error_step"  : "data_process"
        })
        return {"status": "FAILED", "step_results": step_statuses}

    results["close"]   = close
    results["returns"] = returns

    universe_data = step_build_universe(config, close)
    step_statuses["universe"] = "OK" if universe_data else "FAILED"

    if universe_data is None:
        save_pipeline_state({
            "status": "FAILED", "step_results": step_statuses
        })
        return {"status": "FAILED", "step_results": step_statuses}

    results["universe_data"]  = universe_data
    current_universe          = universe_data["current_universe"]
    universe_flags            = universe_data["universe_flags"]

    factor_data = step_compute_factors(returns, close, universe_flags)
    step_statuses["factors"] = "OK" if factor_data else "FAILED"

    if factor_data is None:
        save_pipeline_state({
            "status": "PARTIAL", "step_results": step_statuses
        })
        return {"status": "PARTIAL", "step_results": step_statuses}

    results["factor_data"]    = factor_data
    composite_scores          = factor_data["composite"]

    signal_data = step_generate_signals(
        composite_scores, universe_flags, returns
    )
    step_statuses["signals"] = "OK" if signal_data else "FAILED"

    if signal_data:
        results["signal_data"] = signal_data
        final_signals          = signal_data["final_signals"]
    else:
        fallback = "signals/final_signals.csv"
        if os.path.exists(fallback):
            final_signals = pd.read_csv(
                fallback, index_col=0, parse_dates=False
            )
            logger.warning("Using yesterday's signals as fallback")
        else:
            save_pipeline_state({
                "status": "PARTIAL", "step_results": step_statuses
            })
            return {"status": "PARTIAL", "step_results": step_statuses}

    vol_data = step_update_volatility(
        returns, universe_flags, current_universe
    )
    step_statuses["volatility"] = "OK" if vol_data else "FAILED"

    if vol_data:
        results["vol_data"] = vol_data
        vol_matrix          = vol_data["vol_matrix"]
        index_result        = vol_data["index_result"]
    else:
        fallback = "risk/rolling_vol_matrix.csv"
        if os.path.exists(fallback):
            vol_matrix = pd.read_csv(
                fallback, index_col=0, parse_dates=False
            )
            index_result = {"regime": "UNKNOWN", "forecast_vol": np.nan}
            logger.warning("Using saved vol matrix as fallback")
        else:
            vol_matrix   = pd.DataFrame()
            index_result = {"regime": "UNKNOWN", "forecast_vol": np.nan}

    portfolio_data = step_construct_portfolio(final_signals, vol_matrix)
    step_statuses["portfolio"] = "OK" if portfolio_data else "FAILED"

    if portfolio_data:
        results["portfolio_data"] = portfolio_data
        weights = portfolio_data["weights"]
    else:
        save_pipeline_state({
            "status": "PARTIAL", "step_results": step_statuses
        })
        return {"status": "PARTIAL", "step_results": step_statuses}
    
    backtest_results = step_run_backtest(weights, returns)
    step_statuses["backtest"] = "OK" if backtest_results else "FAILED"

    if backtest_results and "oos_returns" in backtest_results:
        risk_data = step_compute_risk(
            backtest_results["oos_returns"], weights, returns
        )
        step_statuses["risk"] = "OK" if risk_data else "FAILED"
    else:
        logger.warning("Skipping risk — no OOS returns from backtest")
        step_statuses["risk"] = "SKIPPED"

    oos_path = "backtest/oos_returns.csv"
    if os.path.exists(oos_path):
        oos_returns = pd.read_csv(
            oos_path, index_col=0, parse_dates=False
        ).squeeze()
        risk_data = step_compute_risk(oos_returns, weights, returns)
        step_statuses["risk"] = "OK" if risk_data else "FAILED"
        if risk_data:
            results["risk_data"] = risk_data
    else:
        step_statuses["risk"] = "SKIPPED"
        logger.warning("No OOS returns found — skipping risk step")

    all_ok = all(v == "OK" for v in step_statuses.values() if v != "SKIPPED")
    status = "COMPLETE" if all_ok else "PARTIAL"

    today_signals = final_signals.iloc[-1]

    pipeline_state = {
        "status"        : status,
        "run_date"      : run_date,
        "step_results"  : step_statuses,
        "universe_size" : len(current_universe),
        "n_longs"       : int((today_signals == 1).sum()),
        "n_shorts"      : int((today_signals == -1).sum()),
        "market_regime" : index_result.get("regime", "UNKNOWN"),
        "index_vol"     : float(index_result.get("forecast_vol", np.nan) or np.nan)
    }

    save_pipeline_state(pipeline_state)
    results["pipeline_state"] = pipeline_state

    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE {status}: {run_date}")
    for step, stat in step_statuses.items():
        logger.info(f"  {step:<20} → {stat}")
    logger.info(f"{'='*60}\n")

    ok = step_generate_report()
    step_statuses["reporting"] = "OK" if ok else "FAILED"

    logger.info(
        "Pipeline complete. Launch dashboard with: "
        "streamlit run dashboard/app.py"
    )
    # from dashboard.app import step_launch_dashboard
    # step_launch_dashboard()

    return results

def step_generate_report() -> bool:
    try:
        from reporting.report_generator import run_reporting
        logger.info("Step 9: Generating reports...")
        run_reporting()
        logger.info("Step 9 complete: Reports generated")
        return True
    except Exception as e:
        logger.error(f"Step 9 FAILED: {e}\n{traceback.format_exc()}")
        return False


def step_send_alerts(pipeline_state: dict,
        equity_curve: pd.Series,
        oos_returns: pd.Series,
        var_95: float,
        vol_forecasts: pd.Series,
        overall_metrics: dict,
        pdf_path: str = None) -> bool:

    try:
        from alerts.alert_engine import run_alerts
        logger.info("Step 10: Sending alerts...")

        run_alerts(
            pipeline_state=pipeline_state,
            equity_curve=equity_curve,
            oos_returns=oos_returns,
            var_95=var_95,
            vol_forecasts=vol_forecasts,
            overall_metrics=overall_metrics,
            pdf_path=pdf_path
        )

        logger.info("Step 10 complete: Alerts dispatched")
        return True

    except Exception as e:
        logger.error(f"Step 10 FAILED: {e}\n{traceback.format_exc()}")
        return False
    
