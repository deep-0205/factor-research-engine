# from os import close
# import yaml
# import argparse
# from logger import get_logger
# from backtest.backtest_engine import run_backtest
# from factors.factor_engine import run_factor_engine
# from logger import get_logger
# from data.fetch_universe import get_nifty500_tickers
# from data.download_data import download_ohlcv
# from data.processed.process_data import build_close_matrix, compute_returns
# from universe.universe import build_universe 
# from factors.factor_engine import run_factor_engine 
# from signals.signal_engine import run_signal_engine
# from risk.volatility_model import run_volatility_model
# from portfolio.portfolio_construction import run_portfolio_construction
# from backtest.backtest_engine import run_backtest
# from risk.risk_engine import run_risk_engine

# logger = get_logger("main")

# def load_config(path="config.yaml"):
#     with open(path, "r") as file:
#         return yaml.safe_load(file)

# def main():
#     config = load_config()
#     logger.info(f"Starting {config['project']['name']} v{config['project']['version']}")

#     logger.info("Phase 2: Data Layer")

#     tickers = get_nifty500_tickers()

#     download_ohlcv(
#         tickers=tickers,
#         start=config["data"]["start_date"],
#         end=config["data"]["end_date"],
#         raw_path=config["data"]["raw_path"]
#     )

#     close = build_close_matrix(
#         raw_path=config["data"]["raw_path"],
#         processed_path=config["data"]["processed_path"]
#     )

#     returns = compute_returns(
#         close,
#         config["data"]["processed_path"]
#     )
#     logger.info(f"Data pipeline complete. Universe: {returns.shape[1]} stocks")

#     logger.info("Phase 3: Universe Construction")
#     universe_data = build_universe(
#         close_matrix=close,
#         raw_path=config["data"]["raw_path"]
#     )

#     current_universe = universe_data["current_universe"]
#     universe_flags  = universe_data["universe_flags"]
#     logger.info(f"Universe ready: {len(current_universe)} stocks")

#     logger.info("Phase 4: Factor Engine")
#     factor_data = run_factor_engine(
#         close_matrix=close,
#         returns_matrix=returns,
#         universe_flags=universe_flags
#     )
#     composite_scores = factor_data["composite"]
#     logger.info("Factor engine complete.")

#     logger.info("Phase 5: Signal Engine")
#     signal_data = run_signal_engine(
#         composite_scores=composite_scores,
#         universe_flags=universe_flags,
#         returns_matrix=returns
#     )
#     final_signals = signal_data["final_signals"]
#     logger.info("Signal engine complete.")

#     logger.info("Phase 6: Volatility Modeling")
#     vol_data = run_volatility_model(
#         returns_matrix=returns,
#         universe_flags=universe_flags,
#         current_universe=current_universe
#     )
#     vol_scalars = vol_data["vol_scalars"]
#     vol_matrix = vol_data["vol_matrix"]
#     index_result = vol_data["index_result"]
#     logger.info(
#         f"Market regime: {index_result['regime']} | "
#         f"Index vol: {index_result['forecast_vol']:.2%}"
#     )

#     logger.info("Phase 7: Portfolio Construction")
#     portfolio_data = run_portfolio_construction(
#         signals_df=final_signals,
#         vol_matrix=vol_matrix,
#         sector_map=None       
#     )
#     weights = portfolio_data["weights"]
#     logger.info("Portfolio construction complete.")

    
#     logger.info("Phase 8: Walk-Forward Backtesting")
#     backtest_results = run_backtest(
#         weight_matrix=weights,
#         returns_matrix=returns,
#         benchmark_returns=None    
#     )

#     overall_metrics = backtest_results["overall_metrics"]
#     logger.info(
#         f"Backtest complete | "
#         f"CAGR: {overall_metrics['cagr']:.2%} | "
#         f"Sharpe: {overall_metrics['sharpe']:.3f}"
#     )

#     logger.info("Phase 9: Risk Engine")
#     risk_data = run_risk_engine(
#         oos_returns=backtest_results["oos_returns"],
#         weight_matrix=weights,
#         returns_matrix=returns,
#         factor_returns=None
#     )
#     logger.info(
#         f"Risk complete | "
#         f"VaR 95%: {risk_data['risk_report']['hist_var_95']:.4f} | "
#         f"CVaR 95%: {risk_data['risk_report']['cvar_95']:.4f}"
#     )

# if __name__ == "__main__":
#     main()


# main.py

import yaml
import argparse
from logger import get_logger

logger = get_logger("main")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_full_pipeline():
    """Runs the complete pipeline once (used for initial setup)."""
    from scheduler.pipeline_runner import run_daily_pipeline
    return run_daily_pipeline()


def run_scheduler():
    """Starts the scheduler for automated daily runs."""
    from scheduler.scheduler import start_scheduler
    start_scheduler()


def main():
    parser = argparse.ArgumentParser(
        description="Factor Research Engine"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "schedule", "backtest"],
        default="run",
        help=(
            "run=execute pipeline once | "
            "schedule=start daily scheduler | "
            "backtest=run full historical backtest"
        )
    )
    args = parser.parse_args()

    config = load_config()
    logger.info(
        f"Starting {config['project']['name']} | Mode: {args.mode}"
    )

    if args.mode == "run":
        results = run_full_pipeline()
        logger.info(
            f"Pipeline status: {results.get('status', 'UNKNOWN')}"
        )

    elif args.mode == "schedule":
        logger.info("Starting automated scheduler...")
        run_scheduler()

    elif args.mode == "backtest":
        logger.info("Running full historical backtest...")
        run_full_pipeline()

def step_launch_dashboard() -> None:
    logger.info(
        "Dashboard available — run: "
        "streamlit run dashboard/app.py"
    )

if __name__ == "__main__":
    main()
