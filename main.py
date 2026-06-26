from os import close

import yaml
from factors.factor_engine import run_factor_engine
from logger import get_logger
from data.fetch_universe import get_nifty500_tickers
from data.download_data import download_ohlcv
from data.processed.process_data import build_close_matrix, compute_returns
from universe.universe import build_universe 
from factors.factor_engine import run_factor_engine 
from signals.signal_engine import run_signal_engine
from risk.volatility_model import run_volatility_model


logger = get_logger("main")

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    logger.info(f"Starting {config['project']['name']} v{config['project']['version']}")

    logger.info("Phase 2: Data Layer")

    tickers = get_nifty500_tickers()

    download_ohlcv(
        tickers=tickers,
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
        raw_path=config["data"]["raw_path"]
    )

    close = build_close_matrix(
        raw_path=config["data"]["raw_path"],
        processed_path=config["data"]["processed_path"]
    )

    returns = compute_returns(
        close,
        config["data"]["processed_path"]
    )
    logger.info(f"Data pipeline complete. Universe: {returns.shape[1]} stocks")

    logger.info("Phase 3: Universe Construction")
    universe_data = build_universe(
        close_matrix=close,
        raw_path=config["data"]["raw_path"]
    )

    current_universe = universe_data["current_universe"]
    universe_flags  = universe_data["universe_flags"]
    logger.info(f"Universe ready: {len(current_universe)} stocks")

    logger.info("Phase 4: Factor Engine")
    factor_data = run_factor_engine(
        close_matrix=close,
        returns_matrix=returns,
        universe_flags=universe_flags
    )
    composite_scores = factor_data["composite"]
    logger.info("Factor engine complete.")

    logger.info("Phase 5: Signal Engine")
    signal_data = run_signal_engine(
        composite_scores=composite_scores,
        universe_flags=universe_flags,
        returns_matrix=returns
    )
    final_signals = signal_data["final_signals"]
    logger.info("Signal engine complete.")

    logger.info("Phase 6: Volatility Modeling")
    vol_data = run_volatility_model(
        returns_matrix=returns,
        universe_flags=universe_flags,
        current_universe=current_universe
    )
    vol_scalars = vol_data["vol_scalars"]
    vol_matrix = vol_data["vol_matrix"]
    index_result = vol_data["index_result"]
    logger.info(
        f"Market regime: {index_result['regime']} | "
        f"Index vol: {index_result['forecast_vol']:.2%}"
    )


if __name__ == "__main__":
    main()



