# setup_data.py

import yaml
from logger import get_logger

logger = get_logger("setup")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    logger.info("Step 1: Fetching Nifty 500 tickers...")
    from data.fetch_universe import get_nifty500_tickers
    tickers = get_nifty500_tickers()
    logger.info(f"Got {len(tickers)} tickers")

    logger.info("Step 2: Downloading OHLCV data (this takes 30-60 mins)...")
    from data.download_data import download_ohlcv
    download_ohlcv(
        tickers=tickers,
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
        raw_path=config["data"]["raw_path"]
    )

    logger.info("Step 3: Building processed matrices...")
    from data.processed.process_data import build_close_matrix, compute_returns
    close = build_close_matrix(
        raw_path=config["data"]["raw_path"],
        processed_path=config["data"]["processed_path"]
    )
    returns = compute_returns(close, config["data"]["processed_path"])

    logger.info(
        f"Setup complete: {close.shape[1]} stocks, {len(close)} days"
    )
    logger.info("Now run: python main.py --mode run")


if __name__ == "__main__":
    main()
