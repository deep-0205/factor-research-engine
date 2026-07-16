import pandas as pd
import numpy as np
import os
import yaml
from logger import get_logger

logger = get_logger("universe")

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def compute_average_volume(raw_path: str, tickers: list, lookback: int = 60) -> pd.Series:
    avg_volumes = {}

    for ticker in tickers:
        path = os.path.join(raw_path, f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, header=0)

            if df.iloc[0].astype(str).str.contains(
                ticker.split(".")[0], case=False, na=False
            ).any():
                df = pd.read_csv(path, header=0, skiprows=[1])

            df.columns = [str(c).strip().lower() for c in df.columns]

            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
            df = df[~df.index.isna()]

            if "volume" not in df.columns:
                continue

            vol = pd.to_numeric(df["volume"], errors="coerce")
            avg_vol = vol.iloc[-lookback:].mean()

            if pd.notna(avg_vol):
                avg_volumes[ticker] = avg_vol

        except Exception as e:
            logger.warning(f"Volume read failed for {ticker}: {e}")

    return pd.Series(avg_volumes)

def apply_liquidity_filter(avg_volumes: pd.Series, min_avg_volume: int) -> list:
    liquid = avg_volumes[avg_volumes >= min_avg_volume].index.tolist()
    logger.info(
        f"Liquidity filter: {len(avg_volumes)} → {len(liquid)} stocks "
        f"(min volume: {min_avg_volume:,})"
    )
    return liquid

def apply_history_filter(close_matrix: pd.DataFrame, min_history_days: int = 252) -> list:
    valid = close_matrix.columns[
        close_matrix.notna().sum() >= min_history_days
    ].tolist()
    logger.info(f"History filter: {len(close_matrix.columns)} → {len(valid)} stocks")
    return valid

def apply_price_filter(close_matrix: pd.DataFrame, min_price: float = 10.0) -> list:

    if not isinstance(close_matrix.index, pd.DatetimeIndex):
        close_matrix.index = pd.to_datetime(
            close_matrix.index, errors="coerce"
        )
        close_matrix = close_matrix[close_matrix.index.notna()]

    latest_prices = close_matrix.iloc[-1]

    latest_prices = pd.to_numeric(latest_prices, errors="coerce")

    valid = latest_prices[latest_prices >= min_price].index.tolist()

    logger.info(
        f"Price filter: {len(close_matrix.columns)} → "
        f"{len(valid)} stocks (min price: ₹{min_price})"
    )
    return valid

def build_historical_universe(
    close_matrix: pd.DataFrame,
    lookback_days: int = 252,
    min_price: float = 10.0,
    min_data_pct: float = 0.80,) -> pd.DataFrame:
    universe_flags = pd.DataFrame(False, index=close_matrix.index, columns=close_matrix.columns)

    for i, date in enumerate(close_matrix.index):
        if i < lookback_days:
            continue
        window = close_matrix.iloc[i - lookback_days : i]
        current_prices = close_matrix.iloc[i]
        price_ok = current_prices >= min_price
        data_ok = window.notna().mean() >= min_data_pct
        universe_flags.loc[date] = price_ok & data_ok

    logger.info(
        f"Historical universe built: "
        f"avg {universe_flags.sum(axis=1).mean():.0f} stocks/day"
    )
    return universe_flags

def build_universe(close_matrix: pd.DataFrame, raw_path: str, save_path: str = "universe/") -> dict:
    config = load_config()
    os.makedirs(save_path, exist_ok=True)

    all_tickers = close_matrix.columns.tolist()

    history_ok = apply_history_filter(close_matrix)
    price_ok = apply_price_filter(close_matrix)
    avg_volumes = compute_average_volume(raw_path, all_tickers)
    liquidity_ok = apply_liquidity_filter(
        avg_volumes, min_avg_volume=config["universe"]["min_avg_volume"]
    )

    current_universe = list(set(history_ok) & set(price_ok) & set(liquidity_ok))
    logger.info(f"Final current universe: {len(current_universe)} stocks")

    universe_flags = build_historical_universe(close_matrix)

    pd.Series(current_universe).to_csv(os.path.join(save_path, "current_universe.csv"), index=False)
    universe_flags.to_csv(os.path.join(save_path, "universe_flags.csv"))

    return {"current_universe": current_universe, "universe_flags": universe_flags}
