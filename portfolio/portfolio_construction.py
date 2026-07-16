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

    weights = pd.Series(0.0, index=signals.index)

    long_mask  = signals ==  1
    short_mask = signals == -1

    n_long  = long_mask.sum()
    n_short = short_mask.sum()

    if n_long > 0:
        weights[long_mask]  =  1.0 / n_long

    if n_short > 0:
        weights[short_mask] = -1.0 / n_short

    return weights

def volatility_weight(signals: pd.Series, vol_forecasts: pd.Series) -> pd.Series:

    weights = pd.Series(0.0, index=signals.index)

    active = signals[signals != 0]
    if active.empty:
        return weights

    vols = vol_forecasts.reindex(active.index).fillna(vol_forecasts.median())

    vols = vols.replace(0, vols.median())

    raw = active / vols

    long_raw  = raw[raw > 0]
    short_raw = raw[raw < 0]

    if not long_raw.empty:
        weights[long_raw.index]  = long_raw  / long_raw.sum()

    if not short_raw.empty:
        weights[short_raw.index] = short_raw / short_raw.abs().sum() * -1

    return weights

def risk_parity_weight(signals: pd.Series, vol_forecasts: pd.Series, corr_matrix: pd.DataFrame = None) -> pd.Series:

    weights = pd.Series(0.0, index=signals.index)

    active = signals[signals != 0]
    if active.empty:
        return weights

    vols = vol_forecasts.reindex(active.index).fillna(vol_forecasts.median()).replace(0, vol_forecasts.median())

    inv_vol = 1.0 / vols

    long_mask  = active > 0
    short_mask = active < 0

    long_inv  = inv_vol[long_mask]
    short_inv = inv_vol[short_mask]

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

    w = weights.copy()

    w = w.clip(lower=-max_single_position, upper=max_single_position)

    gross = w.abs().sum()
    if gross > max_gross:
        w = w * (max_gross / gross)

    net = w.sum()
    if abs(net) > max_net:
        if net > max_net:
            excess = net - max_net
            long_sum = w[w > 0].sum()
            if long_sum > 0:
                w[w > 0] *= (1 - excess / long_sum)
        else:
            excess = abs(net) - max_net
            short_sum = w[w < 0].abs().sum()
            if short_sum > 0:
                w[w < 0] *= (1 - excess / short_sum)

    logger.debug(
        f"Exposure | Gross: {w.abs().sum():.4f} | "
        f"Net: {w.sum():.4f} | "
        f"Max pos: {w.abs().max():.4f}"
    )
    return w

def apply_sector_neutrality(weights: pd.Series, sector_map: dict, max_sector_net: float = 0.10) -> pd.Series:

    w = weights.copy()

    sector_to_tickers = {}
    for ticker, sector in sector_map.items():
        if ticker not in w.index:
            continue
        sector_to_tickers.setdefault(sector, []).append(ticker)

    for sector, tickers in sector_to_tickers.items():
        sector_weights = w[tickers]
        sector_net     = sector_weights.sum()

        if abs(sector_net) <= max_sector_net:
            continue  

        if sector_net > max_sector_net:
            excess = sector_net - max_sector_net
            long_tickers = [t for t in tickers if w[t] > 0]
            long_sum = w[long_tickers].sum()
            if long_sum > 0:
                for t in long_tickers:
                    w[t] *= (1 - excess / long_sum)
        else:
            excess = abs(sector_net) - max_sector_net
            short_tickers = [t for t in tickers if w[t] < 0]
            short_sum = w[short_tickers].abs().sum()
            if short_sum > 0:
                for t in short_tickers:
                    w[t] *= (1 - excess / short_sum)

        logger.debug(
            f"Sector {sector}: net {sector_net:.4f} → "
            f"{w[tickers].sum():.4f}"
        )

    return w

def build_weight_matrix(
        signals_df: pd.DataFrame,
        vol_matrix: pd.DataFrame,
        method: str = "volatility",
        sector_map: dict = None) -> pd.DataFrame:

    config = load_config()
    weight_matrix = pd.DataFrame(0.0, index=signals_df.index, columns=signals_df.columns)

    for date in signals_df.index:
        signals = signals_df.loc[date]

        if (signals != 0).sum() == 0:
            continue

        if date in vol_matrix.index:
            vol_forecasts = vol_matrix.loc[date]
        else:
            vol_forecasts = pd.Series(0.20, index=signals.index)

        if method == "equal":
            weights = equal_weight(signals)

        elif method == "volatility":
            weights = volatility_weight(signals, vol_forecasts)

        elif method == "risk_parity":
            weights = risk_parity_weight(signals, vol_forecasts)

        else:
            raise ValueError(f"Unknown method: {method}")

        weights = apply_exposure_controls(
            weights,
            max_gross=config["portfolio"]["max_gross_exposure"],
            max_net=config["portfolio"]["max_net_exposure"]
        )

        if sector_map:
            weights = apply_sector_neutrality(weights, sector_map)

        weight_matrix.loc[date] = weights

    logger.info(
        f"Weight matrix built | Method: {method} | "
        f"Shape: {weight_matrix.shape} | "
        f"Avg gross: {weight_matrix.abs().sum(axis=1).mean():.4f}"
    )
    return weight_matrix

def compute_portfolio_stats(weight_matrix: pd.DataFrame) -> pd.DataFrame:

    stats = pd.DataFrame(index=weight_matrix.index)

    stats["gross_exposure"] = weight_matrix.abs().sum(axis=1)
    stats["net_exposure"]   = weight_matrix.sum(axis=1)
    stats["n_long"]         = (weight_matrix > 0).sum(axis=1)
    stats["n_short"]        = (weight_matrix < 0).sum(axis=1)
    stats["max_position"]   = weight_matrix.abs().max(axis=1)

    long_weights = weight_matrix.clip(lower=0)
    long_sum     = long_weights.sum(axis=1).replace(0, np.nan)
    normalized   = long_weights.div(long_sum, axis=0)
    stats["concentration"] = (normalized ** 2).sum(axis=1)

    logger.info(
        f"Portfolio Stats | "
        f"Avg gross: {stats['gross_exposure'].mean():.4f} | "
        f"Avg net: {stats['net_exposure'].mean():.4f} | "
        f"Avg longs: {stats['n_long'].mean():.1f} | "
        f"Avg shorts: {stats['n_short'].mean():.1f}"
    )
    return stats

def run_portfolio_construction(
        signals_df: pd.DataFrame,
        vol_matrix: pd.DataFrame,
        sector_map: dict = None,
        save_path: str = "portfolio/") -> dict:
 
    os.makedirs(save_path, exist_ok=True)

    methods = ["equal", "volatility", "risk_parity"]
    weight_matrices = {}

    for method in methods:
        logger.info(f"Building {method} weight portfolio...")
        w = build_weight_matrix(
            signals_df=signals_df,
            vol_matrix=vol_matrix,
            method=method,
            sector_map=sector_map
        )
        weight_matrices[method] = w
        w.to_csv(os.path.join(save_path, f"weights_{method}.csv"))

    primary_weights = weight_matrices["volatility"]

    stats = compute_portfolio_stats(primary_weights)
    stats.to_csv(os.path.join(save_path, "portfolio_stats.csv"))

    logger.info("Portfolio construction complete. All outputs saved.")

    return {
        "weights"         : primary_weights,
        "weights_equal"   : weight_matrices["equal"],
        "weights_rp"      : weight_matrices["risk_parity"],
        "portfolio_stats" : stats
    }

