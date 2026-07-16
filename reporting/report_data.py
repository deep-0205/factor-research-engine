import pandas as pd
import numpy as np
import os
import yaml
from logger import get_logger

logger = get_logger("report_data")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_report_data() -> dict:

    data = {}

    def safe_read_csv(path, index_col=0, parse_dates=True):
        if not os.path.exists(path):
            logger.warning(f"Missing file: {path}")
            return pd.DataFrame()
        try:
            return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            return pd.DataFrame()

    def safe_read_series(path):
        df = safe_read_csv(path)
        if df.empty:
            return pd.Series(dtype=float)
        return df.iloc[:, 0]

    data["equity_curve"]    = safe_read_series("backtest/equity_curve.csv")
    data["oos_returns"]     = safe_read_series("backtest/oos_returns.csv")
    data["window_results"]  = safe_read_csv(
        "backtest/window_results.csv", index_col=None, parse_dates=False
    )
    data["drawdown_analysis"] = safe_read_csv(
        "backtest/drawdown_analysis.csv", index_col=None, parse_dates=False
    )
    data["overall_metrics"] = safe_read_series("backtest/overall_metrics.csv")

    data["var_95"]          = safe_read_series("risk/var_95.csv")
    data["cvar_95"]         = safe_read_series("risk/cvar_95.csv")
    data["rolling_risk"]    = safe_read_csv("risk/rolling_risk.csv")
    data["stress_tests"]    = safe_read_csv("risk/stress_tests.csv")
    data["tail_risk"]       = safe_read_series("risk/tail_risk.csv")
    data["risk_report"]     = safe_read_series("risk/risk_report.csv")

    data["final_signals"]   = safe_read_csv("signals/final_signals.csv")
    data["signal_stats"]    = safe_read_csv("signals/signal_stats.csv")
    data["turnover"]        = safe_read_series("signals/turnover.csv")

    data["weights"]         = safe_read_csv("portfolio/weights_volatility.csv")
    data["portfolio_stats"] = safe_read_csv("portfolio/portfolio_stats.csv")

    data["composite_scores"] = safe_read_csv("factors/composite_scores.csv")
    data["ic_series"]        = safe_read_series("factors/ic_series.csv")

    state_path = "logs/pipeline_state.yaml"
    if os.path.exists(state_path):
        import yaml as _yaml
        with open(state_path) as f:
            data["pipeline_state"] = _yaml.safe_load(f) or {}
    else:
        data["pipeline_state"] = {}

    logger.info("Report data loaded successfully")
    return data

