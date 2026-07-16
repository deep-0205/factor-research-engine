import pandas as pd
import numpy as np
import os
import yaml
from logger import get_logger

logger = get_logger("backtest_engine")

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)
    
def generate_walk_forward_windows(index: pd.DatetimeIndex, train_months: int = 12, test_months: int = 1) -> list:
    windows = []

    index = pd.DatetimeIndex(index)
    dates = pd.Series(index, index=index)

    monthly_groups = dates.groupby(
        [index.year, index.month]
    ).first()

    months = [pd.Timestamp(d) for d in monthly_groups.values]
    n      = len(months)

    for i in range(train_months, n - test_months + 1):
        train_start = months[i - train_months]
        train_end   = months[i - 1]
        test_start  = months[i]

        test_end_idx    = min(i + test_months - 1, n - 1)
        test_end        = months[test_end_idx]

        test_end_actual = dates[
            (dates >= test_start) & (dates <= test_end)
        ].max()

        windows.append((train_start, train_end, test_start, pd.Timestamp(test_end_actual)))

    logger.info(
        f"Walk-forward windows: {len(windows)} total | "
        f"Train: {train_months}mo | Test: {test_months}mo"
    )
    
    logger.info(
        f"{test_start} -> {test_end}"
    )
    logger.info(
        f"Rows in window: {len(w_test)}"
    )
    return windows


def compute_portfolio_returns(
        weight_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5) -> pd.Series:

    common_dates   = weight_matrix.index.intersection(returns_matrix.index)
    common_tickers = weight_matrix.columns.intersection(returns_matrix.columns)

    w = weight_matrix.loc[common_dates, common_tickers]
    r = returns_matrix.loc[common_dates, common_tickers]

    gross_returns = (w.shift(1) * r).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1)

    total_cost_bps = transaction_cost_bps + slippage_bps
    costs = turnover * (total_cost_bps / 10000)

    net_returns = gross_returns - costs

    logger.info(
        f"Portfolio Returns | "
        f"Gross mean: {gross_returns.mean()*252:.4f} | "
        f"Net mean: {net_returns.mean()*252:.4f} | "
        f"Avg daily cost: {costs.mean()*10000:.2f}bps"
    )
    return net_returns

def compute_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, risk_free_rate: float = 0.065) -> dict:

    daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    total_return = (1 + returns).prod() - 1
    n_days       = len(returns)
    cagr         = (1 + total_return) ** (252 / n_days) - 1

    annual_vol = returns.std() * np.sqrt(252)
    downside   = returns[returns < daily_rf]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else annual_vol

    excess_return = cagr - risk_free_rate
    sharpe        = excess_return / annual_vol if annual_vol > 0 else 0
    sortino       = excess_return / downside_vol if downside_vol > 0 else 0

    equity_curve = (1 + returns).cumprod()
    rolling_max  = equity_curve.cummax()
    drawdown     = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar       = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    wins  = returns[returns > 0]
    loses = returns[returns < 0]
    win_rate      = len(wins) / len(returns) if len(returns) > 0 else 0
    profit_factor = wins.sum() / abs(loses.sum()) if abs(loses.sum()) > 0 else 0

    skewness = returns.skew()
    kurtosis = returns.kurt()

    metrics = {
        "cagr"          : round(cagr, 6),
        "annual_vol"    : round(annual_vol, 6),
        "sharpe"        : round(sharpe, 4),
        "sortino"       : round(sortino, 4),
        "max_drawdown"  : round(max_drawdown, 6),
        "calmar"        : round(calmar, 4),
        "win_rate"      : round(win_rate, 4),
        "profit_factor" : round(profit_factor, 4),
        "total_return"  : round(total_return, 6),
        "skewness"      : round(skewness, 4),
        "kurtosis"      : round(kurtosis, 4),
        "n_days"        : n_days
    }

    if benchmark_returns is not None:
        common = returns.index.intersection(benchmark_returns.index)
        p = returns[common]
        b = benchmark_returns[common]

        cov_matrix = np.cov(p, b)
        beta  = cov_matrix[0, 1] / cov_matrix[1, 1] \
                if cov_matrix[1, 1] > 0 else 0
        b_cagr = (1 + b).prod() ** (252 / len(b)) - 1
        alpha  = cagr - (risk_free_rate + beta * (b_cagr - risk_free_rate))

        metrics["beta"]  = round(beta, 4)
        metrics["alpha"] = round(alpha, 6)

    logger.info(
        f"Performance | CAGR: {cagr:.2%} | Sharpe: {sharpe:.3f} | "
        f"MaxDD: {max_drawdown:.2%} | Sortino: {sortino:.3f}"
    )
    return metrics

def run_walk_forward_backtest(weight_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        train_months: int = 12,
        test_months: int = 1,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5,
        benchmark_returns: pd.Series = None) -> dict:

    windows = generate_walk_forward_windows(weight_matrix.index, train_months, test_months)

    window_results = []
    all_oos_returns = []

    logger.info(f"Starting walk-forward backtest: {len(windows)} windows")

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):

        test_mask_w = (weight_matrix.index >= test_start) & (weight_matrix.index <= test_end)
        test_mask_r = (returns_matrix.index >= test_start) & (returns_matrix.index <= test_end)

        w_test = weight_matrix[test_mask_w]
        r_test = returns_matrix[test_mask_r]

        if w_test.empty or r_test.empty:
            continue

        period_returns = compute_portfolio_returns(
            weight_matrix=w_test,
            returns_matrix=r_test,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps
        )

        bench_window = None
        if benchmark_returns is not None:
            bench_window = benchmark_returns[
                (benchmark_returns.index >= test_start) &
                (benchmark_returns.index <= test_end)
            ]

        metrics = compute_performance_metrics(
            period_returns, bench_window
        )
        metrics["test_start"] = test_start
        metrics["test_end"]   = test_end

        window_results.append(metrics)
        all_oos_returns.append(period_returns)

        test_start_str = pd.Timestamp(test_start).strftime('%Y-%m')
        logger.info(
            f"Window {i+1}/{len(windows)} | "
            f"{test_start_str} | "
            f"Sharpe: {metrics['sharpe']:.3f} | "
            f"Return: {metrics['cagr']:.2%}"
        )

    if not all_oos_returns:
        logger.error("No OOS returns generated")
        return {}

    oos_returns   = pd.concat(all_oos_returns).sort_index()
    equity_curve  = (1 + oos_returns).cumprod()
    window_df     = pd.DataFrame(window_results)

    overall_metrics = compute_performance_metrics(
        oos_returns, benchmark_returns
    )

    logger.info(
        f"\n{'='*50}\n"
        f"WALK-FORWARD BACKTEST COMPLETE\n"
        f"  CAGR         : {overall_metrics['cagr']:.2%}\n"
        f"  Sharpe Ratio : {overall_metrics['sharpe']:.3f}\n"
        f"  Sortino      : {overall_metrics['sortino']:.3f}\n"
        f"  Max Drawdown : {overall_metrics['max_drawdown']:.2%}\n"
        f"  Calmar Ratio : {overall_metrics['calmar']:.3f}\n"
        f"  Win Rate     : {overall_metrics['win_rate']:.2%}\n"
        f"{'='*50}"
    )

    return {
        "oos_returns"    : oos_returns,
        "equity_curve"   : equity_curve,
        "window_results" : window_df,
        "overall_metrics": overall_metrics
    }

def compute_drawdown_analysis(equity_curve: pd.Series) -> pd.DataFrame:
    
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max

    drawdowns = []
    in_drawdown = False
    start = None
    trough_date = None
    trough_val  = 0

    for date, dd in drawdown.items():
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start       = date
            trough_date = date
            trough_val  = dd

        elif dd < trough_val and in_drawdown:
            trough_date = date
            trough_val  = dd

        elif dd == 0 and in_drawdown:
            drawdowns.append({
                "start"    : start,
                "trough"   : trough_date,
                "end"      : date,
                "depth"    : trough_val,
                "duration" : (date - start).days,
                "recovery" : (date - trough_date).days
            })
            in_drawdown = False
            trough_val  = 0

    dd_df = pd.DataFrame(drawdowns)
    if not dd_df.empty:
        dd_df = dd_df.sort_values("depth").reset_index(drop=True)
        logger.info(
            f"Drawdown analysis | "
            f"Total episodes: {len(dd_df)} | "
            f"Worst: {dd_df['depth'].min():.2%} | "
            f"Avg duration: {dd_df['duration'].mean():.0f} days"
        )
    return dd_df

def run_backtest(weight_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        benchmark_returns: pd.Series = None,
        save_path: str = "backtest/") -> dict:

    config = load_config()
    os.makedirs(save_path, exist_ok=True)

    logger.info("Running walk-forward backtest...")
    results = run_walk_forward_backtest(
        weight_matrix=weight_matrix,
        returns_matrix=returns_matrix,
        train_months=config["backtest"]["train_months"],
        test_months=config["backtest"]["test_months"],
        transaction_cost_bps=config["backtest"]["transaction_cost_bps"],
        slippage_bps=config["backtest"]["slippage_bps"],
        benchmark_returns=benchmark_returns
    )

    if not results:
        logger.error("Backtest returned no results")
        return {}

    dd_analysis = compute_drawdown_analysis(results["equity_curve"])

    results["oos_returns"].to_csv(
        os.path.join(save_path, "oos_returns.csv")
    )
    results["equity_curve"].to_csv(
        os.path.join(save_path, "equity_curve.csv")
    )
    results["window_results"].to_csv(
        os.path.join(save_path, "window_results.csv"), index=False
    )
    dd_analysis.to_csv(
        os.path.join(save_path, "drawdown_analysis.csv"), index=False
    )
    pd.Series(results["overall_metrics"]).to_csv(
        os.path.join(save_path, "overall_metrics.csv")
    )

    logger.info("Backtest complete. All outputs saved to backtest/")

    return {**results, "drawdown_analysis": dd_analysis}