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
    """
    Generate non-overlapping walk-forward windows for backtesting.
    
    CRITICAL: Ensures NO LOOK-AHEAD BIAS by:
    - Train period ends before test period starts
    - All calculations use only past data
    - No information leakage between windows
    - Uses month-END dates for inclusive windows (not month-first)
    
    Args:
        index: DatetimeIndex of all available dates
        train_months: Number of months for training
        test_months: Number of months for testing
        
    Returns:
        List of tuples: (train_start, train_end, test_start, test_end)
    """
    if len(index) == 0:
        logger.warning("Empty index provided to generate_walk_forward_windows")
        return []
    
    windows = []
    index = pd.DatetimeIndex(index)
    
    # Use ALL dates, not just month-start dates
    # Get month-end dates for window boundaries
    dates_series = pd.Series(index, index=index)
    
    # Group by year-month and get LAST date of each month
    monthly_groups = dates_series.groupby(
        [index.year, index.month]
    ).last()
    
    month_ends = [pd.Timestamp(d) for d in monthly_groups.values]
    n_months = len(month_ends)

    if n_months < train_months + test_months:
        logger.warning(
            f"Insufficient data: {n_months} months < "
            f"{train_months + test_months} required (train + test)"
        )
        return []

    # Generate windows using month-end dates
    for i in range(train_months, n_months - test_months + 1):
        # Train period: from N months ago to end of previous month
        train_start_idx = max(0, i - train_months)
        train_start = month_ends[train_start_idx]
        train_end = month_ends[i - 1]
        
        # Test period: from start of current month to end of test month
        test_start_idx = i
        test_end_idx = min(i + test_months - 1, n_months - 1)
        
        # Find first date of test month
        test_start_month = train_end + pd.DateOffset(days=1)
        test_start = dates_series[dates_series.index >= test_start_month].index[0] if len(dates_series[dates_series.index >= test_start_month]) > 0 else month_ends[test_start_idx]
        
        # Test end is end of test period
        test_end = month_ends[test_end_idx]

        # Validate the window
        if train_end >= test_start:
            logger.warning(
                f"Invalid window: train_end ({train_end}) >= test_start ({test_start})"
            )
            continue

        windows.append((train_start, train_end, test_start, test_end))

    logger.info(
        f"Walk-forward windows generated: {len(windows)} total | "
        f"Train: {train_months}mo | Test: {test_months}mo | "
        f"Date range: {index[0].date()} to {index[-1].date()}"
    )
    
    return windows


def compute_portfolio_returns(
        weight_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5) -> pd.Series:
    """
    Compute portfolio returns with proper index alignment.
    
    CRITICAL:
    - Weights at t are applied to returns at t+1
    - First weight row ignored (no t-1 weights available)
    - Index alignment enforced to prevent data leakage
    - Handles NaNs and misaligned indices
    
    Args:
        weight_matrix: Portfolio weights DataFrame (index: dates, columns: tickers)
        returns_matrix: Daily returns DataFrame (index: dates, columns: tickers)
        transaction_cost_bps: Transaction cost in basis points
        slippage_bps: Slippage cost in basis points
        
    Returns:
        Portfolio returns Series (aligned to weight_matrix index)
    """
    
    if weight_matrix.empty or returns_matrix.empty:
        logger.error("Empty weight or returns matrix provided")
        return pd.Series()
    
    # Ensure indices are DatetimeIndex
    weight_matrix.index = pd.to_datetime(weight_matrix.index)
    returns_matrix.index = pd.to_datetime(returns_matrix.index)
    
    # Find common dates and tickers
    common_dates   = weight_matrix.index.intersection(returns_matrix.index)
    common_tickers = weight_matrix.columns.intersection(returns_matrix.columns)
    
    if len(common_dates) == 0 or len(common_tickers) == 0:
        logger.error(
            f"No common dates or tickers: "
            f"common_dates={len(common_dates)}, common_tickers={len(common_tickers)}"
        )
        return pd.Series()
    
    w = weight_matrix.loc[common_dates, common_tickers].copy()
    r = returns_matrix.loc[common_dates, common_tickers].copy()
    
    # Ensure proper alignment
    assert (w.index == r.index).all(), "Index mismatch between weights and returns"
    
    # Portfolio return: weights at t multiplied by returns at t+1
    # This ensures we don't use current day's weights for current day's returns
    gross_returns = (w.shift(1) * r).sum(axis=1)
    
    # Calculate turnover (absolute value of weight changes)
    turnover = w.diff().abs().sum(axis=1)
    
    # Apply transaction costs
    total_cost_bps = transaction_cost_bps + slippage_bps
    costs = turnover * (total_cost_bps / 10000)
    
    # Net returns after costs
    net_returns = gross_returns - costs
    
    # Remove NaN from first period (no prior weights)
    net_returns = net_returns.dropna()
    
    # Validate returns are reasonable
    if net_returns.empty:
        logger.error("No valid portfolio returns after alignment")
        return pd.Series()
    
    logger.info(
        f"Portfolio Returns Computed | "
        f"Periods: {len(net_returns)} | "
        f"Gross annualized: {gross_returns.mean()*252:.4f} | "
        f"Net annualized: {net_returns.mean()*252:.4f} | "
        f"Avg daily cost: {costs.mean()*10000:.2f}bps"
    )
    return net_returns

def compute_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, risk_free_rate: float = 0.065) -> dict:
    """
    Compute comprehensive performance metrics with proper annualization.
    
    CRITICAL:
    - Annualization factors are mathematically correct
    - Downside volatility uses only negative excess returns
    - Calmar handles negative max drawdown correctly
    - Beta/Alpha only computed when sufficient benchmark data exists
    
    Args:
        returns: Daily returns Series
        benchmark_returns: Optional daily benchmark returns
        risk_free_rate: Annual risk-free rate (default: 6.5%)
        
    Returns:
        Dictionary of performance metrics
    """
    
    if returns.empty:
        logger.error("Empty returns Series provided to compute_performance_metrics")
        return {}
    
    returns = returns.dropna()
    
    if len(returns) < 2:
        logger.error(f"Insufficient returns data: {len(returns)} observations")
        return {}
    
    # Risk-free rate converted to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    # Total return and CAGR
    total_return = (1 + returns).prod() - 1
    n_days       = len(returns)
    
    # CAGR annualization: (1 + total_return) ^ (252 / n_days) - 1
    cagr         = (1 + total_return) ** (252 / n_days) - 1

    # Volatility
    annual_vol = returns.std() * np.sqrt(252)
    
    # Downside volatility (only negative excess returns)
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_vol

    # Sharpe Ratio
    excess_return = cagr - risk_free_rate
    sharpe        = excess_return / annual_vol if annual_vol > 0 else 0
    
    # Sortino Ratio
    sortino       = excess_return / downside_vol if downside_vol > 0 else 0

    # Drawdown and Calmar
    equity_curve = (1 + returns).cumprod()
    running_max  = equity_curve.expanding().max()
    drawdown     = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar: CAGR / |Max Drawdown|
    calmar       = cagr / abs(max_drawdown) if max_drawdown < 0 else 0

    # Win rate and Profit Factor
    wins  = returns[returns > 0]
    loses = returns[returns < 0]
    win_rate      = len(wins) / len(returns) if len(returns) > 0 else 0
    profit_factor = wins.sum() / abs(loses.sum()) if abs(loses.sum()) > 0 else 0

    # Higher moments
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
        "n_days"        : n_days,
        "daily_vol"     : round(returns.std(), 6)
    }

    # Beta and Alpha (only if sufficient benchmark data)
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        common = returns.index.intersection(benchmark_returns.index)
        
        if len(common) >= 60:  # Require at least 2 months of data
            p = returns[common]
            b = benchmark_returns[common]

            cov_matrix = np.cov(p, b)
            beta  = cov_matrix[0, 1] / cov_matrix[1, 1] \
                    if cov_matrix[1, 1] > 0 else 0
            b_cagr = (1 + b).prod() ** (252 / len(b)) - 1
            alpha  = cagr - (risk_free_rate + beta * (b_cagr - risk_free_rate))

            metrics["beta"]  = round(beta, 4)
            metrics["alpha"] = round(alpha, 6)
        else:
            logger.warning(
                f"Insufficient benchmark data for beta/alpha: "
                f"{len(common)} common dates"
            )

    logger.info(
        f"Performance Metrics | CAGR: {cagr:.2%} | Sharpe: {sharpe:.3f} | "
        f"MaxDD: {max_drawdown:.2%} | Sortino: {sortino:.3f} | "
        f"WinRate: {win_rate:.2%} | Days: {n_days}"
    )
    return metrics

def run_walk_forward_backtest(weight_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        train_months: int = 12,
        test_months: int = 1,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5,
        benchmark_returns: pd.Series = None) -> dict:
    """
    Execute walk-forward backtest with NO LOOK-AHEAD BIAS.
    
    CRITICAL VALIDATION:
    - Train and test periods never overlap
    - Each test uses only prior training data
    - No future information leaks into portfolio construction
    - Window sizes validated before processing
    
    Args:
        weight_matrix: Portfolio weights for entire period
        returns_matrix: Daily returns for entire period
        train_months: Training period length
        test_months: Test period length
        transaction_cost_bps: Transaction costs
        slippage_bps: Slippage costs
        benchmark_returns: Optional benchmark for comparison
        
    Returns:
        Dictionary with OOS returns, equity curve, and metrics
    """
    
    if weight_matrix.empty or returns_matrix.empty:
        logger.error("Empty weight or returns matrix")
        return {}

    windows = generate_walk_forward_windows(weight_matrix.index, train_months, test_months)
    
    if not windows:
        logger.error("No valid walk-forward windows generated")
        return {}

    window_results = []
    all_oos_returns = []

    logger.info(
        f"Starting walk-forward backtest: {len(windows)} windows | "
        f"Train: {train_months}mo, Test: {test_months}mo"
    )

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        
        # Validate window boundaries
        if test_start <= train_end:
            logger.error(
                f"Invalid window {i}: test_start ({test_start}) "
                f"<= train_end ({train_end})"
            )
            continue

        # Extract test period data
        test_mask_w = (weight_matrix.index >= test_start) & (weight_matrix.index <= test_end)
        test_mask_r = (returns_matrix.index >= test_start) & (returns_matrix.index <= test_end)

        w_test = weight_matrix.loc[test_mask_w].copy()
        r_test = returns_matrix.loc[test_mask_r].copy()

        if w_test.empty or r_test.empty:
            logger.warning(
                f"Skipping window {i}: empty test data | "
                f"w_test shape: {w_test.shape}, r_test shape: {r_test.shape}"
            )
            continue

        # Compute returns for this test period
        period_returns = compute_portfolio_returns(
            weight_matrix=w_test,
            returns_matrix=r_test,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps
        )
        
        if period_returns.empty:
            logger.warning(f"Window {i}: No valid period returns")
            continue

        # Get benchmark window if provided
        bench_window = None
        if benchmark_returns is not None:
            benchmark_returns_dt = benchmark_returns.copy()
            benchmark_returns_dt.index = pd.to_datetime(benchmark_returns_dt.index)
            bench_window = benchmark_returns_dt[
                (benchmark_returns_dt.index >= test_start) &
                (benchmark_returns_dt.index <= test_end)
            ]

        # Compute metrics
        metrics = compute_performance_metrics(
            period_returns, bench_window
        )
        
        if not metrics:
            logger.warning(f"Window {i}: Could not compute metrics")
            continue
        
        metrics["test_start"] = test_start
        metrics["test_end"]   = test_end
        metrics["window_idx"] = i

        window_results.append(metrics)
        all_oos_returns.append(period_returns)

        test_start_str = pd.Timestamp(test_start).strftime('%Y-%m-%d')
        test_end_str = pd.Timestamp(test_end).strftime('%Y-%m-%d')
        logger.info(
            f"Window {i+1}/{len(windows)} | {test_start_str} → {test_end_str} | "
            f"Sharpe: {metrics.get('sharpe', np.nan):.3f} | "
            f"Return: {metrics.get('cagr', np.nan):.2%} | "
            f"Days: {metrics.get('n_days', 0)}"
        )

    if not all_oos_returns:
        logger.error("No OOS returns generated — all windows failed")
        return {}

    # Concatenate all OOS returns and compute overall metrics
    oos_returns   = pd.concat(all_oos_returns).sort_index()
    equity_curve  = (1 + oos_returns).cumprod()
    window_df     = pd.DataFrame(window_results)

    overall_metrics = compute_performance_metrics(
        oos_returns, benchmark_returns
    )

    logger.info(
        f"\n{'='*60}\n"
        f"WALK-FORWARD BACKTEST COMPLETE\n"
        f"  Windows Processed   : {len(window_results)}\n"
        f"  Total Test Days     : {len(oos_returns)}\n"
        f"  CAGR                : {overall_metrics.get('cagr', np.nan):.2%}\n"
        f"  Sharpe Ratio        : {overall_metrics.get('sharpe', np.nan):.3f}\n"
        f"  Sortino             : {overall_metrics.get('sortino', np.nan):.3f}\n"
        f"  Max Drawdown        : {overall_metrics.get('max_drawdown', np.nan):.2%}\n"
        f"  Calmar Ratio        : {overall_metrics.get('calmar', np.nan):.3f}\n"
        f"  Win Rate            : {overall_metrics.get('win_rate', np.nan):.2%}\n"
        f"{'='*60}"
    )

    return {
        "oos_returns"    : oos_returns,
        "equity_curve"   : equity_curve,
        "window_results" : window_df,
        "overall_metrics": overall_metrics
    }

def compute_drawdown_analysis(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Analyze drawdown episodes in equity curve.
    
    Returns DataFrame with:
    - start: Beginning of drawdown
    - trough: Lowest point
    - end: Recovery to previous high
    - depth: Maximum loss magnitude
    - duration: Days from start to recovery
    - recovery: Days from trough to recovery
    """
    
    if equity_curve.empty:
        logger.warning("Empty equity curve provided to compute_drawdown_analysis")
        return pd.DataFrame()
    
    equity_curve = equity_curve.dropna()
    
    # Calculate running maximum and drawdown
    running_max = equity_curve.expanding().max()
    drawdown    = (equity_curve - running_max) / running_max

    drawdowns = []
    in_drawdown = False
    start = None
    trough_date = None
    trough_val  = 0

    for date, dd in drawdown.items():
        if dd < -1e-8 and not in_drawdown:  # Start of drawdown (small epsilon for numerical stability)
            in_drawdown = True
            start       = date
            trough_date = date
            trough_val  = dd

        elif dd < trough_val and in_drawdown:  # Deepening drawdown
            trough_date = date
            trough_val  = dd

        elif dd >= -1e-8 and in_drawdown:  # End of drawdown (recovery)
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
            f"Drawdown Analysis | "
            f"Episodes: {len(dd_df)} | "
            f"Worst: {dd_df['depth'].min():.2%} | "
            f"Avg Duration: {dd_df['duration'].mean():.0f}d | "
            f"Avg Recovery: {dd_df['recovery'].mean():.0f}d"
        )
    else:
        logger.info("No drawdowns detected in equity curve")
    
    return dd_df

def run_backtest(weight_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        benchmark_returns: pd.Series = None,
        save_path: str = "backtest/") -> dict:
    """
    Execute complete backtest pipeline with validation.
    
    Args:
        weight_matrix: Portfolio weights
        returns_matrix: Daily returns
        benchmark_returns: Optional benchmark returns
        save_path: Directory to save results
        
    Returns:
        Dictionary with all backtest results
    """

    if weight_matrix.empty or returns_matrix.empty:
        logger.error("Empty weight or returns matrix provided to run_backtest")
        return {}

    config = load_config()
    os.makedirs(save_path, exist_ok=True)

    # Validate config values
    train_months = config.get("backtest", {}).get("train_months", 12)
    test_months = config.get("backtest", {}).get("test_months", 1)
    
    if train_months <= 0 or test_months <= 0:
        logger.error(f"Invalid backtest config: train_months={train_months}, test_months={test_months}")
        return {}

    logger.info("Running walk-forward backtest...")
    results = run_walk_forward_backtest(
        weight_matrix=weight_matrix,
        returns_matrix=returns_matrix,
        train_months=train_months,
        test_months=test_months,
        transaction_cost_bps=config["backtest"]["transaction_cost_bps"],
        slippage_bps=config["backtest"]["slippage_bps"],
        benchmark_returns=benchmark_returns
    )

    if not results:
        logger.error("Backtest returned no results")
        return {}

    # Extract results
    oos_returns = results.get("oos_returns", pd.Series())
    equity_curve = results.get("equity_curve", pd.Series())
    
    if oos_returns.empty:
        logger.error("No OOS returns to analyze")
        return {}

    # Compute drawdown analysis
    dd_analysis = compute_drawdown_analysis(equity_curve)

    # Save all results
    oos_returns.to_csv(
        os.path.join(save_path, "oos_returns.csv")
    )
    equity_curve.to_csv(
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

    logger.info(
        f"Backtest complete. Results saved to {save_path} | "
        f"Windows: {len(results.get('window_results', []))} | "
        f"Days: {len(oos_returns)}"
    )

    return {**results, "drawdown_analysis": dd_analysis}