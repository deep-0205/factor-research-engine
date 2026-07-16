import pandas as pd
import numpy as np
import os
import yaml
from scipy import stats
from scipy.stats import norm
from logger import get_logger

logger = get_logger("risk_engine")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)
    
def compute_historical_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> dict:

    clean = returns.dropna()

    var_1d = np.percentile(clean, (1 - confidence) * 100)

    var_horizon = var_1d * np.sqrt(horizon)

    breaches     = clean[clean < var_1d]
    breach_rate  = len(breaches) / len(clean)
    expected_rate = 1 - confidence

    result = {
        "var_1d"        : round(var_1d, 6),
        "var_horizon"   : round(var_horizon, 6),
        "confidence"    : confidence,
        "horizon"       : horizon,
        "breach_count"  : len(breaches),
        "breach_rate"   : round(breach_rate, 6),
        "expected_rate" : expected_rate,
        "n_observations": len(clean)
    }

    logger.info(
        f"Historical VaR ({confidence:.0%}) | "
        f"1-day: {var_1d:.4f} | "
        f"Breach rate: {breach_rate:.4f} "
        f"(expected: {expected_rate:.4f})"
    )
    return result

def compute_parametric_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> dict:

    clean = returns.dropna()
    mu    = clean.mean()
    sigma = clean.std()

    z_score  = norm.ppf(1 - confidence)
    var_1d   = mu + z_score * sigma
    var_horizon = var_1d * np.sqrt(horizon)

    _, p_value = stats.jarque_bera(clean)
    skewness   = clean.skew()
    kurtosis   = clean.kurt()   

    result = {
        "var_1d"         : round(var_1d, 6),
        "var_horizon"    : round(var_horizon, 6),
        "mu"             : round(mu, 8),
        "sigma"          : round(sigma, 6),
        "z_score"        : round(z_score, 4),
        "confidence"     : confidence,
        "jarque_bera_p"  : round(p_value, 6),
        "is_normal"      : p_value > 0.05,
        "skewness"       : round(skewness, 4),
        "excess_kurtosis": round(kurtosis, 4)
    }

    logger.info(
        f"Parametric VaR ({confidence:.0%}) | "
        f"1-day: {var_1d:.4f} | "
        f"Normal: {p_value > 0.05} (JB p={p_value:.4f})"
    )
    return result

def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> dict:

    clean    = returns.dropna()
    var_threshold = np.percentile(clean, (1 - confidence) * 100)

    tail_returns = clean[clean < var_threshold]

    cvar         = tail_returns.mean()
    tail_std     = tail_returns.std()
    worst_day    = tail_returns.min()
    tail_skew    = tail_returns.skew() if len(tail_returns) > 3 else np.nan

    result = {
        "cvar"           : round(cvar, 6),
        "var_threshold"  : round(var_threshold, 6),
        "confidence"     : confidence,
        "tail_obs"       : len(tail_returns),
        "tail_std"       : round(tail_std, 6),
        "worst_day"      : round(worst_day, 6),
        "tail_skewness"  : round(tail_skew, 4) if not np.isnan(tail_skew) else np.nan,
        "cvar_to_var"    : round(cvar / var_threshold, 4) if var_threshold != 0 else np.nan
    }

    logger.info(
        f"CVaR ({confidence:.0%}) | "
        f"CVaR: {cvar:.4f} | VaR: {var_threshold:.4f} | "
        f"Ratio: {result['cvar_to_var']:.3f}"
    )
    return result

def compute_rolling_risk(returns: pd.Series, window: int = 63) -> pd.DataFrame:

    rf_daily = (1 + 0.065) ** (1/252) - 1

    rolling_metrics = pd.DataFrame(index=returns.index)

    rolling_metrics["rolling_vol"] = (
        returns.rolling(window).std() * np.sqrt(252)
    )

    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()
    rolling_metrics["rolling_sharpe"] = (
        (rolling_mean - rf_daily) / rolling_std * np.sqrt(252)
    )

    rolling_var  = []
    rolling_cvar = []
    rolling_mdd  = []

    for i in range(len(returns)):
        if i < window:
            rolling_var.append(np.nan)
            rolling_cvar.append(np.nan)
            rolling_mdd.append(np.nan)
            continue

        window_ret = returns.iloc[i - window: i]

        var = np.percentile(window_ret.dropna(), 5)
        rolling_var.append(var)

        tail = window_ret[window_ret < var]
        rolling_cvar.append(tail.mean() if len(tail) > 0 else var)

        eq   = (1 + window_ret).cumprod()
        peak = eq.cummax()
        dd   = ((eq - peak) / peak).min()
        rolling_mdd.append(dd)

    rolling_metrics["rolling_var95"]  = rolling_var
    rolling_metrics["rolling_cvar95"] = rolling_cvar
    rolling_metrics["rolling_maxdd"]  = rolling_mdd

    logger.info(
        f"Rolling risk computed | Window: {window}d | "
        f"Avg vol: {rolling_metrics['rolling_vol'].mean():.4f}"
    )
    return rolling_metrics

def run_stress_tests(returns: pd.Series, weight_matrix: pd.DataFrame, returns_matrix: pd.DataFrame) -> pd.DataFrame:

    stress_periods = {
        "COVID_Crash_2020"    : ("2020-02-01", "2020-03-31"),
        "IL&FS_Crisis_2018"   : ("2018-09-01", "2018-12-31"),
        "COVID_Recovery_2020" : ("2020-04-01", "2020-06-30"),
        "Bear_Market_2022"    : ("2022-01-01", "2022-06-30"),
        "Demonetization_2016" : ("2016-11-01", "2016-12-31"),
        "Global_GFC_2008"     : ("2008-09-01", "2009-03-31")
    }

    stress_results = []

    for name, (start, end) in stress_periods.items():
        try:
            period_ret = returns[
                (returns.index >= start) &
                (returns.index <= end)
            ]

            if len(period_ret) < 5:
                logger.warning(
                    f"Stress test {name}: insufficient data, skipping"
                )
                continue

            total_ret  = (1 + period_ret).prod() - 1
            eq         = (1 + period_ret).cumprod()
            peak       = eq.cummax()
            max_dd     = ((eq - peak) / peak).min()
            daily_vol  = period_ret.std() * np.sqrt(252)
            worst_day  = period_ret.min()
            best_day   = period_ret.max()

            stress_results.append({
                "period"       : name,
                "start"        : start,
                "end"          : end,
                "n_days"       : len(period_ret),
                "total_return" : round(total_ret, 6),
                "max_drawdown" : round(max_dd, 6),
                "daily_vol"    : round(daily_vol, 6),
                "worst_day"    : round(worst_day, 6),
                "best_day"     : round(best_day, 6)
            })

            logger.info(
                f"Stress [{name}] | "
                f"Return: {total_ret:.2%} | "
                f"MaxDD: {max_dd:.2%} | "
                f"Vol: {daily_vol:.2%}"
            )

        except Exception as e:
            logger.warning(f"Stress test {name} failed: {e}")

    stress_df = pd.DataFrame(stress_results).set_index("period")
    return stress_df

def compute_factor_risk_decomposition(returns: pd.Series, factor_returns: dict) -> pd.DataFrame:

    from statsmodels.api import OLS, add_constant

    factor_df = pd.DataFrame(factor_returns)
    common    = returns.index.intersection(factor_df.index)

    if len(common) < 30:
        logger.warning("Insufficient data for factor decomposition")
        return pd.DataFrame()

    y = returns[common]
    X = add_constant(factor_df.loc[common])

    try:
        model  = OLS(y, X).fit()
        params = model.params

        results = []
        for name in factor_df.columns:
            beta     = params.get(name, np.nan)
            f_vol    = factor_df[name].std() * np.sqrt(252)
            f_var_contrib = (beta ** 2) * (f_vol ** 2)

            results.append({
                "factor"        : name,
                "beta"          : round(beta, 6),
                "t_stat"        : round(model.tvalues.get(name, np.nan), 4),
                "p_value"       : round(model.pvalues.get(name, np.nan), 6),
                "factor_vol"    : round(f_vol, 6),
                "var_contrib"   : round(f_var_contrib, 8),
                "significant"   : model.pvalues.get(name, 1) < 0.05
            })

        decomp_df = pd.DataFrame(results).set_index("factor")

        logger.info(
            f"Factor Decomposition | "
            f"Alpha: {params.get('const', np.nan)*252:.4f} ann. | "
            f"R²: {model.rsquared:.4f}"
        )

        decomp_df.attrs["alpha"]     = params.get("const", np.nan) * 252
        decomp_df.attrs["r_squared"] = model.rsquared
        decomp_df.attrs["f_pvalue"]  = model.f_pvalue

        return decomp_df

    except Exception as e:
        logger.error(f"Factor decomposition failed: {e}")
        return pd.DataFrame()
    
def compute_tail_risk(returns: pd.Series) -> dict:

    clean = returns.dropna()

    skewness = clean.skew()
    kurtosis = clean.kurt()    

    gains  = clean[clean > 0].sum()
    pains  = clean[clean < 0].abs().sum()
    gain_to_pain = gains / pains if pains > 0 else np.nan

    equity   = (1 + clean).cumprod()
    peak     = equity.cummax()
    drawdown = (equity - peak) / peak
    ulcer    = np.sqrt((drawdown ** 2).mean())

    hist_var  = np.percentile(clean, 5)
    param_var = clean.mean() + norm.ppf(0.05) * clean.std()
    var_ratio = hist_var / param_var if param_var != 0 else np.nan

    threshold = 0.0
    omega_num  = clean[clean > threshold].sum()
    omega_den  = clean[clean < threshold].abs().sum()
    omega      = omega_num / omega_den if omega_den > 0 else np.nan

    result = {
        "skewness"     : round(skewness, 4),
        "excess_kurt"  : round(kurtosis, 4),
        "gain_to_pain" : round(gain_to_pain, 4),
        "ulcer_index"  : round(ulcer, 6),
        "var_ratio"    : round(var_ratio, 4),
        "omega_ratio"  : round(omega, 4),
        "best_day"     : round(clean.max(), 6),
        "worst_day"    : round(clean.min(), 6),
        "positive_days": round((clean > 0).mean(), 4)
    }

    logger.info(
        f"Tail Risk | Skew: {skewness:.4f} | "
        f"Kurt: {kurtosis:.4f} | "
        f"Gain/Pain: {gain_to_pain:.4f} | "
        f"Ulcer: {ulcer:.6f}"
    )
    return result

def run_risk_engine(oos_returns: pd.Series,
        weight_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        factor_returns: dict = None,
        save_path: str = "risk/") -> dict:

    config = load_config()
    os.makedirs(save_path, exist_ok=True)

    logger.info("Computing VaR metrics...")
    hist_var_95  = compute_historical_var(oos_returns, confidence=0.95)
    hist_var_99  = compute_historical_var(oos_returns, confidence=0.99)
    param_var_95 = compute_parametric_var(oos_returns, confidence=0.95)
    param_var_99 = compute_parametric_var(oos_returns, confidence=0.99)

    logger.info("Computing CVaR...")
    cvar_95 = compute_cvar(oos_returns, confidence=0.95)
    cvar_99 = compute_cvar(oos_returns, confidence=0.99)

    logger.info("Computing rolling risk metrics...")
    rolling_risk = compute_rolling_risk(oos_returns, window=63)

    logger.info("Running stress tests...")
    stress_results = run_stress_tests(
        oos_returns, weight_matrix, returns_matrix
    )

    decomp_df = pd.DataFrame()
    if factor_returns:
        logger.info("Running factor risk decomposition...")
        decomp_df = compute_factor_risk_decomposition(
            oos_returns, factor_returns
        )

    logger.info("Computing tail risk statistics...")
    tail_risk = compute_tail_risk(oos_returns)

    risk_report = {
        "hist_var_95"  : hist_var_95["var_1d"],
        "hist_var_99"  : hist_var_99["var_1d"],
        "param_var_95" : param_var_95["var_1d"],
        "param_var_99" : param_var_99["var_1d"],
        "cvar_95"      : cvar_95["cvar"],
        "cvar_99"      : cvar_99["cvar"],
        "is_normal"    : param_var_95["is_normal"],
        "skewness"     : tail_risk["skewness"],
        "excess_kurt"  : tail_risk["excess_kurt"],
        "ulcer_index"  : tail_risk["ulcer_index"],
        "gain_to_pain" : tail_risk["gain_to_pain"],
        "omega_ratio"  : tail_risk["omega_ratio"]
    }

    pd.Series(hist_var_95).to_csv(
        os.path.join(save_path, "var_95.csv")
    )
    pd.Series(cvar_95).to_csv(
        os.path.join(save_path, "cvar_95.csv")
    )
    rolling_risk.to_csv(
        os.path.join(save_path, "rolling_risk.csv")
    )
    stress_results.to_csv(
        os.path.join(save_path, "stress_tests.csv")
    )
    if not decomp_df.empty:
        decomp_df.to_csv(
            os.path.join(save_path, "factor_decomposition.csv")
        )
    pd.Series(tail_risk).to_csv(
        os.path.join(save_path, "tail_risk.csv")
    )
    pd.Series(risk_report).to_csv(
        os.path.join(save_path, "risk_report.csv")
    )

    logger.info("Risk engine complete. All outputs saved to risk/")

    return {
        "hist_var_95"  : hist_var_95,
        "hist_var_99"  : hist_var_99,
        "cvar_95"      : cvar_95,
        "rolling_risk" : rolling_risk,
        "stress_tests" : stress_results,
        "factor_decomp": decomp_df,
        "tail_risk"    : tail_risk,
        "risk_report"  : risk_report
    }

