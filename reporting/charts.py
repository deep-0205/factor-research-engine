import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from logger import get_logger

logger = get_logger("charts")

COLORS = {
    "primary"  : "#1f4e79",
    "secondary": "#2e86c1",
    "positive" : "#1e8449",
    "negative" : "#922b21",
    "neutral"  : "#717d7e",
    "highlight": "#d4ac0d"
}

plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#f8f9fa",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "grid.linestyle"   : "--",
    "font.family"      : "sans-serif",
    "font.size"        : 10,
    "axes.titlesize"   : 12,
    "axes.titleweight" : "bold",
    "axes.spines.top"  : False,
    "axes.spines.right": False
})


def save_fig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Chart saved: {path}")

def plot_equity_curve(equity_curve: pd.Series, oos_returns: pd.Series, save_path: str = None) -> plt.Figure:

    if equity_curve.empty:
        logger.warning("Empty equity curve — skipping chart")
        return None

    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.suptitle("Strategy Equity Curve & Drawdowns", fontsize=14, fontweight="bold", y=1.01)

    ax1.plot(equity_curve.index, equity_curve.values, color=COLORS["primary"], linewidth=1.8, label="Strategy")
    ax1.fill_between(equity_curve.index, 1, equity_curve.values, where=equity_curve.values >= 1, alpha=0.15, color=COLORS["positive"])
    ax1.fill_between(equity_curve.index, 1, equity_curve.values, where=equity_curve.values < 1, alpha=0.15, color=COLORS["negative"])
    ax1.axhline(1.0, color=COLORS["neutral"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.set_ylabel("Portfolio Value (Base = 1.0)")
    ax1.legend(loc="upper left")

    final_val = equity_curve.iloc[-1]
    ax1.annotate(
        f"  {final_val:.2f}×",
        xy=(equity_curve.index[-1], final_val),
        fontsize=10, color=COLORS["primary"], fontweight="bold"
    )

    ax2.fill_between(drawdown.index, drawdown.values, 0, color=COLORS["negative"], alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values, color=COLORS["negative"], linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        save_fig(fig, save_path)

    return fig

def plot_rolling_risk(rolling_risk: pd.DataFrame, save_path: str = None) -> plt.Figure:

    if rolling_risk.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Rolling Risk Metrics (63-Day Window)", fontsize=14, fontweight="bold")

    if "rolling_sharpe" in rolling_risk.columns:
        sharpe = rolling_risk["rolling_sharpe"].dropna()
        ax1.plot(sharpe.index, sharpe.values, color=COLORS["secondary"], linewidth=1.2)
        ax1.axhline(1.0, color=COLORS["positive"], linewidth=0.8, linestyle="--", label="Sharpe = 1.0")
        ax1.axhline(0.0, color=COLORS["negative"], linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.fill_between(sharpe.index, sharpe.values, 0, where=sharpe.values >= 0, alpha=0.1, color=COLORS["positive"])
        ax1.fill_between(sharpe.index, sharpe.values, 0, where=sharpe.values < 0, alpha=0.1, color=COLORS["negative"])
        ax1.set_ylabel("Rolling Sharpe Ratio")
        ax1.legend(loc="upper right")

    if "rolling_vol" in rolling_risk.columns:
        vol = rolling_risk["rolling_vol"].dropna()
        ax2.plot(vol.index, vol.values, color=COLORS["highlight"], linewidth=1.2)
        ax2.fill_between(vol.index, vol.values, 0, alpha=0.15, color=COLORS["highlight"])
        ax2.set_ylabel("Rolling Volatility (Ann.)")
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        save_fig(fig, save_path)

    return fig

def plot_ic_series(ic_series: pd.Series, save_path: str = None) -> plt.Figure:

    if ic_series.empty:
        return None

    ic = ic_series.dropna()
    cumulative_ic = ic.cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Factor Information Coefficient (IC)", fontsize=14, fontweight="bold")

    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in ic.values]
    ax1.bar(ic.index, ic.values, color=colors, alpha=0.7, width=5)
    ax1.axhline(ic.mean(), color=COLORS["highlight"], linewidth=1.2, linestyle="--", label=f"Mean IC = {ic.mean():.4f}")
    ax1.axhline(0, color=COLORS["neutral"], linewidth=0.6, linestyle="-")
    ax1.set_ylabel("IC")
    ax1.legend()

    ax2.plot(cumulative_ic.index, cumulative_ic.values, color=COLORS["primary"], linewidth=1.5)
    ax2.fill_between(cumulative_ic.index, cumulative_ic.values, 0, alpha=0.1, color=COLORS["primary"])
    ax2.set_ylabel("Cumulative IC")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        save_fig(fig, save_path)

    return fig

def plot_stress_tests(stress_tests: pd.DataFrame, save_path: str = None) -> plt.Figure:

    if stress_tests.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stress Test Analysis", fontsize=14, fontweight="bold")

    stress = stress_tests.copy()

    ret_colors = [
        COLORS["positive"] if v >= 0 else COLORS["negative"]
        for v in stress["total_return"].values
    ]
    bars = ax1.barh(stress.index, stress["total_return"] * 100, color=ret_colors, alpha=0.8, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Total Return (%)")
    ax1.set_title("Strategy Return During Stress Periods")

    for bar, val in zip(bars, stress["total_return"].values):
        ax1.text(
            bar.get_width() + 0.3 if val >= 0
            else bar.get_width() - 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9
        )

    dd_colors = [COLORS["negative"]] * len(stress)
    ax2.barh(stress.index, stress["max_drawdown"] * 100, color=dd_colors, alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Max Drawdown (%)")
    ax2.set_title("Max Drawdown During Stress Periods")

    plt.tight_layout()

    if save_path:
        save_fig(fig, save_path)

    return fig


def plot_monthly_returns_heatmap(oos_returns: pd.Series, save_path: str = None) -> plt.Figure:

    if oos_returns.empty:
        return None

    monthly = oos_returns.resample("M").apply(
        lambda x: (1 + x).prod() - 1
    )
    pivot = monthly.groupby(
        [monthly.index.year, monthly.index.month]
    ).first().unstack(level=1)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
    fig.suptitle("Monthly Returns Heatmap", fontsize=14, fontweight="bold")

    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Return (%)"}
    )
    ax.set_xlabel("")
    ax.set_ylabel("Year")

    plt.tight_layout()

    if save_path:
        save_fig(fig, save_path)

    return fig


