import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import os
from datetime import datetime

st.set_page_config(
    page_title="Factor Research Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "primary"  : "#1f4e79",
    "secondary": "#2e86c1",
    "positive" : "#1e8449",
    "negative" : "#922b21",
    "neutral"  : "#717d7e",
    "highlight": "#d4ac0d",
    "bg_card"  : "#f0f4f8"
}

@st.cache_data(ttl=300)   
def load_data() -> dict:

    def safe_csv(path, index_col=0, parse_dates=True):
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path, index_col=index_col,
                               parse_dates=parse_dates)
        except Exception:
            return pd.DataFrame()

    def safe_series(path):
        df = safe_csv(path)
        if df.empty:
            return pd.Series(dtype=float)
        return df.iloc[:, 0]

    def safe_yaml(path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    return {
        "equity_curve"    : safe_series("backtest/equity_curve.csv"),
        "oos_returns"     : safe_series("backtest/oos_returns.csv"),
        "overall_metrics" : safe_series("backtest/overall_metrics.csv"),
        "window_results"  : safe_csv(
            "backtest/window_results.csv",
            index_col=None, parse_dates=False
        ),
        "drawdown_analysis": safe_csv(
            "backtest/drawdown_analysis.csv",
            index_col=None, parse_dates=False
        ),

        "rolling_risk"  : safe_csv("risk/rolling_risk.csv"),
        "stress_tests"  : safe_csv("risk/stress_tests.csv"),
        "risk_report"   : safe_series("risk/risk_report.csv"),
        "tail_risk"     : safe_series("risk/tail_risk.csv"),

        "final_signals" : safe_csv("signals/final_signals.csv"),
        "signal_stats"  : safe_csv("signals/signal_stats.csv"),
        "turnover"      : safe_series("signals/turnover.csv"),

        "weights"       : safe_csv(
            "portfolio/weights_volatility.csv"
        ),
        "portfolio_stats": safe_csv("portfolio/portfolio_stats.csv"),

        "composite_scores": safe_csv("factors/composite_scores.csv"),
        "ic_series"       : safe_series("factors/ic_series.csv"),

        "current_forecasts": safe_series(
            "risk/current_vol_forecasts.csv"
        ),

        "pipeline_state": safe_yaml("logs/pipeline_state.yaml")
    }


def fmt_pct(val, decimals=2):
    try:
        return f"{float(val):.{decimals}%}"
    except Exception:
        return "N/A"


def fmt_num(val, decimals=3):
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return "N/A"


def get_metric(series, key, default=np.nan):
    try:
        return float(series[key])
    except Exception:
        return default
    
def render_sidebar(data: dict):

    st.sidebar.image(
        "https://img.icons8.com/fluency/96/combo-chart.png",
        width=60
    )
    st.sidebar.title("Factor Research Engine")
    st.sidebar.markdown("---")

    state  = data["pipeline_state"]
    status = state.get("status", "UNKNOWN")

    color_map = {
        "COMPLETE": "🟢",
        "PARTIAL" : "🟡",
        "FAILED"  : "🔴",
        "UNKNOWN" : "⚪"
    }
    st.sidebar.markdown(f"**Pipeline:** {color_map.get(status, '⚪')} {status}")
    st.sidebar.markdown(f"**Last Run:** {state.get('run_date', 'N/A')}")
    st.sidebar.markdown(f"**Universe:** {state.get('universe_size', 'N/A')} stocks")
    st.sidebar.markdown("---")

    regime = state.get("market_regime", "N/A")
    regime_color = {
        "LOW"    : "🟢 LOW",
        "MEDIUM" : "🟡 MEDIUM",
        "HIGH"   : "🔴 HIGH",
        "UNKNOWN": "⚪ UNKNOWN"
    }
    st.sidebar.markdown("**Market Regime**")
    st.sidebar.markdown(
        f"### {regime_color.get(regime, regime)}"
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Current Positions**")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Longs",  state.get("n_longs",  0), delta=None)
    col2.metric("Shorts", state.get("n_shorts", 0), delta=None)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Navigate**")
    page = st.sidebar.radio(
        label="",
        options=[
            "📊 Overview",
            "📈 Equity Curve",
            "⚖️  Risk Metrics",
            "🧮 Factor Analysis",
            "💼 Portfolio",
            "🚨 Stress Tests",
            "📋 Signal Monitor"
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown(
        f"*Updated: {datetime.now().strftime('%H:%M:%S')}*"
    )

    return page

def render_overview(data: dict):

    st.title("📊 Strategy Overview")
    st.markdown("---")

    metrics  = data["overall_metrics"]
    risk     = data["risk_report"]
    state    = data["pipeline_state"]

    st.subheader("Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)

    cagr    = get_metric(metrics, "cagr")
    sharpe  = get_metric(metrics, "sharpe")
    sortino = get_metric(metrics, "sortino")
    max_dd  = get_metric(metrics, "max_drawdown")
    calmar  = get_metric(metrics, "calmar")

    c1.metric(
        "CAGR",
        fmt_pct(cagr),
        delta=f"{'▲' if cagr >= 0 else '▼'} vs 0%"
    )
    c2.metric(
        "Sharpe Ratio",
        fmt_num(sharpe),
        delta=f"{'Good ✓' if sharpe >= 1 else 'Below 1.0'}"
    )
    c3.metric("Sortino Ratio", fmt_num(sortino))
    c4.metric("Max Drawdown",  fmt_pct(max_dd))
    c5.metric("Calmar Ratio",  fmt_num(calmar))

    st.markdown("---")

    st.subheader("Risk Metrics")
    r1, r2, r3, r4, r5 = st.columns(5)

    r1.metric("VaR 95%",    fmt_pct(get_metric(risk, "hist_var_95")))
    r2.metric("VaR 99%",    fmt_pct(get_metric(risk, "hist_var_99")))
    r3.metric("CVaR 95%",   fmt_pct(get_metric(risk, "cvar_95")))
    r4.metric("Win Rate",   fmt_pct(get_metric(metrics, "win_rate")))
    r5.metric(
        "Total Return",
        fmt_pct(get_metric(metrics, "total_return"))
    )

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Equity Curve")
        eq = data["equity_curve"]
        if not eq.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                mode="lines",
                name="Strategy",
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor="rgba(31,78,121,0.08)"
            ))
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color=COLORS["neutral"],
                opacity=0.5
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis_title="",
                yaxis_title="Portfolio Value"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Rolling Sharpe (63d)")
        rr = data["rolling_risk"]
        if not rr.empty and "rolling_sharpe" in rr.columns:
            sharpe_roll = rr["rolling_sharpe"].dropna()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sharpe_roll.index,
                y=sharpe_roll.values,
                mode="lines",
                line=dict(color=COLORS["secondary"], width=1.5)
            ))
            fig2.add_hline(
                y=1.0,
                line_dash="dash",
                line_color=COLORS["positive"],
                opacity=0.6
            )
            fig2.add_hline(
                y=0.0,
                line_dash="dash",
                line_color=COLORS["negative"],
                opacity=0.4
            )
            fig2.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                yaxis_title="Sharpe"
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Walk-Forward Window Results")
    wf = data["window_results"]
    if not wf.empty:
        display_cols = [
            c for c in ["test_start", "test_end", "cagr",
                         "sharpe", "sortino", "max_drawdown",
                         "win_rate"]
            if c in wf.columns
        ]
        wf_display = wf[display_cols].copy()

        for col in ["cagr", "max_drawdown", "win_rate"]:
            if col in wf_display.columns:
                wf_display[col] = wf_display[col].apply(
                    lambda x: fmt_pct(x) if pd.notna(x) else "N/A"
                )
        for col in ["sharpe", "sortino"]:
            if col in wf_display.columns:
                wf_display[col] = wf_display[col].apply(
                    lambda x: fmt_num(x) if pd.notna(x) else "N/A"
                )

        st.dataframe(
            wf_display,
            use_container_width=True,
            hide_index=True
        )

def render_equity_curve(data: dict):

    st.title("📈 Equity Curve & Drawdowns")
    st.markdown("---")

    eq  = data["equity_curve"]
    ret = data["oos_returns"]

    if eq.empty:
        st.warning("No equity curve data found. Run backtest first.")
        return

    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("Cumulative Equity", "Drawdown")
    )

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        mode="lines", name="Strategy",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.06)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", name="Drawdown",
        line=dict(color=COLORS["negative"], width=1),
        fill="tozeroy",
        fillcolor="rgba(146,43,33,0.25)"
    ), row=2, col=1)

    fig.update_yaxes(
        tickformat=".0%", row=2, col=1
    )
    fig.update_layout(
        height=550,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Monthly Returns Heatmap")
    if not ret.empty:
        monthly = ret.resample("M").apply(
            lambda x: (1 + x).prod() - 1
        )
        pivot = monthly.groupby(
            [monthly.index.year, monthly.index.month]
        ).first().unstack(level=1)

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [
            month_names[m - 1] for m in pivot.columns
        ]

        fig2 = px.imshow(
            pivot * 100,
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            aspect="auto",
            text_auto=".1f"
        )
        fig2.update_layout(
            height=max(300, len(pivot) * 45),
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="Return %")
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Top Drawdown Episodes")
    dd_df = data["drawdown_analysis"]
    if not dd_df.empty and "depth" in dd_df.columns:
        top10 = dd_df.nsmallest(10, "depth").copy()
        for col in ["depth"]:
            top10[col] = top10[col].apply(fmt_pct)
        st.dataframe(top10, use_container_width=True, hide_index=True)


def render_risk_metrics(data: dict):

    st.title("⚖️ Risk Metrics")
    st.markdown("---")

    rr   = data["rolling_risk"]
    risk = data["risk_report"]
    tail = data["tail_risk"]

    st.subheader("Value at Risk Summary")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Historical VaR 95%",
              fmt_pct(get_metric(risk, "hist_var_95")))
    c2.metric("Historical VaR 99%",
              fmt_pct(get_metric(risk, "hist_var_99")))
    c3.metric("CVaR 95%",
              fmt_pct(get_metric(risk, "cvar_95")))
    c4.metric("CVaR 99%",
              fmt_pct(get_metric(risk, "cvar_99")))

    st.markdown("---")

    if not rr.empty:
        st.subheader("Rolling Risk Over Time (63-Day Window)")

        tab1, tab2, tab3 = st.tabs([
            "Volatility", "Sharpe Ratio", "Rolling VaR"
        ])

        with tab1:
            if "rolling_vol" in rr.columns:
                vol = rr["rolling_vol"].dropna()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vol.index, y=vol.values,
                    mode="lines", name="Rolling Vol",
                    line=dict(color=COLORS["highlight"], width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(212,172,13,0.12)"
                ))
                fig.update_yaxes(tickformat=".0%")
                fig.update_layout(
                    height=350, showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if "rolling_sharpe" in rr.columns:
                sr = rr["rolling_sharpe"].dropna()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sr.index, y=sr.values,
                    mode="lines", name="Rolling Sharpe",
                    line=dict(color=COLORS["secondary"], width=1.5)
                ))
                fig.add_hline(y=1.0, line_dash="dash",
                              line_color=COLORS["positive"])
                fig.add_hline(y=0.0, line_dash="dash",
                              line_color=COLORS["negative"],
                              opacity=0.4)
                fig.add_hrect(
                    y0=1.0, y1=sr.max() + 0.1,
                    fillcolor="rgba(30,132,73,0.06)",
                    line_width=0
                )
                fig.update_layout(
                    height=350, showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if "rolling_var95" in rr.columns:
                rv = rr["rolling_var95"].dropna()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rv.index, y=rv.values,
                    mode="lines", name="Rolling VaR 95%",
                    line=dict(color=COLORS["negative"], width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(146,43,33,0.12)"
                ))
                fig.update_yaxes(tickformat=".2%")
                fig.update_layout(
                    height=350, showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Return Distribution")
    ret = data["oos_returns"]
    if not ret.empty:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=ret.values * 100,
                nbinsx=80,
                name="Daily Returns",
                marker_color=COLORS["secondary"],
                opacity=0.75
            ))

            var_val = get_metric(risk, "hist_var_95") * 100
            if not np.isnan(var_val):
                fig.add_vline(
                    x=var_val,
                    line_dash="dash",
                    line_color=COLORS["negative"],
                    annotation_text=f"VaR 95%: {var_val:.2f}%",
                    annotation_position="top right"
                )

            fig.update_layout(
                height=350,
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("**Distribution Statistics**")
            tail_stats = {
                "Skewness"      : fmt_num(
                    get_metric(tail, "skewness")
                ),
                "Excess Kurtosis": fmt_num(
                    get_metric(tail, "excess_kurt")
                ),
                "Best Day"      : fmt_pct(
                    get_metric(tail, "best_day")
                ),
                "Worst Day"     : fmt_pct(
                    get_metric(tail, "worst_day")
                ),
                "Positive Days" : fmt_pct(
                    get_metric(tail, "positive_days")
                ),
                "Gain/Pain"     : fmt_num(
                    get_metric(tail, "gain_to_pain")
                ),
                "Omega Ratio"   : fmt_num(
                    get_metric(tail, "omega_ratio")
                ),
                "Ulcer Index"   : fmt_num(
                    get_metric(tail, "ulcer_index"), decimals=5
                )
            }
            for label, val in tail_stats.items():
                st.markdown(
                    f"**{label}:** {val}"
                )


def render_factor_analysis(data: dict):

    st.title("🧮 Factor Analysis")
    st.markdown("---")

    ic = data["ic_series"]

    if ic.empty:
        st.warning("No IC data found. Run factor engine first.")
        return

    ic_clean = ic.dropna()

    st.subheader("Information Coefficient Summary")
    c1, c2, c3, c4 = st.columns(4)

    ic_mean = ic_clean.mean()
    ic_std  = ic_clean.std()
    ic_ir   = ic_mean / ic_std if ic_std > 0 else 0

    c1.metric("Mean IC", fmt_num(ic_mean, 4), delta="Good" if ic_mean > 0.03 else "Weak")
    c2.metric("IC Std Dev", fmt_num(ic_std, 4))
    c3.metric("IC IR (Mean/Std)", fmt_num(ic_ir, 4))
    c4.metric("Positive IC Days", fmt_pct((ic_clean > 0).mean()))

    st.markdown("---")

    st.subheader("IC Over Time")
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=("Daily IC", "Cumulative IC")
    )

    colors_ic = [
        COLORS["positive"] if v >= 0 else COLORS["negative"]
        for v in ic_clean.values
    ]

    fig.add_trace(go.Bar(
        x=ic_clean.index,
        y=ic_clean.values,
        name="IC",
        marker_color=colors_ic,
        opacity=0.75
    ), row=1, col=1)

    fig.add_hline(
        y=ic_mean,
        line_dash="dash",
        line_color=COLORS["highlight"],
        annotation_text=f"Mean={ic_mean:.4f}",
        row=1, col=1
    )

    cum_ic = ic_clean.cumsum()
    fig.add_trace(go.Scatter(
        x=cum_ic.index,
        y=cum_ic.values,
        mode="lines",
        name="Cumulative IC",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(31,78,121,0.08)"
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Today's Composite Factor Scores")
    scores = data["composite_scores"]
    if not scores.empty:
        today = scores.iloc[-1].dropna().sort_values(ascending=False)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Top 15 Stocks (Long Candidates)**")
            top15 = today.head(15).reset_index()
            top15.columns = ["Ticker", "Score"]
            top15["Score"] = top15["Score"].apply(lambda x: f"{x:.4f}")
            st.dataframe(top15, use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("**Bottom 15 Stocks (Short Candidates)**")
            bot15 = today.tail(15).reset_index()
            bot15.columns = ["Ticker", "Score"]
            bot15["Score"] = bot15["Score"].apply(lambda x: f"{x:.4f}")
            st.dataframe(bot15, use_container_width=True, hide_index=True)


def render_portfolio(data: dict):

    st.title("💼 Portfolio")
    st.markdown("---")

    weights      = data["weights"]
    port_stats   = data["portfolio_stats"]
    vol_forecast = data["current_forecasts"]

    if weights.empty:
        st.warning("No portfolio data found.")
        return

    today_weights = weights.iloc[-1].dropna()
    today_weights = today_weights[today_weights != 0]

    longs  = today_weights[today_weights > 0].sort_values(
        ascending=False
    )
    shorts = today_weights[today_weights < 0].sort_values()

    st.subheader("Exposure Summary")
    c1, c2, c3, c4 = st.columns(4)

    gross = today_weights.abs().sum()
    net   = today_weights.sum()

    c1.metric("Gross Exposure", fmt_pct(gross))
    c2.metric("Net Exposure",   fmt_pct(net))
    c3.metric("Long Positions", len(longs))
    c4.metric("Short Positions", len(shorts))

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Long Book (Top 20)")
        if not longs.empty:
            top20_long = longs.head(20)
            fig = go.Figure(go.Bar(
                x=top20_long.values * 100,
                y=top20_long.index,
                orientation="h",
                marker_color=COLORS["positive"],
                opacity=0.8
            ))
            fig.update_layout(
                height=max(300, len(top20_long) * 28),
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Weight (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Short Book (Top 20)")
        if not shorts.empty:
            top20_short = shorts.head(20)
            fig = go.Figure(go.Bar(
                x=top20_short.values * 100,
                y=top20_short.index,
                orientation="h",
                marker_color=COLORS["negative"],
                opacity=0.8
            ))
            fig.update_layout(
                height=max(300, len(top20_short) * 28),
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Weight (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    if not port_stats.empty:
        st.subheader("Historical Exposure")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.5],
            vertical_spacing=0.08,
            subplot_titles=("Gross Exposure", "Net Exposure")
        )

        if "gross_exposure" in port_stats.columns:
            fig.add_trace(go.Scatter(
                x=port_stats.index,
                y=port_stats["gross_exposure"],
                mode="lines",
                line=dict(color=COLORS["primary"], width=1.5),
                name="Gross"
            ), row=1, col=1)

        if "net_exposure" in port_stats.columns:
            net_s = port_stats["net_exposure"]
            fig.add_trace(go.Scatter(
                x=port_stats.index,
                y=net_s,
                mode="lines",
                line=dict(color=COLORS["secondary"], width=1.5),
                fill="tozeroy",
                fillcolor="rgba(46,134,193,0.1)",
                name="Net"
            ), row=2, col=1)
            fig.add_hline(
                y=0, line_dash="dash",
                line_color=COLORS["neutral"],
                opacity=0.5, row=2, col=1
            )

        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Current Volatility Forecasts (GARCH)")
    if not vol_forecast.empty:
        vf = vol_forecast.sort_values(ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Highest Vol Stocks**")
            top_vol = vf.head(15).reset_index()
            top_vol.columns = ["Ticker", "Ann. Vol"]
            top_vol["Ann. Vol"] = top_vol["Ann. Vol"].apply(fmt_pct)
            st.dataframe(top_vol, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Lowest Vol Stocks**")
            low_vol = vf.tail(15).reset_index()
            low_vol.columns = ["Ticker", "Ann. Vol"]
            low_vol["Ann. Vol"] = low_vol["Ann. Vol"].apply(fmt_pct)
            st.dataframe(low_vol, use_container_width=True, hide_index=True)



def render_stress_tests(data: dict):
    """
    Stress test results across historical crisis periods.
    """
    st.title("🚨 Stress Tests")
    st.markdown("---")

    st_df = data["stress_tests"]

    if st_df.empty:
        st.warning("No stress test data found.")
        return

    st.subheader("Strategy Return During Stress Periods")

    colors_stress = [
        COLORS["positive"] if v >= 0 else COLORS["negative"]
        for v in st_df["total_return"].values
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Return", "Max Drawdown")
    )

    fig.add_trace(go.Bar(
        x=st_df.index,
        y=st_df["total_return"] * 100,
        marker_color=colors_stress,
        opacity=0.85,
        name="Return"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=st_df.index,
        y=st_df["max_drawdown"] * 100,
        marker_color=COLORS["negative"],
        opacity=0.75,
        name="Max DD"
    ), row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Stress Period Detail")
    display = st_df.copy()

    for col in ["total_return", "max_drawdown", "daily_vol", "worst_day", "best_day"]:
        if col in display.columns:
            display[col] = display[col].apply(fmt_pct)

    st.dataframe(display, use_container_width=True)

# dashboard/app.py (continued)

def main():

    with st.spinner("Loading pipeline data..."):
        data = load_data()

    page = render_sidebar(data)

    if page == "📊 Overview":
        render_overview(data)

    elif page == "📈 Equity Curve":
        render_equity_curve(data)

    elif page == "⚖️  Risk Metrics":
        render_risk_metrics(data)

    elif page == "🧮 Factor Analysis":
        render_factor_analysis(data)

    elif page == "💼 Portfolio":
        render_portfolio(data)

    elif page == "🚨 Stress Tests":
        render_stress_tests(data)

    elif page == "📋 Signal Monitor":
        render_signal_monitor(data)


if __name__ == "__main__":
    main()