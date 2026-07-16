import os
import pandas as pd
import numpy as np
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from logger import get_logger
from reporting.report_data import load_report_data
from reporting.charts import (
    plot_equity_curve, plot_rolling_risk,
    plot_ic_series, plot_stress_tests,
    plot_monthly_returns_heatmap, save_fig
)

logger = get_logger("report_generator")


def _fmt_pct(val, decimals=2):
    try:
        return f"{float(val):.{decimals}%}"
    except Exception:
        return "N/A"


def _fmt_num(val, decimals=3):
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return "N/A"


def generate_charts(data: dict, chart_dir: str = "reporting/charts/") -> dict:

    os.makedirs(chart_dir, exist_ok=True)
    charts = {}

    fig = plot_equity_curve(
        equity_curve=data["equity_curve"],
        oos_returns=data["oos_returns"],
        save_path=os.path.join(chart_dir, "equity_curve.png")
    )
    charts["equity"] = os.path.join(chart_dir, "equity_curve.png")

    fig = plot_rolling_risk(
        rolling_risk=data["rolling_risk"],
        save_path=os.path.join(chart_dir, "rolling_risk.png")
    )
    charts["rolling_risk"] = os.path.join(chart_dir, "rolling_risk.png")

    fig = plot_monthly_returns_heatmap(
        oos_returns=data["oos_returns"],
        save_path=os.path.join(chart_dir, "monthly_heatmap.png")
    )
    charts["monthly"] = os.path.join(chart_dir, "monthly_heatmap.png")

    fig = plot_ic_series(
        ic_series=data["ic_series"],
        save_path=os.path.join(chart_dir, "ic_series.png")
    )
    charts["ic"] = os.path.join(chart_dir, "ic_series.png")

    fig = plot_stress_tests(
        stress_tests=data["stress_tests"],
        save_path=os.path.join(chart_dir, "stress_tests.png")
    )
    charts["stress"] = os.path.join(chart_dir, "stress_tests.png")

    logger.info(f"All charts saved to {chart_dir}")
    return charts


def build_template_context(data: dict, charts: dict) -> dict:

    metrics = data.get("overall_metrics", pd.Series(dtype=float))
    risk    = data.get("risk_report",     pd.Series(dtype=float))
    tail    = data.get("tail_risk",       pd.Series(dtype=float))
    state   = data.get("pipeline_state",  {})

    def m(key):
        try:
            return float(metrics[key])
        except Exception:
            return np.nan

    def r(key):
        try:
            return float(risk[key])
        except Exception:
            return np.nan

    def t(key):
        try:
            return float(tail[key])
        except Exception:
            return np.nan

    stress_table = []
    st = data.get("stress_tests", pd.DataFrame())
    if not st.empty:
        for period, row in st.iterrows():
            stress_table.append({
                "period"          : period,
                "total_return"    : _fmt_pct(row.get("total_return", np.nan)),
                "total_return_raw": float(row.get("total_return", 0)),
                "max_drawdown"    : _fmt_pct(row.get("max_drawdown", np.nan)),
                "daily_vol"       : _fmt_pct(row.get("daily_vol", np.nan)),
                "worst_day"       : _fmt_pct(row.get("worst_day", np.nan)),
                "n_days"          : int(row.get("n_days", 0))
            })

    drawdown_table = []
    dd = data.get("drawdown_analysis", pd.DataFrame())
    if not dd.empty and "depth" in dd.columns:
        top5 = dd.nsmallest(5, "depth")
        for _, row in top5.iterrows():
            drawdown_table.append({
                "start"   : str(row.get("start", ""))[:10],
                "trough"  : str(row.get("trough", ""))[:10],
                "end"     : str(row.get("end", ""))[:10],
                "depth"   : _fmt_pct(row.get("depth", np.nan)),
                "duration": int(row.get("duration", 0)),
                "recovery": int(row.get("recovery", 0))
            })

    context = {
        "report_title"  : "Factor Research Engine — Strategy Report",
        "strategy_name" : "NSE Nifty 500 Long/Short Factor Strategy",
        "generated_at"  : datetime.now().strftime("%Y-%m-%d %H:%M IST"),

        "pipeline_status": state.get("status", "UNKNOWN"),
        "run_date"       : state.get("run_date", "N/A"),
        "universe_size"  : state.get("universe_size", "N/A"),
        "n_longs"        : state.get("n_longs", "N/A"),
        "n_shorts"       : state.get("n_shorts", "N/A"),
        "market_regime"  : state.get("market_regime", "N/A"),

        "cagr"        : _fmt_pct(m("cagr")),
        "cagr_pos"    : m("cagr") >= 0,
        "sharpe"      : _fmt_num(m("sharpe")),
        "sharpe_pos"  : m("sharpe") >= 1.0,
        "sortino"     : _fmt_num(m("sortino")),
        "max_drawdown": _fmt_pct(m("max_drawdown")),
        "calmar"      : _fmt_num(m("calmar")),
        "win_rate"    : _fmt_pct(m("win_rate")),

        "var_95"     : _fmt_pct(r("hist_var_95")),
        "var_99"     : _fmt_pct(r("hist_var_99")),
        "cvar_95"    : _fmt_pct(r("cvar_95")),
        "skewness"   : _fmt_num(t("skewness")),
        "excess_kurt": _fmt_num(t("excess_kurt")),
        "ulcer_index": _fmt_num(t("ulcer_index"), decimals=5),
        "omega_ratio": _fmt_num(t("omega_ratio")),

        "chart_equity"      : os.path.abspath(charts.get("equity", "")),
        "chart_rolling_risk": os.path.abspath(charts.get("rolling_risk", "")),
        "chart_monthly"     : os.path.abspath(charts.get("monthly", "")),
        "chart_ic"          : os.path.abspath(charts.get("ic", "")),
        "chart_stress"      : os.path.abspath(charts.get("stress", "")),

        "stress_table"  : stress_table,
        "drawdown_table": drawdown_table
    }

    return context


def generate_html_report(context: dict, output_path: str = "reporting/report.html") -> str:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    template_dir = os.path.join(
        os.path.dirname(__file__), "templates"
    )
    env      = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")

    html = template.render(**context)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved: {output_path}")
    return output_path


def generate_pdf_report(html_path: str, output_path: str = "reporting/report.pdf") -> str:

    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(output_path)
        logger.info(f"PDF report saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        logger.info("HTML report is still available")
        return html_path


def run_reporting(save_dir: str = "reporting/") -> dict:

    os.makedirs(save_dir, exist_ok=True)

    logger.info("Starting report generation...")

    data = load_report_data()

    charts = generate_charts(
        data, chart_dir=os.path.join(save_dir, "charts/")
    )

    context = build_template_context(data, charts)

    date_str    = datetime.now().strftime("%Y-%m-%d")
    html_path   = os.path.join(save_dir, f"report_{date_str}.html")
    html_output = generate_html_report(context, html_path)

    pdf_path   = os.path.join(save_dir, f"report_{date_str}.pdf")
    pdf_output = generate_pdf_report(html_output, pdf_path)

    logger.info(
        f"Reporting complete | "
        f"HTML: {html_output} | PDF: {pdf_output}"
    )

    return {
        "html_path": html_output,
        "pdf_path" : pdf_output,
        "charts"   : charts
    }

