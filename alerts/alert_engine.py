import smtplib
import requests
import pandas as pd
import numpy as np
import yaml
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from logger import get_logger

logger = get_logger("alert_engine")

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)
    
def check_drawdown_breach(equity_curve: pd.Series, threshold: float = -0.10) -> dict:

    if equity_curve.empty:
        return {"breach": False}

    rolling_max      = equity_curve.cummax()
    current_drawdown = (
        equity_curve.iloc[-1] - rolling_max.iloc[-1]
    ) / rolling_max.iloc[-1]

    breach   = current_drawdown < threshold
    severity = (
        "CRITICAL" if current_drawdown < threshold * 2
        else "WARNING"
    )

    if breach:
        logger.warning(
            f"DRAWDOWN BREACH | "
            f"Current: {current_drawdown:.2%} | "
            f"Threshold: {threshold:.2%} | "
            f"Severity: {severity}"
        )

    return {
        "breach"           : breach,
        "current_drawdown" : current_drawdown,
        "threshold"        : threshold,
        "severity"         : severity if breach else None
    }

def check_var_breach(oos_returns: pd.Series, var_95: float, multiplier: float = 1.5) -> dict:

    if oos_returns.empty or np.isnan(var_95):
        return {"breach": False}

    today_return  = oos_returns.iloc[-1]
    breach_level  = var_95 * multiplier
    breach        = today_return < breach_level

    if breach:
        logger.warning(
            f"VaR BREACH | "
            f"Today: {today_return:.4f} | "
            f"VaR: {var_95:.4f} | "
            f"Breach level: {breach_level:.4f}"
        )

    return {
        "breach"       : breach,
        "today_return" : today_return,
        "var_95"       : var_95,
        "breach_level" : breach_level,
        "excess"       : today_return - breach_level if breach else 0
    }

def check_vol_spike(vol_forecasts: pd.Series, threshold: float = 0.30) -> dict:

    if vol_forecasts.empty:
        return {"spike": False}

    current_vol = vol_forecasts.mean()
    spike       = current_vol > threshold

    if spike:
        logger.warning(
            f"VOL SPIKE | "
            f"Current vol: {current_vol:.2%} | "
            f"Threshold: {threshold:.2%}"
        )

    return {
        "spike"      : spike,
        "current_vol": current_vol,
        "threshold"  : threshold
    }

def check_rebalance_complete(pipeline_state: dict) -> dict:

    status   = pipeline_state.get("status", "UNKNOWN")
    complete = status == "COMPLETE"

    return {
        "complete"  : complete,
        "status"    : status,
        "n_longs"   : pipeline_state.get("n_longs", 0),
        "n_shorts"  : pipeline_state.get("n_shorts", 0),
        "run_date"  : pipeline_state.get("run_date", "N/A"),
        "regime"    : pipeline_state.get("market_regime", "N/A")
    }

def send_email_alert(subject: str, body_html: str, attachment_path: str = None) -> bool:

    config = load_config()
    cfg    = config.get("alerts", {}).get("email", {})

    if not cfg.get("enabled", False):
        logger.info("Email alerts disabled in config")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg["sender"]
        msg["To"]      = cfg["recipient"]

        msg.attach(MIMEText(body_html, "html"))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(attachment_path)}"
            )
            msg.attach(part)

        with smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"]) as server:
            server.starttls()
            server.login(cfg["sender"], cfg["app_password"])
            server.sendmail(
                cfg["sender"], cfg["recipient"], msg.as_string()
            )

        logger.info(f"Email sent: {subject}")
        return True

    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False


def send_telegram_alert(message: str) -> bool:

    config = load_config()
    cfg    = config.get("alerts", {}).get("telegram", {})

    if not cfg.get("enabled", False):
        logger.info("Telegram alerts disabled in config")
        return False

    try:
        url     = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
        payload = {
            "chat_id"   : cfg["chat_id"],
            "text"      : message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            logger.info(f"Telegram sent: {message[:60]}...")
            return True
        else:
            logger.error(
                f"Telegram failed: {response.status_code} "
                f"{response.text}"
            )
            return False

    except Exception as e:
        logger.error(f"Telegram failed: {e}")
        return False
    
def build_rebalance_email(rebalance_info: dict, metrics: dict) -> tuple:

    subject = (
        f"[Factor Engine] Rebalance Complete — "
        f"{rebalance_info['run_date']} | "
        f"Regime: {rebalance_info['regime']}"
    )

    body = f"""
    <html><body style="font-family: Arial, sans-serif; color: #2c3e50;">
    <div style="background:#1f4e79;color:white;padding:20px;
                border-radius:8px;margin-bottom:20px;">
      <h2 style="margin:0;">Daily Rebalance Complete</h2>
      <p style="margin:4px 0;opacity:0.85;">
        {rebalance_info['run_date']}
      </p>
    </div>

    <table style="width:100%;border-collapse:collapse;">
      <tr style="background:#f0f4f8;">
        <td style="padding:10px;font-weight:bold;">Status</td>
        <td style="padding:10px;">{rebalance_info['status']}</td>
      </tr>
      <tr>
        <td style="padding:10px;font-weight:bold;">Long Positions</td>
        <td style="padding:10px;color:#1e8449;">
          {rebalance_info['n_longs']}
        </td>
      </tr>
      <tr style="background:#f0f4f8;">
        <td style="padding:10px;font-weight:bold;">Short Positions</td>
        <td style="padding:10px;color:#922b21;">
          {rebalance_info['n_shorts']}
        </td>
      </tr>
      <tr>
        <td style="padding:10px;font-weight:bold;">Market Regime</td>
        <td style="padding:10px;">{rebalance_info['regime']}</td>
      </tr>
      <tr style="background:#f0f4f8;">
        <td style="padding:10px;font-weight:bold;">CAGR</td>
        <td style="padding:10px;">
          {metrics.get('cagr', 'N/A')}
        </td>
      </tr>
      <tr>
        <td style="padding:10px;font-weight:bold;">Sharpe Ratio</td>
        <td style="padding:10px;">
          {metrics.get('sharpe', 'N/A')}
        </td>
      </tr>
    </table>

    <p style="margin-top:20px;color:#717d7e;font-size:12px;">
      Factor Research Engine — automated daily report
    </p>
    </body></html>
    """
    return subject, body


def build_risk_alert_email(alert_type: str, details: dict) -> tuple:

    severity_color = {
        "CRITICAL": "#922b21",
        "WARNING" : "#d4ac0d",
        "INFO"    : "#1e8449"
    }.get(details.get("severity", "WARNING"), "#717d7e")

    subject = (
        f"[ALERT] {alert_type} — "
        f"{details.get('severity', 'WARNING')} — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    rows = "".join([
        f"""<tr {'style="background:#f0f4f8;"' if i % 2 == 0 else ''}>
            <td style="padding:10px;font-weight:bold;">{k}</td>
            <td style="padding:10px;">{v}</td></tr>"""
        for i, (k, v) in enumerate(details.items())
        if k != "severity"
    ])

    body = f"""
    <html><body style="font-family: Arial, sans-serif; color: #2c3e50;">
    <div style="background:{severity_color};color:white;padding:20px; border-radius:8px;margin-bottom:20px;">
      <h2 style="margin:0;">⚠ Risk Alert: {alert_type}</h2>
      <p style="margin:4px 0;opacity:0.85;">
        {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
      </p>
    </div>
    <table style="width:100%;border-collapse:collapse;">
      {rows}
    </table>
    <p style="margin-top:20px;color:#717d7e;font-size:12px;">
      Factor Research Engine — automated risk monitoring
    </p>
    </body></html>
    """
    return subject, body


def build_telegram_message(alert_type: str, details: dict) -> str:

    emoji = {
        "Rebalance Complete": "✅",
        "Drawdown Breach"   : "🔴",
        "VaR Breach"        : "⚠️",
        "Vol Spike"         : "📈"
    }.get(alert_type, "ℹ️")

    lines = [f"{emoji} *{alert_type}*",
             f"_{datetime.now().strftime('%Y-%m-%d %H:%M IST')}_",
             ""]

    for k, v in details.items():
        if k == "severity":
            continue
        if isinstance(v, float):
            lines.append(f"• *{k}*: {v:.4f}")
        else:
            lines.append(f"• *{k}*: {v}")

    return "\n".join(lines)

def run_alerts(pipeline_state: dict,
        equity_curve: pd.Series,
        oos_returns: pd.Series,
        var_95: float,
        vol_forecasts: pd.Series,
        overall_metrics: dict,
        pdf_path: str = None) -> dict:
 
    config    = load_config()
    alert_cfg = config.get("alerts", {})
    results   = {}

    dd_check = check_drawdown_breach(
        equity_curve,
        threshold=alert_cfg.get("drawdown_threshold", -0.10)
    )
    results["drawdown"] = dd_check

    if dd_check["breach"]:
        details = {
            "severity"        : dd_check["severity"],
            "Current Drawdown": f"{dd_check['current_drawdown']:.2%}",
            "Threshold"       : f"{dd_check['threshold']:.2%}"
        }
        subj, body = build_risk_alert_email("Drawdown Breach", details)
        tg_msg     = build_telegram_message("Drawdown Breach", details)

        send_email_alert(subj, body)
        send_telegram_alert(tg_msg)

    var_check = check_var_breach(
        oos_returns, var_95,
        multiplier=alert_cfg.get("var_breach_multiplier", 1.5)
    )
    results["var"] = var_check

    if var_check["breach"]:
        details = {
            "severity"    : "WARNING",
            "Today Return": f"{var_check['today_return']:.4f}",
            "VaR 95%"     : f"{var_check['var_95']:.4f}",
            "Breach Level": f"{var_check['breach_level']:.4f}"
        }
        subj, body = build_risk_alert_email("VaR Breach", details)
        tg_msg     = build_telegram_message("VaR Breach", details)

        send_email_alert(subj, body)
        send_telegram_alert(tg_msg)

    vol_check = check_vol_spike(
        vol_forecasts,
        threshold=alert_cfg.get("vol_spike_threshold", 0.30)
    )
    results["vol_spike"] = vol_check

    if vol_check["spike"]:
        details = {
            "severity"   : "WARNING",
            "Current Vol": f"{vol_check['current_vol']:.2%}",
            "Threshold"  : f"{vol_check['threshold']:.2%}"
        }
        subj, body = build_risk_alert_email("Vol Spike", details)
        tg_msg     = build_telegram_message("Vol Spike", details)

        send_email_alert(subj, body)
        send_telegram_alert(tg_msg)

    rb_check = check_rebalance_complete(pipeline_state)
    results["rebalance"] = rb_check

    metrics_fmt = {
        "cagr"  : f"{overall_metrics.get('cagr', 0):.2%}",
        "sharpe": f"{overall_metrics.get('sharpe', 0):.3f}"
    }

    subj, body = build_rebalance_email(rb_check, metrics_fmt)
    tg_msg     = build_telegram_message("Rebalance Complete", {
        **rb_check, **metrics_fmt
    })

    send_email_alert(subj, body, attachment_path=pdf_path)
    send_telegram_alert(tg_msg)

    logger.info(
        f"Alerts complete | "
        f"DD breach: {dd_check['breach']} | "
        f"VaR breach: {var_check['breach']} | "
        f"Vol spike: {vol_check['spike']}"
    )

    return results

