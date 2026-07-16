from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from scheduler.pipeline_runner import run_daily_pipeline
from dashboard.app import step_launch_dashboard
from logger import get_logger
import pytz

logger = get_logger("scheduler")


def start_scheduler():

    scheduler = BlockingScheduler(
        timezone=pytz.timezone("Asia/Kolkata")
    )

    scheduler.add_job(
        func=run_daily_pipeline,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=15,
            timezone=pytz.timezone("Asia/Kolkata")
        ),
        id="daily_pipeline",
        name="Factor Research Engine — Daily Run",
        misfire_grace_time=300,  
        coalesce=True             
    )

    logger.info(
        "Scheduler started | "
        "Daily pipeline: Mon–Fri 16:15 IST"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()

def step_launch_dashboard() -> None:
    logger.info(
        "Pipeline complete. Launch dashboard with: "
        "streamlit run dashboard/app.py"
    )

step_launch_dashboard()
