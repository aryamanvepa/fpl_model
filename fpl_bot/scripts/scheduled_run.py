"""Entry point for the weekly, deadline-triggered automation.

Windows Task Scheduler runs this once a day; most days it's a fast no-op.
It only does real work (refresh data, run all 4 models + ensemble, diff
against held squads, queue transfers) once the next gameweek deadline is
within DEADLINE_WINDOW_HOURS, and only once per gameweek (tracked in the
digest_runs table) so a shifted or double deadline doesn't cause duplicate
runs or duplicate approval-queue spam.

Run manually with: python -m fpl_bot.scripts.scheduled_run
"""

import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fpl_bot import db
from fpl_bot.ingest import ingest_bootstrap

LOG_PATH = Path(__file__).parent.parent / "data" / "scheduler.log"
# Trigger once the next deadline is within this many hours. Checked on Mon/Wed:
# Wed->Sat 11:00 deadline is ~65h out, Mon->Fri 18:30 is ~101h out -- 84h comfortably
# catches the Wednesday-before-a-Saturday-deadline case without Monday ever firing
# on a normal week (which is fine, Monday is just a safety net for early deadlines).
DEADLINE_WINDOW_HOURS = 84


def _setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger


def _already_ran(gw: int) -> bool:
    conn = db.get_connection()
    try:
        return conn.execute("SELECT 1 FROM digest_runs WHERE gw = ?", (gw,)).fetchone() is not None
    finally:
        conn.close()


def _mark_ran(gw: int) -> None:
    conn = db.get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO digest_runs (gw, run_at) VALUES (?, ?)",
            (gw, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def run() -> None:
    log = _setup_logging()
    log.info("=" * 60)
    log.info("Scheduled check starting.")

    try:
        ingest_bootstrap.run()
    except Exception:
        log.exception("Data refresh failed -- aborting this run.")
        return

    conn = db.get_connection()
    try:
        row = conn.execute("SELECT id, deadline_time FROM gameweeks WHERE is_next = 1").fetchone()
        if row is None:
            row = conn.execute(
                "SELECT id, deadline_time FROM gameweeks WHERE finished = 0 ORDER BY id LIMIT 1"
            ).fetchone()
    finally:
        conn.close()

    if row is None:
        log.info("No upcoming gameweek found -- nothing to do.")
        return

    gw_id, deadline_str = row
    deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
    hours_until = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600

    if hours_until < 0:
        log.info(f"GW{gw_id} deadline already passed ({deadline_str}) -- waiting for the next gameweek to appear as 'next'.")
        return
    if hours_until > DEADLINE_WINDOW_HOURS:
        log.info(f"GW{gw_id} deadline is {hours_until:.0f}h away (window is {DEADLINE_WINDOW_HOURS}h) -- skipping for now.")
        return
    if _already_ran(gw_id):
        log.info(f"Already ran the full digest for GW{gw_id} -- skipping.")
        return

    log.info(f"GW{gw_id} deadline is {hours_until:.0f}h away -- running the full digest.")
    from fpl_bot.scripts.daily_digest import run as run_digest

    try:
        digest_text = run_digest(refresh=False)  # already refreshed above
        log.info("Digest completed:\n" + digest_text)
    except Exception:
        log.exception("Digest run failed.")
        return

    _mark_ran(gw_id)

    if os.environ.get("SMTP_HOST"):
        try:
            from fpl_bot.notify.email import send_digest_email

            send_digest_email(f"FPL Weekly Digest -- GW{gw_id}", digest_text)
            log.info("Emailed digest.")
        except Exception:
            log.exception("Email send failed (digest still ran and the approval queue is populated).")
    else:
        log.info("SMTP not configured -- skipping email. Digest is available via the dashboard or approve_pending CLI.")

    log.info("Scheduled run complete.")


if __name__ == "__main__":
    run()
