"""Email delivery for the daily digest, via plain SMTP -- works with Gmail
(with an app password), Outlook, or any other SMTP provider. WhatsApp is a
documented follow-up in PLAN.md once a Twilio/WhatsApp Business account exists.

Configure with environment variables:
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, DIGEST_TO_EMAIL

Nothing in this module sends anything on import -- it only sends when
send_digest_email() is explicitly called, and raises a clear error rather
than silently no-op'ing if it isn't configured.
"""

import os
import smtplib
from email.mime.text import MIMEText

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
DIGEST_TO_EMAIL = os.environ.get("DIGEST_TO_EMAIL")


def send_digest_email(subject: str, body: str, to_email: str | None = None) -> None:
    missing = [n for n, v in [("SMTP_HOST", SMTP_HOST), ("SMTP_USER", SMTP_USER), ("SMTP_PASSWORD", SMTP_PASSWORD)] if not v]
    if missing:
        raise RuntimeError(
            f"Email isn't configured -- missing environment variable(s): {', '.join(missing)}."
        )

    recipient = to_email or DIGEST_TO_EMAIL or SMTP_USER
    if not recipient:
        raise RuntimeError("No recipient set -- pass to_email or set DIGEST_TO_EMAIL.")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = recipient

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
