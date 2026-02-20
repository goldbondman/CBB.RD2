#!/usr/bin/env python3
"""
send_email.py — Shared email sender for CBB Analytics pipeline.

Usage:
    python send_email.py \
        --subject "CBB Results Mar 15" \
        --body-file /tmp/email_body.txt \
        --attachments data/results_log.csv data/results_summary.csv \
        --critical-count 2

Environment variables:
    GMAIL_ADDRESS       Sender Gmail address
    GMAIL_APP_PASSWORD  Google App Password
    NOTIFY_EMAIL        Recipient address

Exit codes: 0 = success or skipped (missing secrets), 1 = SMTP failure
"""

import argparse
import logging
import os
import smtplib
import sys
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def build_subject(base: str, critical_count: int) -> str:
    if critical_count > 0:
        return f"🚨 {base}"
    return base


def send(sender: str, password: str, recipient: str,
         subject: str, body: str, attachments: list) -> None:
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    attached = 0
    for path in attachments:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 10:
            log.warning(f"Skipping attachment (missing/empty): {p}")
            continue
        with open(p, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{p.name}"')
        msg.attach(part)
        attached += 1

    log.info(f"Sending to {recipient} ({attached} attachment(s))...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
    log.info(f"Sent: {subject}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send CBB analytics email report")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--attachments", nargs="*", default=[])
    parser.add_argument("--critical-count", type=int, default=0)
    args = parser.parse_args()

    sender = os.environ.get("GMAIL_ADDRESS", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    recipient = os.environ.get("NOTIFY_EMAIL", "").strip()

    if not sender or not password:
        log.info("[SKIP] Email secrets not configured")
        return 0
    if not recipient:
        log.warning("[SKIP] NOTIFY_EMAIL not set")
        return 0

    body_path = Path(args.body_file)
    if not body_path.exists():
        log.error(f"Body file not found: {body_path}")
        return 1

    subject = build_subject(args.subject, args.critical_count)
    try:
        send(sender, password, recipient, subject,
             body_path.read_text(), args.attachments)
        return 0
    except Exception as exc:
        log.error(f"SMTP failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
