"""notify.py: deliver alerts to the security desk (webhooks to VMS/PSIM/SOC, e-mail with snapshot).

Delivery runs in a background thread with a queue and retries, so a slow mail server or a down webhook never
stalls video analysis. Failures are logged; the event stays visible in the operator console regardless.
"""
import json
import logging
import queue
import smtplib
import threading
import time
from email.message import EmailMessage

import httpx

log = logging.getLogger("weapons.notify")


class Notifier:
    def __init__(self, cfg, site_name):
        self.cfg, self.site = cfg, site_name
        self.q = queue.Queue(maxsize=1000)
        self.http = httpx.Client(timeout=10)
        self.thread = threading.Thread(target=self._run, name="notifier", daemon=True)
        self.thread.start()

    def send(self, payload, snapshot_path=None, escalation=False):
        try:
            self.q.put_nowait((payload, snapshot_path, escalation))
        except queue.Full:
            log.error("notification queue full: dropping %s", payload.get("id"))

    def _run(self):
        while True:
            payload, snap, escalation = self.q.get()
            urls = [u for u in self.cfg.get("escalation_webhooks" if escalation else "webhooks", []) if u]
            for url in urls:
                self._retry(lambda: self.http.post(url, json=payload).raise_for_status(), f"webhook {url}")
            email = self.cfg.get("email", {})
            if email.get("enabled"):
                self._retry(lambda: self._email(email, payload, snap), "email")

    @staticmethod
    def _retry(fn, what, attempts=3):
        for i in range(attempts):
            try:
                fn()
                return True
            except Exception as exc:
                log.warning("%s failed (%d/%d): %s", what, i + 1, attempts, exc)
                time.sleep(2 ** i)
        return False

    def _email(self, cfg, payload, snap):
        msg = EmailMessage()
        prefix = "ESCALATION - " if payload.get("escalation") else ""
        msg["Subject"] = f"{prefix}[{self.site}] {payload['label'].upper()} detected - {payload['camera_name']}"
        msg["From"], msg["To"] = cfg["sender"], ", ".join(cfg["recipients"])
        msg.set_content(
            f"تنبيه أمني / Security alert\n\nSite: {self.site}\nCamera: {payload['camera_name']} ({payload['camera']})\n"
            f"Object: {payload['label']} (score {payload['score']:.2f})\nTime: {payload['time']}\nEvent: {payload['id']}\n\n"
            "Verify on the operator console before acting. This is an automated detection.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))
        if snap:
            with open(snap, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=f"{payload['id']}.jpg")
        with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"]), timeout=20) as s:
            s.starttls()
            if cfg.get("username"):
                s.login(cfg["username"], cfg["password"])
            s.send_message(msg)
