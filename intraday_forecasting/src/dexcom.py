"""Dexcom Share EU reader, based on the user's working ESP32 flow.

Credentials: DEXCOM_USER and DEXCOM_PASSWORD environment variables.
python -m src.dexcom --once          # fetch last 24 h once
python -m src.dexcom                 # poll every 60 seconds
All TLS certificates are verified. No credentials/session IDs are logged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import uuid

from src.journal import Journal

BASE = "https://shareous1.dexcom.com/ShareWebServices/Services/"
APP_ID = "d89443d2-327c-4a6f-89e5-496bbb0317db"
AUTH = "General/AuthenticatePublisherAccount"
LOGIN = "General/LoginPublisherAccountById"
READ = "Publisher/ReadPublisherLatestGlucoseValues"
SESSION_ERRORS = {"SessionIdNotFound", "SessionNotValid", "SessionIdInvalid"}


class DexcomError(RuntimeError):
    def __init__(self, message, session_expired=False, permanent=False):
        super().__init__(message)
        self.session_expired = session_expired
        self.permanent = permanent


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DexcomError("Unexpected Dexcom redirect")


def post(endpoint, body):
    if endpoint not in {AUTH, LOGIN, READ}:
        raise ValueError("Unsupported Dexcom endpoint")
    request = urllib.request.Request(BASE + endpoint, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "application/json", "Accept-Encoding": "identity"})
    # Default urllib HTTPS handler verifies certificates/hostnames; reject
    # redirects instead of forwarding credentials to another endpoint.
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=20) as response:
            value = json.load(response)
    except urllib.error.HTTPError as e:
        session_expired = e.code == 401
        permanent = endpoint in {AUTH, LOGIN} and e.code in {400, 401, 403}
        try:
            value = json.loads(e.read(8192))
            session_expired |= isinstance(value, dict) and value.get("Code") in SESSION_ERRORS
            permanent |= endpoint in {AUTH, LOGIN} and isinstance(value, dict) and bool(value.get("Code"))
        except (ValueError, UnicodeError):
            pass
        raise DexcomError(f"Dexcom HTTP {e.code}", session_expired, permanent) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise DexcomError("Dexcom connection failed") from None
    except (ValueError, UnicodeError):
        raise DexcomError("Malformed Dexcom JSON response") from None
    if isinstance(value, dict) and value.get("Code"):
        raise DexcomError("Dexcom returned an API error", value["Code"] in SESSION_ERRORS, endpoint in {AUTH, LOGIN})
    return value


def extract_id(value, key):
    candidate = value if isinstance(value, str) else value.get(key) if isinstance(value, dict) else None
    try:
        parsed = uuid.UUID(candidate)
        if parsed.int == 0:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise DexcomError("Dexcom returned an invalid account/session ID") from None
    return str(parsed)


def parse_readings(value):
    if not isinstance(value, list):
        raise DexcomError("Expected a Dexcom readings array")
    readings = []
    for row in value:
        if not isinstance(row, dict):
            raise DexcomError("Malformed Dexcom reading")
        match = re.fullmatch(r"/?Date\((\d+)(?:[+-]\d{4})?\)/?", str(row.get("WT", "")))
        amount = row.get("Value")
        if not match or isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
            raise DexcomError("Invalid Dexcom WT timestamp or glucose value")
        try:
            observed = datetime.fromtimestamp(int(match[1]) / 1000, tz=timezone.utc)
        except (OverflowError, ValueError, OSError):
            raise DexcomError("Dexcom timestamp is out of range") from None
        readings.append({"observed_at": observed, "glucose": float(amount), "trend": row.get("Trend")})
    return sorted(readings, key=lambda r: r["observed_at"])


class ShareClient:
    def __init__(self, username, password, transport=post):
        if not username or not password:
            raise ValueError("Set DEXCOM_USER and DEXCOM_PASSWORD locally")
        self.username, self.password = username, password
        self.transport = transport
        self.session_id = None

    def login(self):
        account = extract_id(self.transport(AUTH, {"accountName": self.username, "password": self.password, "applicationId": APP_ID}), "accountId")
        self.session_id = extract_id(self.transport(LOGIN, {"accountId": account, "password": self.password, "applicationId": APP_ID}), "sessionId")

    def readings(self, minutes=1440, max_count=288):
        if not 1 <= minutes <= 1440 or not 1 <= max_count <= 288:
            raise ValueError("Share history is limited to 1440 minutes/288 readings per request")
        if not self.session_id:
            self.login()
        for attempt in range(2):
            try:
                return parse_readings(self.transport(READ, {"sessionId": self.session_id, "minutes": minutes, "maxCount": max_count}))
            except DexcomError as e:
                if attempt or not e.session_expired:
                    raise
                self.session_id = None
                self.login()


def collect_once(client, journal):
    readings = client.readings()
    inserted, conflicts = 0, 0
    for reading in readings:
        try:
            event_id = journal.ingest_glucose(reading["glucose"], reading["observed_at"])
        except ValueError:
            # Keep previous data intact; do not silently overwrite a conflicting
            # sample or accept a future reading from a clock error.
            conflicts += 1
            continue
        if event_id:
            inserted += 1
    return {"received": len(readings), "inserted": inserted, "rejected": conflicts,
            "latest_sample_age_seconds": (journal.clock() - readings[-1]["observed_at"]).total_seconds() if readings else None,
            "note": "First-seen age includes bootstrap/backfill; steady-state live latency must be analysed separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("Polling interval must be at least 60 seconds")
    try:
        client = ShareClient(os.environ.get("DEXCOM_USER"), os.environ.get("DEXCOM_PASSWORD"))
    except ValueError as e:
        parser.error(str(e))
    journal = Journal(args.db)
    delay = args.interval
    try:
        while True:
            try:
                print(json.dumps(collect_once(client, journal)), flush=True)
                delay = args.interval
            except DexcomError as e:
                print(json.dumps({"error": str(e), "retry_seconds": delay}), flush=True)
                if args.once or e.permanent:
                    raise SystemExit(1)
                delay = min(delay * 2, 900)
            if args.once:
                break
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        journal.close()


if __name__ == "__main__":
    main()
