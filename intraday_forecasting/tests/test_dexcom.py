from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from src.dexcom import AUTH, LOGIN, READ, DexcomError, ShareClient, collect_once, extract_id, parse_readings
from src.journal import Journal

ACCOUNT = "11111111-1111-4111-8111-111111111111"
SESSION = "22222222-2222-4222-8222-222222222222"


class DexcomTests(unittest.TestCase):
    def test_old_and_new_id_response_formats(self):
        self.assertEqual(extract_id(ACCOUNT, "accountId"), ACCOUNT)
        self.assertEqual(extract_id({"sessionId": SESSION}, "sessionId"), SESSION)
        with self.assertRaises(DexcomError):
            extract_id("00000000-0000-0000-0000-000000000000", "accountId")

    def test_wt_epoch_is_utc_and_readings_are_sorted(self):
        rows = [{"WT": "/Date(1750000300000+0200)/", "Value": 120}, {"WT": "Date(1750000000000)", "Value": 100}]
        data = parse_readings(rows)
        self.assertEqual(data[0]["glucose"], 100)
        self.assertEqual(data[0]["observed_at"].timestamp(), 1750000000)
        self.assertEqual((data[1]["observed_at"] - data[0]["observed_at"]).total_seconds(), 300)

    def test_invalid_readings_not_replaced_by_zero(self):
        for rows in [{}, [{"WT": "bad", "Value": 100}], [{"WT": "Date(1750000000000)", "Value": 0}]]:
            with self.assertRaises(DexcomError):
                parse_readings(rows)
        self.assertEqual(parse_readings([]), [])

    def test_session_expiry_reauthenticates_once(self):
        calls, reads = [], []
        def transport(endpoint, body):
            calls.append(endpoint)
            if endpoint == AUTH:
                return {"accountId": ACCOUNT}
            if endpoint == LOGIN:
                return SESSION
            if endpoint == READ:
                reads.append(body)
                if len(reads) == 1:
                    raise DexcomError("expired", session_expired=True)
                return []
        client = ShareClient("fixture", "fixture", transport)
        self.assertEqual(client.readings(), [])
        self.assertEqual(calls, [AUTH, LOGIN, READ, AUTH, LOGIN, READ])
        self.assertEqual(reads[0]["minutes"], 1440)
        self.assertEqual(reads[0]["maxCount"], 288)

    def test_network_failure_does_not_trigger_login_storm(self):
        calls = []
        def transport(endpoint, body):
            calls.append(endpoint)
            if endpoint == AUTH:
                return ACCOUNT
            if endpoint == LOGIN:
                return SESSION
            raise DexcomError("offline")
        client = ShareClient("fixture", "fixture", transport)
        with self.assertRaises(DexcomError):
            client.readings()
        self.assertEqual(calls, [AUTH, LOGIN, READ])

    def test_rejected_credentials_are_not_retried_by_client(self):
        calls = []
        def transport(endpoint, body):
            calls.append(endpoint)
            raise DexcomError("authentication rejected", permanent=True)
        with self.assertRaises(DexcomError) as caught:
            ShareClient("fixture", "fixture", transport).readings()
        self.assertTrue(caught.exception.permanent)
        self.assertEqual(calls, [AUTH])

    def test_poll_deduplicates_and_retains_first_arrival(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "events.sqlite", clock=lambda: datetime.fromtimestamp(1750000060, timezone.utc))
            class FakeClient:
                def readings(self):
                    return parse_readings([{"WT": "Date(1750000000000)", "Value": 110}])
            self.assertEqual(collect_once(FakeClient(), journal)["inserted"], 1)
            self.assertEqual(collect_once(FakeClient(), journal)["inserted"], 0)
            event = journal.snapshot()[0]
            self.assertEqual((datetime.fromisoformat(event["available_at"]) - datetime.fromisoformat(event["event_time"])).total_seconds(), 60)
            with self.assertRaises(ValueError):
                journal.ingest_glucose(120, event["event_time"])
            self.assertEqual(journal.snapshot()[0]["amount"], 110)
            journal.close()


if __name__ == "__main__":
    unittest.main()
