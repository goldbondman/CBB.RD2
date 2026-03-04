from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("validate_predictions_freshness.py")


class ValidatePredictionsFreshnessSmokeTest(unittest.TestCase):
    def _write_predictions(self, data_dir: Path, generated_at: datetime, game_times: list[datetime]) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "predictions_mc_latest.csv"
        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["generated_at_utc", "game_datetime_utc"])
            writer.writeheader()
            for game_time in game_times:
                writer.writerow(
                    {
                        "generated_at_utc": generated_at.isoformat(),
                        "game_datetime_utc": game_time.isoformat(),
                    }
                )

    def _run_validator(self, data_dir: Path, max_age_hours: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--data-dir",
                str(data_dir),
                "--max-age-hours",
                str(max_age_hours),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_smoke_cases(self) -> None:
        now = datetime.now(timezone.utc)

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # case A: fresh file, only past games -> SKIP (exit 0)
            self._write_predictions(
                data_dir,
                generated_at=now - timedelta(hours=1),
                game_times=[now - timedelta(days=2), now - timedelta(days=1)],
            )
            case_a = self._run_validator(data_dir, max_age_hours=999)
            self.assertEqual(case_a.returncode, 0, msg=case_a.stdout + case_a.stderr)
            self.assertIn("result: SKIP", case_a.stdout)

            # case B: fresh file, has future games -> PASS (exit 0)
            self._write_predictions(
                data_dir,
                generated_at=now - timedelta(hours=1),
                game_times=[now - timedelta(hours=2), now + timedelta(days=1)],
            )
            case_b = self._run_validator(data_dir, max_age_hours=999)
            self.assertEqual(case_b.returncode, 0, msg=case_b.stdout + case_b.stderr)
            self.assertIn("result: PASS", case_b.stdout)

            # case C: stale file -> FAIL (exit 2)
            self._write_predictions(
                data_dir,
                generated_at=now - timedelta(hours=72),
                game_times=[now + timedelta(days=1)],
            )
            case_c = self._run_validator(data_dir, max_age_hours=1)
            self.assertEqual(case_c.returncode, 2, msg=case_c.stdout + case_c.stderr)
            self.assertIn("stale predictions", case_c.stdout)


if __name__ == "__main__":
    unittest.main()
