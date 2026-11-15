
import unittest
from datetime import datetime, timezone

import pandas as pd

from market_data import (
    select_eligible_history,
    validate_freshness,
)


class MarketValidationTests(unittest.TestCase):

    def test_sunday_usdinr_excluded(self):
        dates = pd.to_datetime([
            "2026-11-12",
            "2026-11-13",
            "2026-11-15",
        ]).tz_localize("Europe/London")

        closes = pd.Series(
            [93.70, 94.03, 94.24],
            index=dates,
        )

        result = select_eligible_history(
            "usd_inr", closes
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            str(result.index[-1].date()),
            "2026-11-13",
        )
        self.assertAlmostEqual(
            float(result.iloc[-1]), 94.03
        )

    def test_recent_friday_accepted_on_sunday(self):
        dates = pd.to_datetime([
            "2026-11-12",
            "2026-11-13",
        ]).tz_localize("Asia/Kolkata")

        closes = pd.Series(
            [25299.35, 25144.30],
            index=dates,
        )

        now = datetime(
            2026, 11, 15, 13, 30,
            tzinfo=timezone.utc,
        )

        self.assertEqual(
            validate_freshness(closes, now),
            2,
        )

    def test_stale_observation_rejected(self):
        dates = pd.to_datetime([
            "2026-11-05",
            "2026-11-06",
        ]).tz_localize("Asia/Kolkata")

        closes = pd.Series(
            [25000, 25100],
            index=dates,
        )

        now = datetime(
            2026, 11, 15, 13, 30,
            tzinfo=timezone.utc,
        )

        with self.assertRaisesRegex(
            ValueError, "Stale observation"
        ):
            validate_freshness(closes, now)

    def test_future_observation_rejected(self):
        dates = pd.to_datetime([
            "2026-11-15",
            "2026-11-17",
        ]).tz_localize("Asia/Kolkata")

        closes = pd.Series(
            [25000, 25100],
            index=dates,
        )

        now = datetime(
            2026, 11, 15, 13, 30,
            tzinfo=timezone.utc,
        )

        with self.assertRaisesRegex(
            ValueError, "Future observation"
        ):
            validate_freshness(closes, now)


if __name__ == "__main__":
    unittest.main(verbosity=2)
