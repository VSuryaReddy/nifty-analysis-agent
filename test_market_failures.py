
import unittest
from unittest.mock import patch

import pandas as pd

import market_data


def make_history(values):
    dates = pd.date_range(
        "2026-11-10",
        periods=len(values),
        freq="D",
        tz="Asia/Kolkata",
    )
    return pd.DataFrame(
        {"Close": values},
        index=dates,
    )


class MarketFailureTests(unittest.TestCase):

    def test_empty_download_rejected(self):
        with patch("market_data.yf.Ticker") as ticker:
            ticker.return_value.history.return_value = (
                pd.DataFrame()
            )

            with self.assertRaisesRegex(
                ValueError, "No closing-price data"
            ):
                market_data.fetch_history("^NSEI")

    def test_zero_price_rejected(self):
        with patch("market_data.yf.Ticker") as ticker:
            ticker.return_value.history.return_value = (
                make_history([25000, 0])
            )

            with self.assertRaisesRegex(
                ValueError, "Invalid closing price"
            ):
                market_data.fetch_history("^NSEI")

    def test_negative_price_rejected(self):
        with patch("market_data.yf.Ticker") as ticker:
            ticker.return_value.history.return_value = (
                make_history([25000, -100])
            )

            with self.assertRaisesRegex(
                ValueError, "Invalid closing price"
            ):
                market_data.fetch_history("^NSEI")

    def test_infinite_price_rejected(self):
        with patch("market_data.yf.Ticker") as ticker:
            ticker.return_value.history.return_value = (
                make_history([25000, float("inf")])
            )

            with self.assertRaisesRegex(
                ValueError, "Invalid closing price"
            ):
                market_data.fetch_history("^NSEI")

    def test_secondary_market_failure_recorded(self):
        original_markets = market_data.MARKETS

        try:
            market_data.MARKETS = {
                "brent_crude": "BZ=F"
            }

            with patch(
                "market_data.fetch_history",
                side_effect=RuntimeError(
                    "Simulated download failure"
                ),
            ):
                result = (
                    market_data.fetch_market_snapshot()
                )

            self.assertNotIn(
                "brent_crude",
                result["markets"],
            )

            self.assertIn(
                "brent_crude",
                result["errors"],
            )

        finally:
            market_data.MARKETS = original_markets


if __name__ == "__main__":
    unittest.main(verbosity=2)
