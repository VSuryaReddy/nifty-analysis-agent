
import unittest

from analyst import validate_analysis


class AnalystMissingDataTests(unittest.TestCase):

    def setUp(self):
        self.market_data = {
            "markets": {
                "nifty_50": {"close": 25144.30}
            },
            "errors": {
                "brent_crude": "Simulated failure"
            },
        }

        self.analysis = {
            "market_summary": "Nifty analysis",
            "important_events": [],
            "next_day": {
                "outlook": "Mixed",
                "up_probability": 35,
                "down_probability": 40,
                "sideways_probability": 25,
                "explanation": "Mixed conditions",
            },
            "next_2_6_weeks": {
                "outlook": "Mixed",
                "bullish_probability": 40,
                "bearish_probability": 40,
                "sideways_probability": 20,
                "explanation": "Mixed conditions",
            },
            "key_risks": [],
            "missing_information": [],
        }

        self.qualified_news = {"events": []}

    def test_missing_brent_must_be_acknowledged(self):
        with self.assertRaisesRegex(
            ValueError, "brent_crude"
        ):
            validate_analysis(
                self.analysis,
                self.qualified_news,
                self.market_data,
            )

    def test_acknowledged_missing_brent_passes(self):
        self.analysis["missing_information"] = [
            "Brent crude market data unavailable"
        ]

        result = validate_analysis(
            self.analysis,
            self.qualified_news,
            self.market_data,
        )

        self.assertEqual(
            result["missing_information"],
            ["Brent crude market data unavailable"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
