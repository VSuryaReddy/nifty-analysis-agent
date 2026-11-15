
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import os

import main


class PipelineFailureTests(unittest.TestCase):

    def test_missing_brent_does_not_stop_pipeline(self):
        snapshot = {
            "collected_at_utc": "2026-11-15T13:00:00+00:00",
            "markets": {
                "nifty_50": {
                    "symbol": "^NSEI",
                    "as_of": "2026-11-13",
                    "close": 25144.30,
                    "daily_change_pct": -0.61,
                    "sma20": 23642.28,
                    "sma50": 23331.79,
                    "rsi14": 75.99,
                    "five_session_return_pct": 2.48,
                }
            },
            "errors": {
                "brent_crude": "Simulated download failure"
            },
        }

        news = {
            "article_count": 1,
            "errors": [],
        }

        grouped = {"events": [], "unassigned_articles": []}
        quality = {"accepted_events": []}
        facts = {"assessed_events": []}

        analysis = {
            "market_summary": "Test analysis"
        }

        with TemporaryDirectory() as directory:
            old_directory = os.getcwd()

            try:
                os.chdir(directory)

                with (
                    patch(
                        "main.fetch_market_snapshot",
                        return_value=snapshot,
                    ),
                    patch(
                        "main.fetch_news",
                        return_value=news,
                    ),
                    patch(
                        "main.group_news_into_events",
                        return_value=grouped,
                    ),
                    patch(
                        "main.assess_event_quality",
                        return_value=quality,
                    ),
                    patch(
                        "main.verify_event_facts",
                        return_value=facts,
                    ),
                    patch(
                        "main.assess_headline_agreement",
                        return_value=facts,
                    ),
                    patch(
                        "main.apply_evidence_assessment",
                        return_value=facts,
                    ),
                    patch(
                        "main.analyze_market",
                        return_value=analysis,
                    ) as mock_analyst,
                ):
                    main.main()

                mock_analyst.assert_called_once()

                passed_market_data = (
                    mock_analyst.call_args.args[0]
                )

                self.assertIn(
                    "nifty_50",
                    passed_market_data["markets"],
                )

                self.assertNotIn(
                    "brent_crude",
                    passed_market_data["markets"],
                )

                self.assertIn(
                    "brent_crude",
                    passed_market_data["errors"],
                )

                reports = list(
                    Path("reports").glob("*.json")
                )

                self.assertEqual(len(reports), 1)

            finally:
                os.chdir(old_directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
