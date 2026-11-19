
import unittest
from datetime import date

from analyst import validate_analysis


SOURCE = {
    "title": "Official meeting schedule",
    "url": "https://example.org/calendar"
}

NEWS = {
    "events": [
        {
            "claims": [
                {
                    "sources": [SOURCE]
                }
            ]
        }
    ]
}


def base_analysis(events):
    return {
        "market_summary": "Test",
        "important_events": [],
        "next_day": {
            "outlook": "Mixed",
            "up_probability": 35,
            "down_probability": 40,
            "sideways_probability": 25,
            "explanation": "Test"
        },
        "next_2_6_weeks": {
            "outlook": "Mixed",
            "bullish_probability": 35,
            "bearish_probability": 40,
            "sideways_probability": 25,
            "explanation": "Test"
        },
        "key_risks": [],
        "missing_information": [],
        "upcoming_high_impact_events": events
    }


def event(when, source=SOURCE):
    return {
        "event": "Policy meeting",
        "date": when,
        "why_watch": "Policy uncertainty",
        "sources": [source]
    }


class UpcomingEventsTests(unittest.TestCase):

    def test_yesterday_excluded_today_retained_future_retained(self):
        result = validate_analysis(
            base_analysis([
                event("2026-11-15"),
                event("2026-11-16"),
                event("2026-11-17")
            ]),
            NEWS,
            report_date=date(2026, 11, 16)
        )

        self.assertEqual(
            [
                e["date"]
                for e in result["upcoming_high_impact_events"]
            ],
            ["2026-11-16", "2026-11-17"]
        )

    def test_unqualified_source_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "unqualified source"
        ):
            validate_analysis(
                base_analysis([
                    event(
                        "2026-11-17",
                        {
                            "title": "Invented",
                            "url": "https://invalid.example"
                        }
                    )
                ]),
                NEWS,
                report_date=date(2026, 11, 16)
            )

    def test_invalid_date_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "invalid date"
        ):
            validate_analysis(
                base_analysis([
                    event("2026-11-31")
                ]),
                NEWS,
                report_date=date(2026, 11, 16)
            )

    def test_empty_watchlist_accepted(self):
        result = validate_analysis(
            base_analysis([]),
            NEWS,
            report_date=date(2026, 11, 16)
        )

        self.assertEqual(
            result["upcoming_high_impact_events"],
            []
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
