"""Focused offline integration tests for analyst.py upcoming-event watchlist.
Run from the project directory: python -m unittest -v test_upcoming_events_integration.py
"""
import copy
import unittest
from datetime import date
# Allow offline tests to run even where optional API packages are not installed.
import sys
import types
try:
    import dotenv
except ImportError:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda: None)
try:
    import openai
except ImportError:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=object)
from analyst import prepare_qualified_news, validate_analysis

SOURCE = {
    "id": 7,
    "title": "Central bank confirms policy decision scheduled for 18 November 2026",
    "url": "https://example.org/fictional-policy-calendar",
    "source": "Reserve Bank of India",
    "published_at_utc": "2026-11-16T05:00:00+00:00",
}


def fact_result():
    return {"assessed_events": [{
        "category": "india_macro",
        "relevance_assessment": {"relevance": "HIGH"},
        "sources": [copy.deepcopy(SOURCE)],
        "fact_assessment": {"claims": [{
            "claim": "A policy decision is scheduled for 18 November 2026.",
            "claim_type": "reported",
            "headline_grounding": "SUPPORTED_BY_HEADLINE",
            "supporting_article_ids": [7],
            "evidence_assessment": {"evidence_status": "CREDIBLE_SINGLE_SOURCE"},
        }]},
    }]}


def base_analysis(upcoming):
    return {
        "market_summary": "Controlled offline test",
        "important_events": [],
        "next_day": {"outlook": "Mixed", "up_probability": 35,
                     "down_probability": 40, "sideways_probability": 25,
                     "explanation": "Test baseline"},
        "next_2_6_weeks": {"outlook": "Mixed", "bullish_probability": 40,
                           "bearish_probability": 40, "sideways_probability": 20,
                           "explanation": "Test baseline"},
        "key_risks": [],
        "missing_information": [],
        "upcoming_high_impact_events": upcoming,
    }


def upcoming_event(day, source=None):
    s = SOURCE if source is None else source
    return {
        "event": "Scheduled policy decision (synthetic test event)",
        "date": day,
        "why_watch": "Policy decisions may affect interest-rate expectations",
        "sources": [{"title": s["title"], "url": s["url"]}],
    }


class UpcomingEventsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.qualified_news = prepare_qualified_news(fact_result(), {"markets": {}})
        self.report_date = date(2026, 11, 17)

    def validate(self, upcoming):
        return validate_analysis(
            base_analysis(upcoming), self.qualified_news,
            report_date=self.report_date,
        )

    def test_qualified_news_survives_preparation(self):
        events = self.qualified_news["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0]["claims"]), 1)
        self.assertEqual(events[0]["claims"][0]["sources"][0]["url"], SOURCE["url"])

    def test_today_and_future_retained_yesterday_removed(self):
        result = self.validate([
            upcoming_event("2026-11-16"),
            upcoming_event("2026-11-17"),
            upcoming_event("2026-11-18"),
        ])
        self.assertEqual(
            [item["date"] for item in result["upcoming_high_impact_events"]],
            ["2026-11-17", "2026-11-18"],
        )

    def test_unqualified_source_rejected(self):
        other = {"title": "Unqualified article", "url": "https://invalid.example/article"}
        with self.assertRaisesRegex(ValueError, "unqualified source"):
            self.validate([upcoming_event("2026-11-18", other)])

    def test_missing_date_rejected(self):
        item = upcoming_event("2026-11-18")
        del item["date"]
        with self.assertRaisesRegex(ValueError, "invalid date"):
            self.validate([item])

    def test_invalid_date_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid date"):
            self.validate([upcoming_event("2026-11-31")])

    def test_empty_watchlist_accepted(self):
        self.assertEqual(self.validate([])["upcoming_high_impact_events"], [])

    def test_probabilities_unchanged_by_watchlist(self):
        baseline = self.validate([])
        populated = self.validate([upcoming_event("2026-11-18")])
        self.assertEqual(baseline["next_day"], populated["next_day"])
        self.assertEqual(baseline["next_2_6_weeks"], populated["next_2_6_weeks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
