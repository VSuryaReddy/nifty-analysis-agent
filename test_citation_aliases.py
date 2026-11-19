import copy
import unittest
from datetime import date

from analyst import prepare_qualified_news, validate_analysis
from test_upcoming_events_validated import base_analysis, fact_result


RSS_URL = 'https://news.google.com/rss/articles/qualified-article'
PUBLISHER_URL = 'https://publisher.example/news/policy-meeting'
EXCLUDED_URL = 'https://publisher.example/news/excluded-article'
REPORT_DATE = date(2026, 11, 17)
MARKET_DATA = {'markets': {}, 'collected_at_utc': '2026-11-17T12:00:00Z'}


class CitationAliasTests(unittest.TestCase):
    def setUp(self):
        self.facts = fact_result()
        source = self.facts['assessed_events'][0]['sources'][0]
        source.update(url=RSS_URL, original_article_url=PUBLISHER_URL)
        self.title = source['title']

    def qualified_news(self):
        return prepare_qualified_news(self.facts, MARKET_DATA)

    def analysis(self, url, kind, title=None, event_date='2026-11-18'):
        citation = {'title': self.title if title is None else title, 'url': url}
        result = base_analysis([])
        if kind == 'important':
            result['important_events'] = [{
                'event': 'Policy decision', 'impact': 'Mixed',
                'reason': 'Qualified claim', 'sources': [citation],
            }]
        else:
            result['upcoming_high_impact_events'] = [{
                'event': 'Policy decision', 'date': event_date,
                'why_watch': 'Policy uncertainty', 'sources': [citation],
            }]
        return result

    def validate(self, analysis):
        return validate_analysis(analysis, self.qualified_news(), report_date=REPORT_DATE)

    def test_both_aliases_pass_for_both_event_types(self):
        for kind in ('important', 'upcoming'):
            for url in (RSS_URL, PUBLISHER_URL):
                with self.subTest(kind=kind, url=url):
                    self.validate(self.analysis(url, kind))

    def test_unrelated_urls_fail_for_both_event_types(self):
        for kind in ('important', 'upcoming'):
            for url in ('https://unrelated.example/article', PUBLISHER_URL + '?invented=1'):
                with self.subTest(kind=kind, url=url), self.assertRaises(ValueError):
                    self.validate(self.analysis(url, kind))

    def test_exact_title_required_for_both_aliases(self):
        for kind in ('important', 'upcoming'):
            for url in (RSS_URL, PUBLISHER_URL):
                with self.subTest(kind=kind, url=url), self.assertRaises(ValueError):
                    self.validate(self.analysis(url, kind, title=self.title + ' '))

    def test_publisher_alias_from_excluded_article_fails(self):
        event = self.facts['assessed_events'][0]
        excluded = copy.deepcopy(event['sources'][0])
        # This fresh article remains in event metadata but supports no retained claim.
        excluded.update(id=99, original_article_url=EXCLUDED_URL)
        event['sources'].append(excluded)
        for kind in ('important', 'upcoming'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(self.analysis(EXCLUDED_URL, kind))

    def test_source_only_on_unqualified_claim_fails(self):
        event = self.facts['assessed_events'][0]
        excluded = copy.deepcopy(event['sources'][0])
        excluded.update(id=99, url=RSS_URL + '-excluded', original_article_url=EXCLUDED_URL)
        event['sources'].append(excluded)
        claim = copy.deepcopy(event['fact_assessment']['claims'][0])
        claim['supporting_article_ids'] = [99]
        claim['evidence_assessment']['evidence_status'] = 'INSUFFICIENT_EVIDENCE'
        event['fact_assessment']['claims'].append(claim)
        for kind in ('important', 'upcoming'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(self.analysis(EXCLUDED_URL, kind))

    def test_mismatched_claim_date_fails_for_both_aliases(self):
        for url in (RSS_URL, PUBLISHER_URL):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, 'scheduled date not supported'):
                self.validate(self.analysis(url, 'upcoming', event_date='2026-11-20'))

    def test_relative_claim_date_fails_for_both_aliases(self):
        self.facts['assessed_events'][0]['fact_assessment']['claims'][0]['claim'] = 'The meeting is next week.'
        for url in (RSS_URL, PUBLISHER_URL):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, 'scheduled date not supported'):
                self.validate(self.analysis(url, 'upcoming'))

    def test_date_on_different_cited_source_cannot_authorize_either_alias(self):
        event = self.facts['assessed_events'][0]
        other = copy.deepcopy(event['sources'][0])
        other.update(id=99, title='Other article', url=RSS_URL + '-other', original_article_url=EXCLUDED_URL)
        event['sources'].append(other)
        claim = copy.deepcopy(event['fact_assessment']['claims'][0])
        claim.update(claim='The meeting is on 20 November 2026.', supporting_article_ids=[99])
        event['fact_assessment']['claims'].append(claim)
        for url in (RSS_URL, PUBLISHER_URL):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, 'scheduled date not supported'):
                self.validate(self.analysis(url, 'upcoming', event_date='2026-11-20'))

    def test_missing_publisher_alias_preserves_rss_citations(self):
        self.facts['assessed_events'][0]['sources'][0].pop('original_article_url')
        self.validate(self.analysis(RSS_URL, 'important'))
        with self.assertRaises(ValueError):
            self.validate(self.analysis(PUBLISHER_URL, 'important'))

    def test_invalid_publisher_aliases_are_not_allowlisted(self):
        for url in ('javascript:alert(1)', '/relative', 'https://publisher.example:bad/story',
                    'https://user:password@publisher.example/story', 'https://publisher.example/a b'):
            self.facts['assessed_events'][0]['sources'][0]['original_article_url'] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.validate(self.analysis(url, 'important'))


if __name__ == '__main__':
    unittest.main()
