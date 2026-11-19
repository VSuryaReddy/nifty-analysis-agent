import json
import socket
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import news
import news_freshness as freshness
from analyst import prepare_qualified_news
from news_intelligence import group_news_into_events
from fact_verification import verify_event_facts

NOW = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
RECENT = '2026-11-16T05:00:00+00:00'
URL = 'https://publisher.example/article'


def page(date):
    return f'<meta property="article:published_time" content="{date}">'


def source():
    return dict(id=0, category='india_macro', title='India RBI policy', source='Reuters',
                url=URL, published_at_utc=RECENT, original_published_at_utc=RECENT,
                freshness_status='VERIFIED_FRESH', freshness_checked_at_utc=NOW.isoformat())


def facts(article):
    return {'assessed_events': [{'sources': [article], 'fact_assessment': {'claims': [{
        'claim': 'RBI policy', 'claim_type': 'reported',
        'headline_grounding': 'SUPPORTED_BY_HEADLINE', 'supporting_article_ids': [0],
        'evidence_assessment': {'evidence_status': 'CREDIBLE_SINGLE_SOURCE'}}]}}]}


class FreshnessTests(unittest.TestCase):
    def verify(self, html):
        with patch.object(freshness, 'fetch_public_page', return_value=(URL, html)):
            return freshness.verify_original_publication(URL, NOW)

    def collect(self, html):
        entry = dict(title='India RBI policy', link=URL, published_parsed=NOW.utctimetuple())
        with patch.object(news, 'parse_rss_with_retry', return_value=types.SimpleNamespace(entries=[entry])), patch.object(freshness, 'fetch_public_page', return_value=(URL, html)):
            return news.fetch_news(reference_time=NOW)

    def test_april_original_november_rss_rejected(self):
        result = self.collect(page('2026-04-15T08:00:00Z'))
        self.assertEqual(result['article_count'], 0)
        rejected = result['diagnostic_articles'][0]
        self.assertEqual(rejected['freshness_status'], 'STALE')
        self.assertTrue(rejected['rss_published_at_utc'].startswith('2026-11-17'))
        self.assertTrue(rejected['original_published_at_utc'].startswith('2026-04-15'))
        self.assertEqual(result['collection_statistics']['stale_articles_rejected'], 1)
        self.assertEqual(result['coverage_status'], 'COMPLETE')

    def test_recent_original_accepted(self):
        result = self.collect(page(RECENT))
        self.assertEqual(result['article_count'], 1)
        self.assertEqual(result['articles'][0]['original_article_url'], URL)
        self.assertEqual(result['collection_statistics']['verified_fresh_articles'], 1)

    def test_missing_date_unverified(self):
        result = self.collect('<html></html>')
        self.assertEqual(result['article_count'], 0)
        self.assertEqual(result['diagnostic_articles'][0]['freshness_status'], 'UNVERIFIED_DATE')
        self.assertEqual(result['collection_statistics']['original_publication_verification_failures'], 1)

    def test_future_date_rejected(self):
        self.assertEqual(self.verify(page('2026-11-18T01:00:00Z'))['freshness_status'], 'UNVERIFIED_DATE')

    def test_conflicting_dates_unverified(self):
        html = page(RECENT) + '<script type="application/ld+json">' + json.dumps({'@type': 'NewsArticle', 'datePublished': '2026-04-15T08:00:00Z'}) + '</script>'
        self.assertEqual(self.verify(html)['freshness_status'], 'UNVERIFIED_DATE')

    def test_modification_only_unverified(self):
        html = '<meta property="article:modified_time" content="' + RECENT + '"><script type="application/ld+json">' + json.dumps({'@type': 'NewsArticle', 'dateModified': RECENT}) + '</script>'
        self.assertEqual(self.verify(html)['freshness_status'], 'UNVERIFIED_DATE')

    def test_ambiguous_and_malformed_dates(self):
        for date in ('bad date', '2026-11-16', '2026-11-16T05:00:00'):
            with self.subTest(date=date):
                self.assertEqual(self.verify(page(date))['freshness_status'], 'UNVERIFIED_DATE')

    def test_timezone_normalized(self):
        self.assertEqual(self.verify(page('2026-11-16T10:30:00+05:30'))['original_published_at_utc'], RECENT)

    def test_structured_graph(self):
        html = '<script type="application/ld+json">' + json.dumps({'@graph': [{'@type': 'NewsArticle', 'datePublished': RECENT}]}) + '</script>'
        self.assertEqual(self.verify(html)['freshness_status'], 'VERIFIED_FRESH')

    def test_timeout_and_resolution_failure(self):
        for error in (TimeoutError('timeout'), socket.gaierror('resolution failed')):
            with patch.object(freshness, 'fetch_public_page', side_effect=error):
                result = freshness.verify_original_publication(URL, NOW)
                self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
                self.assertTrue(result['publication_verification_failed'])

    def test_unresolved_google_url(self):
        with patch.object(freshness, 'fetch_public_page', return_value=('https://news.google.com/rss/articles/opaque', page(RECENT))):
            result = freshness.verify_original_publication(URL, NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertIsNone(result['original_article_url'])

    def test_dns_deadline_is_bounded(self):
        with patch.object(freshness.threading, 'Thread'), patch.object(freshness.time, 'monotonic', return_value=100):
            with self.assertRaises(TimeoutError):
                freshness.resolve_public_addresses('publisher.example', 443, 99)

    def test_missing_rss_date_can_use_verified_publisher_date(self):
        entry = dict(title='India RBI policy', link=URL)
        with patch.object(news, 'parse_rss_with_retry', return_value=types.SimpleNamespace(entries=[entry])), patch.object(freshness, 'fetch_public_page', return_value=(URL, page(RECENT))):
            result = news.fetch_news(reference_time=NOW)
        self.assertEqual(result['article_count'], 1)
        self.assertIsNone(result['articles'][0]['rss_published_at_utc'])

    def test_boundary_seven_days(self):
        self.assertEqual(self.verify(page('2026-11-10T12:00:00Z'))['freshness_status'], 'VERIFIED_FRESH')
        self.assertEqual(self.verify(page('2026-11-10T11:59:59Z'))['freshness_status'], 'STALE')

    def test_grouping_excludes_unverified_and_legacy(self):
        for status in ('STALE', 'UNVERIFIED_DATE', None):
            article = source()
            article['freshness_status'] = status
            with patch('news_intelligence.OpenAI') as client:
                result = group_news_into_events({'collected_at_utc': NOW.isoformat(), 'articles': [article]})
            self.assertEqual(result['events'], [])
            client.assert_not_called()

    def test_grouping_preserves_provenance(self):
        with patch('news_intelligence.OpenAI') as client:
            client.return_value.responses.create.return_value.output_text = json.dumps({'events': [{'article_ids': [0]}], 'ignored_articles': []})
            result = group_news_into_events({'collected_at_utc': NOW.isoformat(), 'articles': [source()]})
        self.assertEqual(result['events'][0]['sources'][0]['original_published_at_utc'], RECENT)

    def test_final_evidence_rejects_stale_unverified_legacy_and_forged_fresh(self):
        for status, date in [('STALE', RECENT), ('UNVERIFIED_DATE', RECENT), (None, RECENT), ('VERIFIED_FRESH', '2026-04-15T08:00:00Z'), ('VERIFIED_FRESH', '2026-11-18T00:00:00Z')]:
            article = source()
            article.update(freshness_status=status, original_published_at_utc=date)
            result = prepare_qualified_news(facts(article), {'collected_at_utc': NOW.isoformat(), 'markets': {}})
            self.assertEqual(result['events'], [])

    def test_final_evidence_accepts_verified_fresh(self):
        result = prepare_qualified_news(facts(source()), {'collected_at_utc': NOW.isoformat(), 'markets': {}})
        self.assertEqual(len(result['events']), 1)

    def test_fact_boundary_rejects_ineligible_sources(self):
        article = source()
        article['freshness_status'] = 'STALE'
        with patch('fact_verification.OpenAI') as client, self.assertRaises(ValueError):
            verify_event_facts({'accepted_events': [{'sources': [article]}]})
        client.assert_not_called()

    def test_retry_preserved(self):
        feed = types.SimpleNamespace(bozo=False, entries=[])
        with patch.object(news.feedparser, 'parse', create=True, side_effect=[OSError('temporary'), feed]) as parse, patch.object(news.time, 'sleep') as sleep:
            self.assertIs(news.parse_rss_with_retry('test'), feed)
        self.assertEqual(parse.call_count, 2)
        sleep.assert_called_once()

    def test_article_limit_and_duplicates_preserved(self):
        entries = [dict(title=f'India RBI policy {i}', link=f'{URL}/{i}')
                   for i in range(12)]
        entries.insert(1, entries[0])
        with patch.object(news, 'parse_rss_with_retry', return_value=types.SimpleNamespace(entries=entries)), patch.object(freshness, 'fetch_public_page', return_value=(URL, page(RECENT))):
            result = news.fetch_news(reference_time=NOW)
        self.assertEqual(len([a for a in result['articles'] if a['category'] == 'india_macro']), 8)
        self.assertEqual(len({a['title'] for a in result['articles']}), result['article_count'])

    def test_http_page_read_and_redirect_limit(self):
        response = MagicMock(status=200)
        response.read1.side_effect = [page(RECENT).encode(), b'']
        with patch.object(socket, 'getaddrinfo', return_value=[(0, 0, 0, '', ('8.8.8.8', 80))]), patch.object(socket, 'create_connection'), patch.object(freshness.http.client, 'HTTPConnection') as conn:
            conn.return_value.getresponse.return_value = response
            result = freshness.verify_original_publication('http://publisher.example/article', NOW)
        self.assertEqual(result['freshness_status'], 'VERIFIED_FRESH')
        response.status = 302
        response.getheader.return_value = '/loop'
        with patch.object(socket, 'getaddrinfo', return_value=[(0, 0, 0, '', ('8.8.8.8', 80))]), patch.object(socket, 'create_connection'), patch.object(freshness.http.client, 'HTTPConnection') as conn:
            conn.return_value.getresponse.return_value = response
            result = freshness.verify_original_publication('http://publisher.example/article', NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertIn('redirects', result['freshness_reason'])

    def test_collection_reference_used_in_final_evidence(self):
        data = facts(source())
        data['news_collected_at_utc'] = NOW.isoformat()
        result = prepare_qualified_news(data, {'collected_at_utc': '2026-11-16T00:00:00Z', 'markets': {}})
        self.assertEqual(len(result['events']), 1)

    def test_private_destinations_rejected_before_connection(self):
        for address in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', '::ffff:127.0.0.1'):
            with patch.object(socket, 'getaddrinfo', return_value=[(0, 0, 0, '', (address, 443))]), patch.object(socket, 'create_connection') as connect:
                result = freshness.verify_original_publication(URL, NOW)
            self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
            connect.assert_not_called()

    def test_redirect_to_private_address_rejected(self):
        response = MagicMock(status=302)
        response.getheader.return_value = 'http://127.0.0.1/secret'
        addresses = [[(0, 0, 0, '', ('8.8.8.8', 80))], [(0, 0, 0, '', ('127.0.0.1', 80))]]
        with patch.object(socket, 'getaddrinfo', side_effect=addresses), patch.object(socket, 'create_connection') as connect, patch.object(freshness.http.client, 'HTTPConnection') as conn:
            conn.return_value.getresponse.return_value = response
            result = freshness.verify_original_publication('http://publisher.example/article', NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertEqual(connect.call_count, 1)


if __name__ == '__main__':
    unittest.main()
