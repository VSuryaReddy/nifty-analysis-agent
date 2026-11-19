import json
import socket
import ssl
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

import news
import news_freshness as freshness


NOW = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
GOOGLE = 'https://news.google.com/rss/articles/article-token'
PUBLISHER = 'https://publisher.example/news/policy'
RECENT = '2026-11-16T05:00:00Z'


def publisher_page(date=RECENT):
    return f'<meta property="article:published_time" content="{date}">'


def google_parameters(article_id='article-token'):
    return (f'<div data-n-a-id="{article_id}" data-n-a-ts="1794940188" '
            'data-n-a-sg="signed-article-token"></div>')


def rpc_response(url=PUBLISHER):
    payload = json.dumps(['garturlres', url])
    return ")]}'\n\n123\n" + json.dumps([
        ['wrb.fr', 'Fbv4je', payload], ['di', 1], ['af.httprm', 0]])


class PublisherResolutionTests(unittest.TestCase):
    def test_direct_http_redirect_to_publisher(self):
        redirect = MagicMock(status=302)
        redirect.getheader.return_value = PUBLISHER.replace('https:', 'http:')
        publisher = MagicMock(status=200)
        publisher.read1.side_effect = [publisher_page().encode(), b'']
        public_address = [(0, 0, 0, '', ('8.8.8.8', 80))]
        with patch.object(socket, 'getaddrinfo', return_value=public_address), \
                patch.object(socket, 'create_connection'), \
                patch.object(freshness.http.client, 'HTTPConnection') as connection:
            connection.return_value.getresponse.side_effect = [redirect, publisher]
            result = freshness.verify_original_publication(GOOGLE.replace('https:', 'http:'), NOW)
        self.assertEqual(result['freshness_status'], 'VERIFIED_FRESH')
        self.assertEqual(result['original_article_url'], PUBLISHER.replace('https:', 'http:'))

    def test_unambiguous_html_publisher_identity(self):
        for html in (
            f'<link rel="canonical" href="{PUBLISHER}">',
            f'<meta property="og:url" content="{PUBLISHER}">',
            f'<a data-n-a-id="article-token" href="{PUBLISHER}">Read article</a>',
        ):
            with self.subTest(html=html), patch.object(
                    freshness, 'fetch_public_page', side_effect=[
                        (GOOGLE, html), (PUBLISHER, publisher_page())]) as fetch:
                result = freshness.verify_original_publication(GOOGLE, NOW)
            self.assertEqual(result['freshness_status'], 'VERIFIED_FRESH')
            self.assertEqual(fetch.call_args_list[1].args[0], PUBLISHER)
            self.assertEqual(fetch.call_args_list[0].args[1], fetch.call_args_list[1].args[1])

    def test_navigation_links_are_not_article_identity(self):
        html = f'<a href="{PUBLISHER}">Some link</a><a href="https://other.example/">Home</a>'
        with patch.object(freshness, 'fetch_public_page', return_value=(GOOGLE, html)) as fetch:
            result = freshness.verify_original_publication(GOOGLE, NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertEqual(fetch.call_count, 1)

    def test_ambiguous_html_does_not_try_rpc_or_publisher(self):
        html = (f'<link rel="canonical" href="{PUBLISHER}">'
                '<meta property="og:url" content="https://other.example/story">'
                + google_parameters())
        with patch.object(freshness, 'fetch_public_page', return_value=(GOOGLE, html)) as fetch:
            result = freshness.verify_original_publication(GOOGLE, NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertIn('Ambiguous', result['freshness_reason'])
        self.assertEqual(fetch.call_count, 1)

    def test_signed_rpc_resolution_and_request_identity(self):
        with patch.object(freshness, 'fetch_public_page', side_effect=[
                (GOOGLE, google_parameters()), (GOOGLE, rpc_response()),
                (PUBLISHER, publisher_page())]) as fetch:
            result = freshness.verify_original_publication(GOOGLE, NOW)
        self.assertEqual(result['freshness_status'], 'VERIFIED_FRESH')
        call = fetch.call_args_list[1]
        form = parse_qs(call.kwargs['body'].decode())
        frame = json.loads(form['f.req'][0])[0][0]
        request = json.loads(frame[1])
        self.assertEqual(frame[0], 'Fbv4je')
        self.assertEqual(request[2:], ['article-token', 1794940188, 'signed-article-token'])
        self.assertEqual(len({c.args[1] for c in fetch.call_args_list}), 1)

    def test_rpc_requires_parameters_for_same_article(self):
        with patch.object(freshness, 'fetch_public_page', return_value=(GOOGLE, google_parameters('different-id'))) as fetch:
            result = freshness.verify_original_publication(GOOGLE, NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertEqual(fetch.call_count, 1)

    def test_rpc_null_malformed_and_conflicting_responses(self):
        responses = [
            json.dumps([['wrb.fr', 'Fbv4je', None]]),
            'not json',
            json.dumps([['wrb.fr', 'wrong-rpc', json.dumps(['garturlres', PUBLISHER])]]),
            json.dumps([['wrb.fr', 'Fbv4je', json.dumps(['garturlres', url])]
                        for url in (PUBLISHER, 'https://other.example/story')]),
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                freshness.decode_google_resolution_response(response)

    def test_unsafe_discovered_publisher_blocked_by_transport(self):
        google_page = f'<link rel="canonical" href="http://127.0.0.1/secret">'
        google_response = MagicMock(status=200)
        google_response.read1.side_effect = [google_page.encode(), b'']
        addresses = [[(0, 0, 0, '', ('8.8.8.8', 80))],
                     [(0, 0, 0, '', ('127.0.0.1', 80))]]
        with patch.object(socket, 'getaddrinfo', side_effect=addresses), \
                patch.object(socket, 'create_connection') as connect, \
                patch.object(freshness.http.client, 'HTTPConnection') as connection:
            connection.return_value.getresponse.return_value = google_response
            result = freshness.verify_original_publication(GOOGLE.replace('https:', 'http:'), NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertIn('Non-public', result['freshness_reason'])
        self.assertEqual(connect.call_count, 1)

    def test_rpc_post_redirect_does_not_forward_body(self):
        response = MagicMock(status=307)
        with patch.object(socket, 'getaddrinfo', return_value=[(0, 0, 0, '', ('8.8.8.8', 80))]), \
                patch.object(socket, 'create_connection') as connect, \
                patch.object(freshness.http.client, 'HTTPConnection') as connection:
            connection.return_value.getresponse.return_value = response
            with self.assertRaisesRegex(ValueError, 'POST redirected'):
                freshness.fetch_public_page('http://news.google.com/rpc', freshness.time.monotonic() + 8, body=b'payload')
        self.assertEqual(connect.call_count, 1)

    def test_publisher_timeout_after_google_discovery(self):
        html = f'<link rel="canonical" href="{PUBLISHER}">'
        for error in (TimeoutError('publisher request timed out'),
                      ssl.SSLCertVerificationError('publisher certificate invalid')):
            with self.subTest(error=error), patch.object(
                    freshness, 'fetch_public_page', side_effect=[(GOOGLE, html), error]):
                result = freshness.verify_original_publication(GOOGLE, NOW)
            self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
            self.assertIn(str(error), result['freshness_reason'])

    def test_google_resolution_keeps_april_original_stale(self):
        entry = dict(title='India RBI policy', link=GOOGLE, published_parsed=NOW.utctimetuple())
        with patch.object(news, 'parse_rss_with_retry') as feed, \
                patch.object(freshness, 'fetch_public_page', side_effect=[
                    (GOOGLE, f'<meta property="og:url" content="{PUBLISHER}">'),
                    (PUBLISHER, publisher_page('2026-04-15T08:00:00Z'))]):
            feed.return_value.entries = [entry]
            result = news.fetch_news(reference_time=NOW)
        self.assertEqual(result['article_count'], 0)
        self.assertEqual(result['diagnostic_articles'][0]['freshness_status'], 'STALE')

    def test_unrelated_malformed_json_ld_does_not_override_valid_meta(self):
        html = publisher_page() + '<script type="application/ld+json">{bad json}</script>'
        with patch.object(freshness, 'fetch_public_page', return_value=(PUBLISHER, html)):
            result = freshness.verify_original_publication(PUBLISHER, NOW)
        self.assertEqual(result['freshness_status'], 'VERIFIED_FRESH')

    def test_conflicting_publication_still_rejected_with_broken_unrelated_block(self):
        html = (publisher_page() + '<script type="application/ld+json">bad</script>'
                '<script type="application/ld+json">'
                + json.dumps({'@type': 'NewsArticle', 'datePublished': '2026-04-15T08:00:00Z'})
                + '</script>')
        with patch.object(freshness, 'fetch_public_page', return_value=(PUBLISHER, html)):
            result = freshness.verify_original_publication(PUBLISHER, NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')
        self.assertIn('Conflicting', result['freshness_reason'])

    def test_malformed_article_date_still_rejected_with_valid_meta(self):
        html = (publisher_page() + '<script type="application/ld+json">'
                + json.dumps({'@type': 'NewsArticle', 'datePublished': 'invalid date'})
                + '</script>')
        with patch.object(freshness, 'fetch_public_page', return_value=(PUBLISHER, html)):
            result = freshness.verify_original_publication(PUBLISHER, NOW)
        self.assertEqual(result['freshness_status'], 'UNVERIFIED_DATE')


if __name__ == '__main__':
    unittest.main()
