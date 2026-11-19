import unittest
import sys
import types
sys.modules.setdefault("feedparser", types.ModuleType("feedparser"))
from unittest.mock import patch
import news

class NewsFallbackTests(unittest.TestCase):
    def test_primary_success(self):
        with patch.object(news, 'parse_rss_with_retry', return_value=types.SimpleNamespace(entries=[])) as parse:
            result = news.fetch_news()
        self.assertEqual(result['coverage_status'], 'COMPLETE')
        self.assertEqual(result['errors'], {})
        self.assertTrue(all(v == 'primary' for v in result['category_coverage'].values()))
        self.assertEqual(parse.call_count, len(news.NEWS_QUERIES))

    def test_primary_failure_fallback_success(self):
        def fake_parse(query):
            if query == news.NEWS_QUERIES['global_macro']:
                raise ValueError('malformed primary')
            return types.SimpleNamespace(entries=[])
        with patch.object(news, 'parse_rss_with_retry', side_effect=fake_parse):
            result = news.fetch_news()
        self.assertEqual(result['category_coverage']['global_macro'], 'fallback')
        self.assertEqual(result['coverage_status'], 'PARTIAL')
        self.assertEqual(result['errors'], {})

    def test_both_fail(self):
        with patch.object(news, 'parse_rss_with_retry', side_effect=ValueError('bad XML')):
            result = news.fetch_news()
        self.assertEqual(result['coverage_status'], 'PARTIAL')
        self.assertEqual(len(result['errors']), len(news.NEWS_QUERIES))
        self.assertTrue(all(v == 'failed' for v in result['category_coverage'].values()))


if __name__ == '__main__':
    unittest.main()
