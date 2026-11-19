
import json
import calendar
import time
from urllib.parse import quote
from datetime import datetime, timedelta, timezone

import feedparser

from news_freshness import MAX_AGE_DAYS, verify_original_publication, is_current_article

NEWS_QUERIES = {
    "india_macro": (
        "India RBI inflation GDP interest rates "
        "monetary policy"
    ),
    "india_policy": (
        "India government economic policy "
        "budget tariffs regulation"
    ),
    "india_markets": (
        "Nifty Bank Nifty FII DII "
        "Indian stock market"
    ),
    "global_macro": (
        "Federal Reserve US inflation "
        "Treasury yields interest rates"
    ),
    "global_markets": (
        "S&P 500 Nasdaq global stocks "
        "market selloff"
    ),
    "global_risks": (
        "Brent crude OPEC geopolitical "
        "oil supply trade tariffs"
    ),
}

# Alternative search terms used only if the primary feed cannot be parsed.
FALLBACK_NEWS_QUERIES = {
    "india_macro": "India economy RBI inflation GDP",
    "india_policy": "India economic policy government RBI SEBI",
    "india_markets": "Nifty Indian stocks foreign investors FII",
    "global_macro": "US economy Fed policy inflation Treasury",
    "global_markets": "US stocks Wall Street Nasdaq S&P 500",
    "global_risks": "oil prices OPEC geopolitics trade tensions",
}

MAX_ARTICLES_PER_CATEGORY = 8
RSS_MAX_ATTEMPTS = 3
RSS_RETRY_DELAY_SECONDS = 1


def build_rss_url(query):
    encoded_query = quote(
        f"{query} when:7d"
    )

    return (
        "https://news.google.com/rss/search"
        f"?q={encoded_query}"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )


def is_relevant_article(category, title):
    title_lower = title.lower()

    if category == "india_policy":
        india_terms = (
            "india", "indian", "rbi", "nifty",
            "sebi", "union budget", "new delhi",
        )
        return any(
            term in title_lower for term in india_terms
        )

    return True



def is_recent_article(published_at_utc, max_age_days=7):
    if published_at_utc is None:
        return False

    try:
        if isinstance(published_at_utc, datetime):
            published = published_at_utc

        elif isinstance(published_at_utc, str):
            published = datetime.fromisoformat(
                published_at_utc.replace("Z", "+00:00")
            )

        else:
            return False

        if published.tzinfo is None:
            return False

        now = datetime.now(timezone.utc)
        age = now - published

        return (
            timedelta(0)
            <= age
            <= timedelta(days=max_age_days)
        )

    except (ValueError, TypeError):
        return False


def parse_rss_with_retry(query):
    """Retry unusable RSS responses; never silently accept malformed empty feeds."""
    last_error = None
    for attempt in range(RSS_MAX_ATTEMPTS):
        try:
            feed = feedparser.parse(
                build_rss_url(query),
                agent="NiftyResearchPrototype/0.1",
            )
            if feed.bozo and not feed.entries:
                raise ValueError(str(feed.bozo_exception))
            return feed
        except Exception as exc:
            last_error = exc
            if attempt < RSS_MAX_ATTEMPTS - 1:
                time.sleep(RSS_RETRY_DELAY_SECONDS)
    raise ValueError(
        f"RSS failed after {RSS_MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def fetch_news(reference_time=None):
    now = reference_time or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Collection time requires a timezone")
    now = now.astimezone(timezone.utc)

    result = {
        "collected_at_utc": now.isoformat(),
        "articles": [],
        "errors": {},
        "diagnostic_articles": [],
        "collection_statistics": {
            "total_rss_candidates": 0,
            "verified_fresh_articles": 0,
            "stale_articles_rejected": 0,
            "unverified_date_articles": 0,
            "original_publication_verification_failures": 0,
        },
        # Each category is primary, fallback, or failed.
        "category_coverage": {},
    }

    # Track duplicate article titles across categories
    seen = set()

    for category, query in NEWS_QUERIES.items():
        try:
            try:
                feed = parse_rss_with_retry(query)
                result["category_coverage"][category] = "primary"
            except Exception as primary_error:
                try:
                    feed = parse_rss_with_retry(FALLBACK_NEWS_QUERIES[category])
                    result["category_coverage"][category] = "fallback"
                except Exception as fallback_error:
                    raise ValueError(
                        f"Primary failed: {primary_error}; "
                        f"fallback failed: {fallback_error}"
                    ) from fallback_error

            result["collection_statistics"]["total_rss_candidates"] += len(feed.entries)
            count = 0

            for entry in feed.entries:

                # Limit articles per category
                if count >= MAX_ARTICLES_PER_CATEGORY:
                    break

                published = None
                try:
                    if entry.get("published_parsed"):
                        published = datetime.fromtimestamp(
                            calendar.timegm(entry.get("published_parsed")), tz=timezone.utc)
                except (ValueError, TypeError, OverflowError):
                    pass
                # Extract article details
                title = entry.get(
                    "title", ""
                ).strip()

                link = entry.get(
                    "link", ""
                )

                if not title or not link:
                    continue

                # Step 2: Basic relevance validation
                if not is_relevant_article(
                    category,
                    title
                ):
                    continue

                # Step 3: Exact-title duplicate detection
                key = title.casefold()

                if key in seen:
                    continue

                seen.add(key)

                verification = verify_original_publication(link, now)
                article = {
                    "category": category,
                    "title": title,
                    "published_at_utc": verification["original_published_at_utc"],
                    "rss_published_at_utc": published.isoformat() if published else None,
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "url": link,
                    **verification,
                }
                statistics = result["collection_statistics"]
                if verification["publication_verification_failed"]:
                    statistics["original_publication_verification_failures"] += 1
                if verification["freshness_status"] == "STALE":
                    statistics["stale_articles_rejected"] += 1
                elif verification["freshness_status"] == "UNVERIFIED_DATE":
                    statistics["unverified_date_articles"] += 1
                else:
                    statistics["verified_fresh_articles"] += 1
                if is_current_article(article, now):
                    result["articles"].append(article)
                    count += 1
                else:
                    result["diagnostic_articles"].append(article)

        except Exception as exc:
            result["category_coverage"][category] = "failed"
            result["errors"][category] = str(exc)

    result["coverage_status"] = (
        "COMPLETE" if all(
            status == "primary" for status in result["category_coverage"].values()
        ) else "PARTIAL"
    )
    result["article_count"] = len(
        result["articles"]
    )

    return result


if __name__ == "__main__":
    news = fetch_news()
    print(json.dumps(
        news,
        indent=2,
        ensure_ascii=False,
    ))
