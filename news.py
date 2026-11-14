
import json
import calendar
from datetime import datetime, timezone
from urllib.parse import quote
from datetime import datetime, timedelta, timezone

import feedparser


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

MAX_ARTICLES_PER_CATEGORY = 8
MAX_AGE_DAYS = 7


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


def fetch_news():
    now = datetime.now(timezone.utc)

    result = {
        "collected_at_utc": now.isoformat(),
        "articles": [],
        "errors": {},
    }

    # Track duplicate article titles across categories
    seen = set()

    for category, query in NEWS_QUERIES.items():
        try:
            feed = feedparser.parse(
                build_rss_url(query),
                agent="NiftyResearchPrototype/0.1",
            )

            if feed.bozo and not feed.entries:
                raise ValueError(
                    str(feed.bozo_exception)
                )

            count = 0

            for entry in feed.entries:

                # Limit articles per category
                if count >= MAX_ARTICLES_PER_CATEGORY:
                    break

                # Skip articles without publication dates
                if not entry.get("published_parsed"):
                    continue

                # Convert RSS publication date to UTC
                published = datetime.fromtimestamp(
                    calendar.timegm(
                        entry.published_parsed
                    ),
                    tz=timezone.utc,
                )

                # Step 1: Freshness validation
                if not is_recent_article(
                    published,
                    max_age_days=MAX_AGE_DAYS
                ):
                    continue

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

                # Step 4: Store accepted article
                result["articles"].append({
                    "category": category,
                    "title": title,
                    "published_at_utc": (
                        published.isoformat()
                    ),
                    "source": entry.get(
                        "source", {}
                    ).get(
                        "title", "Unknown"
                    ),
                    "url": link,
                })

                count += 1

        except Exception as exc:
            result["errors"][category] = str(exc)

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
