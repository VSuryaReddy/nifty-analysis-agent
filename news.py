
import json
import calendar
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser


NEWS_QUERIES = {
    "indian_economy": (
        "India RBI inflation GDP economy"
    ),
    "indian_markets": (
        "Nifty FII DII banking Indian stock market"
    ),
    "global_economy": (
        "Federal Reserve US inflation treasury yields stocks"
    ),
    "global_risks": (
        "Brent crude oil OPEC geopolitical trade tariffs"
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


def fetch_news():
    now = datetime.now(timezone.utc)

    result = {
        "collected_at_utc": now.isoformat(),
        "articles": [],
        "errors": {},
    }

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
                if count >= MAX_ARTICLES_PER_CATEGORY:
                    break

                if not entry.get("published_parsed"):
                    continue

                published = datetime.fromtimestamp(
                    calendar.timegm(
                        entry.published_parsed
                    ),
                    tz=timezone.utc,
                )

                age_days = (
                    now - published
                ).total_seconds() / 86400

                if not 0 <= age_days <= MAX_AGE_DAYS:
                    continue

                title = entry.get("title", "").strip()
                link = entry.get("link", "")

                if not title or not link:
                    continue

                key = title.casefold()

                if key in seen:
                    continue

                seen.add(key)

                result["articles"].append({
                    "category": category,
                    "title": title,
                    "published_at_utc": (
                        published.isoformat()
                    ),
                    "source": entry.get(
                        "source", {}
                    ).get("title", "Unknown"),
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
