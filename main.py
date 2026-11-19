
import json
from datetime import datetime, timezone
from pathlib import Path

from market_data import fetch_market_snapshot
from news import fetch_news
from news_intelligence import group_news_into_events
from event_quality import assess_event_quality
from fact_verification import (
    verify_event_facts,
    assess_headline_agreement,
    apply_evidence_assessment,
)
from analyst import analyze_market


def main():
    print("Starting Nifty AI Analysis...\n")

    print("1. Fetching market data...")
    market_data = fetch_market_snapshot()

    if "nifty_50" not in market_data["markets"]:
        raise RuntimeError(
            "Nifty data unavailable. Analysis stopped."
        )

    print(
        f"   Market inputs: "
        f"{len(market_data['markets'])}"
    )

    print("\n2. Fetching financial news...")
    news_data = fetch_news()

    print(
        f"   Articles collected: "
        f"{news_data['article_count']}"
    )

    print("   Freshness statistics:", json.dumps(news_data.get("collection_statistics", {})))

    if news_data["article_count"] == 0:
        print(json.dumps(news_data, indent=2, ensure_ascii=False))
        raise RuntimeError(
            "No verified fresh news collected. Analysis stopped."
        )

    print("\n3. Grouping news into events...")
    grouped_news = group_news_into_events(news_data)

    print("\n4. Filtering market-relevant events...")
    quality_result = assess_event_quality(grouped_news)

    print("\n5. Assessing claim evidence...")
    fact_result = verify_event_facts(quality_result)
    fact_result = assess_headline_agreement(fact_result)
    fact_result = apply_evidence_assessment(fact_result)
    fact_result["news_collected_at_utc"] = news_data.get("collected_at_utc")

    print("\n6. Analyzing with OpenAI...")
    analysis = analyze_market(market_data, fact_result)

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "market_data": market_data,
        "news_count": news_data["article_count"],
        "news_errors": news_data.get("errors", []),
        "news_collection_statistics": news_data.get("collection_statistics", {}),
        "news_category_coverage": news_data.get("category_coverage", {}),
        "news_diagnostic_articles": news_data.get("diagnostic_articles", []),
        "news_pipeline": {
            "grouped_events": len(
                grouped_news.get("events", [])
            ),
            "accepted_events": len(
                quality_result.get("accepted_events", [])
            ),
            "assessed_events": len(
                fact_result.get("assessed_events", [])
            ),
            "unassigned_articles": grouped_news.get(
                "unassigned_articles", []
            ),
        },
        "news_evidence": fact_result,
        "analysis": analysis,
    }

    Path("reports").mkdir(exist_ok=True)

    filename = (
        "reports/nifty_analysis_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n===== NIFTY AI ANALYSIS =====\n")
    print(
        json.dumps(
            analysis,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nReport saved to: {filename}")


if __name__ == "__main__":
    main()
