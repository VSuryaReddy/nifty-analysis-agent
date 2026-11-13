
import json
from datetime import datetime, timezone
from pathlib import Path

from market_data import fetch_market_snapshot
from news import fetch_news
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

    if news_data["article_count"] == 0:
        raise RuntimeError(
            "No news collected. Analysis stopped."
        )

    print("\n3. Analyzing with OpenAI...")
    analysis = analyze_market(
        market_data,
        news_data,
    )

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "market_data": market_data,
        "news_count": news_data["article_count"],
        "news_errors": news_data["errors"],
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
