
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from news import fetch_news
from news_freshness import is_current_article
from datetime import datetime


load_dotenv()


def group_news_into_events(news_data):
    reference_time = datetime.fromisoformat(news_data["collected_at_utc"])
    if reference_time.tzinfo is None:
        raise ValueError("News collection time requires UTC timezone")
    articles = [a for a in news_data.get("articles", [])
                if is_current_article(a, reference_time)]

    if not articles:
        return {
            "events": [],
            "ignored_articles": [],
            "unassigned_articles": [],
            "errors": ["No news articles available"],
        }


    client = OpenAI()

    model = os.getenv(
        "OPENAI_MODEL", "gpt-4.1-mini"
    )

    # Assign stable IDs for linking events to sources
    indexed_articles = [
        {
            **article,
            "id": index,
            "category": article["category"],
            "title": article["title"],
            "source": article["source"],
            "published_at_utc": article["published_at_utc"],
            "url": article["url"],
        }
        for index, article in enumerate(articles)
    ]

    
    system_prompt = """
You are a financial news intelligence assistant
specializing in the Indian stock market.

OBJECTIVE

Analyze the supplied news headlines and identify
meaningful economic, geopolitical, and financial
market developments that could materially influence
Nifty 50 or Bank Nifty over:

- The next trading session
- The next 2-6 weeks

Your task is EVENT IDENTIFICATION AND GROUPING,
not factual verification or market prediction.

You have access only to the supplied headlines
and their metadata, not full article content.

RULES

1. EVENT RELEVANCE

Identify only developments that could meaningfully
affect Indian equity markets.

Prioritize:
- Indian inflation, GDP, RBI policy and bond yields
- Institutional flows and banking-sector developments
- US inflation, Federal Reserve policy and Treasury yields
- Crude oil supply, demand and geopolitical disruptions
- Significant global equity-market developments
- Major Indian regulatory or government policy changes

Exclude:
- Routine daily market commentary and trade setups
- Generic investment advice and promotional content
- Individual stock stories without broader market impact
- Historical institutional holdings in individual stocks
- Generic price predictions without new evidence
- Headlines without an identifiable material development

Place excluded articles in ignored_articles.

2. EVENT GROUPING

Group articles only when they describe the same
underlying development.

Do not group articles simply because they discuss
the same country, sector, commodity or broad topic.

Do not combine opposing or unrelated developments
into a single directional event.

For example:
- An oil-price decline caused by easing geopolitical
  tensions is different from an oil-price increase
  caused by supply disruptions.
- A US market sell-off is different from a
  subsequent market recovery.

Prefer a smaller number of meaningful events
over broad topic summaries.

3. FACTS, FORECASTS AND OPINIONS

Clearly distinguish:
- Reported developments or published figures
- Forecasts and analyst expectations
- Opinions and speculation

Use information_type:
- reported
- forecast
- opinion
- mixed

Never describe forecasts or expectations as
confirmed policy decisions or actual outcomes.

When an event contains both reported information
and forecasts, clearly separate them in the summary.

4. SOURCE GROUNDING

Use only information explicitly supported by
the supplied headlines and metadata.

Do not invent:
- Economic figures or market prices
- Government or central-bank decisions
- Dates or announcements
- Sources or article IDs
- Causes not established by the headlines

Multiple similar headlines do not automatically
confirm that their claims are true.

Treat all headline claims as unverified until
a separate verification step is completed.

5. POTENTIAL NIFTY IMPACT

For each relevant event, classify its potential
impact on Nifty as:

- positive
- negative
- mixed
- uncertain

Explain the possible economic transmission
mechanism in impact_reason.

For example:
Higher US Treasury yields could increase
pressure on emerging-market capital flows.

Impact classifications are hypotheses,
not verified market predictions.

If the direction cannot reasonably be inferred
from the headlines, use uncertain.

6. ARTICLE ASSIGNMENT

Every supplied article ID must appear exactly once:

- In one event's article_ids, OR
- In ignored_articles

Never assign an article to multiple events.

Never invent article IDs.

If an article does not provide sufficient
information to identify a relevant development,
place it in ignored_articles.

7. OUTPUT FORMAT

Return valid JSON only.

Use exactly this structure:

{
  "events": [
    {
      "event_name": "Short descriptive name",
      "category": "india_macro",
      "summary": "Headline-grounded event summary",
      "information_type": "reported",
      "potential_nifty_impact": "uncertain",
      "impact_reason": "Potential impact mechanism",
      "article_ids": [0, 1]
    }
  ],
  "ignored_articles": [2, 3]
}

Preserve the original category of the underlying
articles when assigning event categories.

Do not include Markdown, explanations, or
additional text outside the JSON object.
"""


    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Group these news articles into "
                    "market-relevant events.\n\n"
                    "Return JSON.\n\n"
                    + json.dumps(
                        indexed_articles,
                        ensure_ascii=False,
                    )
                ),
            },
        ],
        text={
            "format": {
                "type": "json_object"
            }
        },
    )

    output = json.loads(response.output_text)
    
    if (
        not isinstance(output, dict)
        or not isinstance(output.get("events"), list)
        or not isinstance(
            output.get("ignored_articles"), list
        )
    ):
        raise ValueError(
            "AI returned an unexpected JSON structure"
        )

    for event in output["events"]:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("article_ids"), list)
        ):
            raise ValueError(
                "AI returned an invalid event structure"
            )

    
    
    # Validate and normalize AI article assignments
    valid_ids = set(range(len(indexed_articles)))
    assigned_ids = set()
    duplicate_ids = set()

    # Process event assignments
    for event in output.get("events", []):
        unique_event_ids = []

        for article_id in event.get("article_ids", []):

            # Reject invalid references
            if (
                type(article_id) is not int
                or article_id not in valid_ids
            ):
                raise ValueError(
                    f"Invalid article ID: {article_id}"
                )

            # Keep the first assignment
            if article_id in assigned_ids:
                duplicate_ids.add(article_id)
                continue

            assigned_ids.add(article_id)
            unique_event_ids.append(article_id)

        event["article_ids"] = unique_event_ids

    # Remove events that lost all article references
    output["events"] = [
        event
        for event in output.get("events", [])
        if event["article_ids"]
    ]

    # Process explicitly ignored articles
    ignored_ids = set()

    for article_id in output.get(
        "ignored_articles", []
    ):
        if (
            type(article_id) is not int
            or article_id not in valid_ids
        ):
            raise ValueError(
                f"Invalid ignored article ID: {article_id}"
            )

        if article_id in assigned_ids:
            duplicate_ids.add(article_id)
            continue

        ignored_ids.add(article_id)

    # Find articles omitted by AI
    missing_ids = valid_ids - assigned_ids - ignored_ids

    # # Add omitted articles to ignored list
    # ignored_ids.update(missing_ids)

    
    # Articles the AI explicitly chose to ignore
    output["ignored_articles"] = sorted(ignored_ids)

    # Articles the AI failed to classify
    output["unassigned_articles"] = sorted(missing_ids)


    # Diagnostic messages
    if duplicate_ids:
        print(
            "Duplicate assignments corrected:",
            sorted(duplicate_ids)
        )

    
    if missing_ids:
        print(
            "Articles not classified by AI:",
            sorted(missing_ids)
        )

    # Attach original source metadata
    for event in output["events"]:
        event["sources"] = [
            indexed_articles[article_id]
            for article_id in event["article_ids"]
        ]

    return output



if __name__ == "__main__":
    news_data = fetch_news()

    print(
        "Collected articles:",
        news_data["article_count"]
    )

    result = group_news_into_events(news_data)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
