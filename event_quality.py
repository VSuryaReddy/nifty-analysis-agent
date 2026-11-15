
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = """
You are a financial news relevance analyst specializing
in the Indian stock market.

OBJECTIVE

Assess whether grouped financial news events contain
meaningful developments that could influence Nifty 50
or Bank Nifty over:

- The next trading session
- The next 2-6 weeks

You are evaluating RELEVANCE ONLY.

Do not independently verify facts, predict market
direction, or resolve conflicting news reports.

INPUT

You will receive events containing:
- Event name and summary
- Event category
- Potential Nifty impact
- Article IDs
- Original source headlines and metadata

RELEVANCE CLASSIFICATION

HIGH:
A development with plausible potential to materially
affect Indian equity markets or market regime.

Examples:
- RBI or Federal Reserve policy developments
- Major inflation or GDP releases
- Significant changes in US Treasury yields
- Major crude oil supply disruptions
- Important Indian regulatory decisions
- Major geopolitical developments

MEDIUM:
A meaningful development affecting market sentiment,
a major index sector, or near-term conditions, but
unlikely to change the broader market regime alone.

Examples:
- Broad banking-sector developments
- Significant global equity-market movements
- Important institutional flow developments

LOW:
Information without a meaningful independent
market-moving development.

Examples:
- Generic investment advice
- Routine market-opening predictions
- Daily trading setups
- Generic market commentary
- Individual stock investment recommendations
- Historical institutional holdings in penny stocks
- Unsupported speculative price forecasts

ARTICLE-LEVEL REVIEW

Evaluate each article within an event independently.

An article should be retained only if its headline
supports the event's relevance.

If an event contains relevant and irrelevant articles,
retain the relevant article IDs and reject the others.

Do not treat all articles in an event as relevant
merely because the event itself appears important.

Do not invent article IDs.

DECISION RULES

1. HIGH and MEDIUM events are accepted.
2. LOW events are rejected.
3. An event with no relevant supporting articles
   must be rejected.
4. Exclude unrelated or low-value articles even
   when grouped into an otherwise relevant event.
5. Use only the supplied headlines and metadata.
6. Do not describe headline claims as verified facts.
7. Do not change event names, summaries, or impact
   classifications in this phase.
8. Do not create new events or split existing events.
9. Every input event must receive exactly one assessment.
10. Return JSON only.

OUTPUT FORMAT

{
  "assessments": [
    {
      "event_index": 0,
      "relevance": "HIGH",
      "reason": "Material economic development",
      "relevant_article_ids": [0, 1],
      "rejected_article_ids": [2]
    }
  ]
}
"""


def assess_event_quality(grouped_news):
    """
    Assess the market relevance of grouped news events.

    This function performs relevance filtering only.
    It does not verify factual claims.
    """

    events = grouped_news.get("events", [])

    if not isinstance(events, list):
        raise ValueError("events must be a list")

    if not events:
        return {
            "accepted_events": [],
            "rejected_events": [],
            "errors": []
        }

    client = OpenAI()

    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-4.1-mini"
    )

    # Provide only the information needed for relevance review.
    review_input = []

    for index, event in enumerate(events):
        review_input.append({
            "event_index": index,
            "event_name": event["event_name"],
            "category": event["category"],
            "summary": event["summary"],
            "potential_nifty_impact": event[
                "potential_nifty_impact"
            ],
            "article_ids": event["article_ids"],
            "sources": [
                {
                    "id": source["id"],
                    "title": source["title"],
                    "source": source["source"],
                    "published_at_utc": source[
                        "published_at_utc"
                    ]
                }
                for source in event["sources"]
            ]
        })

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(
                    review_input,
                    ensure_ascii=False
                )
            }
        ],
        text={
            "format": {
                "type": "json_object"
            }
        }
    )

    output = json.loads(response.output_text)

    if (
        not isinstance(output, dict)
        or not isinstance(output.get("assessments"), list)
    ):
        raise ValueError(
            "AI returned an invalid assessment structure"
        )

    assessments = output["assessments"]

    # Build a lookup by event index.
    assessment_by_index = {}

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("Invalid assessment object")

        event_index = assessment.get("event_index")

        if (
            type(event_index) is not int
            or event_index < 0
            or event_index >= len(events)
            or event_index in assessment_by_index
        ):
            raise ValueError(
                f"Invalid or duplicate event index: {event_index}"
            )

        relevance = assessment.get("relevance")

        if relevance not in ("HIGH", "MEDIUM", "LOW"):
            raise ValueError(
                f"Invalid relevance: {relevance}"
            )

        reason = assessment.get("reason")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"Missing relevance reason: {event_index}"
            )

        relevant_ids = assessment.get(
            "relevant_article_ids"
        )

        rejected_ids = assessment.get(
            "rejected_article_ids"
        )

        if (
            not isinstance(relevant_ids, list)
            or not isinstance(rejected_ids, list)
        ):
            raise ValueError(
                f"Invalid article lists: {event_index}"
            )

        original_ids = set(
            events[event_index]["article_ids"]
        )

        # Explicitly reject duplicate IDs and invalid types.
        all_ids = relevant_ids + rejected_ids

        if any(type(article_id) is not int for article_id in all_ids):
            raise ValueError(
                f"Invalid article ID type: {event_index}"
            )

        if (
            len(all_ids) != len(set(all_ids))
            or set(all_ids) != original_ids
        ):
            raise ValueError(
                f"Incomplete or invalid article assignment "
                f"for event {event_index}"
            )

        assessment_by_index[event_index] = assessment

    if len(assessment_by_index) != len(events):
        raise ValueError(
            "AI did not assess every event"
        )

    accepted_events = []
    rejected_events = []

    for index, event in enumerate(events):
        assessment = assessment_by_index[index]

        relevance = assessment["relevance"]

        relevant_ids = assessment["relevant_article_ids"]
        rejected_ids = assessment["rejected_article_ids"]

        # Preserve the original event without modifying it.
        reviewed_event = dict(event)

        reviewed_event["relevance_assessment"] = {
            "relevance": relevance,
            "reason": assessment["reason"],
            "relevant_article_ids": relevant_ids,
            "rejected_article_ids": rejected_ids
        }

        # LOW events and events with no relevant articles
        # cannot be accepted.
        if (
            relevance == "LOW"
            or not relevant_ids
        ):
            rejected_events.append(reviewed_event)
        else:
            # Downstream consumers should see only
            # article IDs retained by relevance review.
            reviewed_event["article_ids"] = relevant_ids

            reviewed_event["sources"] = [
                source
                for source in event["sources"]
                if source["id"] in set(relevant_ids)
            ]

            accepted_events.append(reviewed_event)

    return {
        "accepted_events": accepted_events,
        "rejected_events": rejected_events,
        "errors": []
    }


if __name__ == "__main__":
    from news import fetch_news
    from news_intelligence import group_news_into_events

    news_data = fetch_news()

    print(
        "Collected articles:",
        news_data["article_count"]
    )

    grouped_news = group_news_into_events(news_data)

    print(
        "Grouped events:",
        len(grouped_news["events"])
    )

    result = assess_event_quality(grouped_news)

    print(
        "Accepted events:",
        len(result["accepted_events"])
    )

    print(
        "Rejected events:",
        len(result["rejected_events"])
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )
