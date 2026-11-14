
import json
import os
import re


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = """
You are a financial news evidence analyst.

OBJECTIVE

Extract and assess important factual claims from
market-relevant news events.

You have access ONLY to event summaries and
original source headlines.

You are NOT performing independent fact verification.

RULES

1. Extract material claims about:
   - Economic data and numerical figures
   - Central bank or government decisions
   - Bond yields and interest rates
   - Oil prices and supply developments
   - Institutional investment flows
   - Significant global market movements

2. Separate actual reported developments from:
   - Forecasts
   - Expectations
   - Opinions

3. For each claim, determine headline grounding:

   SUPPORTED_BY_HEADLINE:
   The original headline explicitly states the claim.

   PARTIALLY_SUPPORTED:
   The headline supports only part of the claim.

   NOT_SUPPORTED:
   The claim is absent from the supplied headlines
   or adds details not established by them.

4. A headline supporting a claim does NOT prove
   that the claim is factually true.

5. Never mark a claim VERIFIED based on headlines.

6. Do not invent figures, dates, article IDs,
   evidence, or source details.

7. Extract at most five material claims per event.

8. Preserve the distinction between forecasts
   and actual outcomes.

9. Return exactly one assessment per input event.

OUTPUT JSON

{
  "assessments": [
    {
      "event_index": 0,
      "claims": [
        {
          "claim": "US 10-year Treasury yield exceeded 5%",
          "claim_type": "reported",
          "headline_grounding": "SUPPORTED_BY_HEADLINE",
          "supporting_article_ids": [16],
          "reason": "The headline explicitly states this"
        }
      ]
    }
  ]
}
"""


def verify_event_facts(quality_result):
    """
    Extract material claims and assess headline grounding.

    IMPORTANT:
    This does not independently verify factual accuracy.
    All claims remain UNVERIFIED until checked against
    independent evidence.
    """

    events = quality_result.get("accepted_events", [])

    if not isinstance(events, list):
        raise ValueError("accepted_events must be a list")

    if not events:
        return {
            "assessed_events": [],
            "errors": []
        }

    review_input = []

    for index, event in enumerate(events):
        review_input.append({
            "event_index": index,
            "event_name": event["event_name"],
            "summary": event["summary"],
            "information_type": event["information_type"],
            "sources": [
                {
                    "id": source["id"],
                    "title": source["title"],
                    "published_at_utc": source[
                        "published_at_utc"
                    ]
                }
                for source in event["sources"]
            ]
        })

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

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
        raise ValueError("Invalid AI assessment structure")

    assessments = output["assessments"]
    assessment_map = {}

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("Invalid assessment object")

        index = assessment.get("event_index")

        if (
            type(index) is not int
            or index < 0
            or index >= len(events)
            or index in assessment_map
        ):
            raise ValueError(
                f"Invalid or duplicate event index: {index}"
            )

        claims = assessment.get("claims")

        if not isinstance(claims, list) or len(claims) > 5:
            raise ValueError(
                f"Invalid claims list for event {index}"
            )

        valid_ids = {
            source["id"]
            for source in events[index]["sources"]
        }

        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("Invalid claim object")

            if (
                not isinstance(claim.get("claim"), str)
                or not claim["claim"].strip()
            ):
                raise ValueError("Missing claim text")

            if claim.get("claim_type") not in (
                "reported", "forecast", "opinion", "mixed"
            ):
                raise ValueError("Invalid claim type")

            if claim.get("headline_grounding") not in (
                "SUPPORTED_BY_HEADLINE",
                "PARTIALLY_SUPPORTED",
                "NOT_SUPPORTED"
            ):
                raise ValueError("Invalid grounding status")

            article_ids = claim.get(
                "supporting_article_ids"
            )

            if not isinstance(article_ids, list):
                raise ValueError("Invalid supporting IDs")

            if (
                any(type(aid) is not int for aid in article_ids)
                or len(article_ids) != len(set(article_ids))
                or not set(article_ids).issubset(valid_ids)
            ):
                raise ValueError(
                    "Invalid supporting article references"
                )

            if (
                claim["headline_grounding"]
                == "SUPPORTED_BY_HEADLINE"
                and not article_ids
            ):
                raise ValueError(
                    "Supported claim requires source references"
                )

            if not isinstance(claim.get("reason"), str):
                raise ValueError("Missing claim reason")

            # No independent evidence is checked here.
            claim["verification_status"] = "UNVERIFIED"

        assessment_map[index] = claims

    if len(assessment_map) != len(events):
        raise ValueError("Not every event was assessed")

    assessed_events = []

    for index, event in enumerate(events):
        reviewed_event = dict(event)

        reviewed_event["fact_assessment"] = {
            "verification_status": "UNVERIFIED",
            "claims": assessment_map[index]
        }

        assessed_events.append(reviewed_event)

    return {
        "assessed_events": assessed_events,
        "errors": []
    }


AGREEMENT_PROMPT = """
You are reviewing financial news evidence.

You receive claims and their supporting RSS headlines.

Your job is to assess headline agreement, not factual truth.

For each claim, return exactly one assessment.

Allowed agreement statuses:

AGREE:
At least two distinct publisher identities explicitly
support the same material claim, including its figures,
dates, and scope. Do not infer agreement merely because
the headlines discuss the same topic.

SINGLE_SUPPORT:
Only one publisher identity explicitly supports the
specific claim, or the remaining headlines discuss
different aspects of the topic.

CONFLICT:
Two or more headlines make materially incompatible
claims about the same metric, event, and reference period.

UNCLEAR:
The available headlines do not establish agreement
or conflict sufficiently.

Important:
- Price rises and falls on different dates are not
  automatically contradictory.
- Predictions are not confirmed outcomes.
- Opinions are not established facts.
- Do not invent dates, figures, or article details.
- Do not assume separate publishers used independent
  original reporting.
- Do not use outside knowledge.
- If a claim combines multiple developments, require
  support for the complete claim.
- Return JSON only.

Output:
{
  "assessments": [
    {
      "claim_id": "0:0",
      "agreement": "SINGLE_SUPPORT",
      "reason": "Only one headline supports the exact claim"
    }
  ]
}
"""


def assess_headline_agreement(fact_result):
    """
    Assess semantic agreement for all extracted claims
    in one batch.

    This is headline comparison, not fact verification.
    """

    review_input = []
    claim_lookup = {}

    for event_index, event in enumerate(
        fact_result["assessed_events"]
    ):
        sources_by_id = {
            source["id"]: source
            for source in event["sources"]
        }

        claims = event["fact_assessment"]["claims"]

        for claim_index, claim in enumerate(claims):
            claim_id = f"{event_index}:{claim_index}"
            claim_lookup[claim_id] = claim

            supporting_sources = []

            for article_id in claim.get(
                "supporting_article_ids", []
            ):
                source = sources_by_id.get(article_id)

                if source:
                    supporting_sources.append({
                        "id": article_id,
                        "publisher": normalize_publisher(
                            source
                        ),
                        "title": source["title"],
                        "published_at_utc": source[
                            "published_at_utc"
                        ]
                    })

            review_input.append({
                "claim_id": claim_id,
                "claim": claim["claim"],
                "claim_type": claim["claim_type"],
                "headline_grounding": claim[
                    "headline_grounding"
                ],
                "supporting_sources": supporting_sources
            })

    if not review_input:
        return fact_result

    client = OpenAI()
    model = os.getenv(
        "OPENAI_MODEL", "gpt-4.1-mini"
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": AGREEMENT_PROMPT
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
    assessments = output.get("assessments")

    if not isinstance(assessments, list):
        raise ValueError(
            "Invalid agreement assessment structure"
        )

    allowed = {
        "AGREE",
        "SINGLE_SUPPORT",
        "CONFLICT",
        "UNCLEAR"
    }

    seen = set()

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("Invalid assessment object")

        claim_id = assessment.get("claim_id")
        agreement = assessment.get("agreement")
        reason = assessment.get("reason")

        if (
            claim_id not in claim_lookup
            or claim_id in seen
            or agreement not in allowed
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(
                f"Invalid agreement assessment: {claim_id}"
            )

        seen.add(claim_id)

        claim_lookup[claim_id]["headline_agreement"] = {
            "status": agreement,
            "reason": reason
        }

    if seen != set(claim_lookup):
        raise ValueError(
            "Not every claim received an agreement assessment"
        )

    return fact_result




PREFERRED_PUBLISHERS = {
    "reuters",
    "bloomberg",
    "associated press",
    "financial times",
    "wall street journal",
    "cnbc",
    "cnbc tv18",
    "economic times",
    "business standard",
    "mint",
    "the hindu businessline",
    "the hindu",
    "times of india",
    "moneycontrol",
    "ndtv profit",
    "business today",
    "upstox",
    "dd news",
    "reserve bank of india",
    "mospi",
    "us treasury",
}

PUBLISHER_ALIASES = {
    "wsj": "wall street journal",
    "wsj.com": "wall street journal",
    "reuters.com": "reuters",
    "bloomberg.com": "bloomberg",
    "ap news": "associated press",
    "cnbctv18.com": "cnbc tv18",
    "the economic times": "economic times",
    "bfsi.economictimes.indiatimes.com": "economic times",
    "economictimes.indiatimes.com": "economic times",
    "livemint": "mint",
    "livemint.com": "mint",
    "moneycontrol.com": "moneycontrol",
    "upstox.com": "upstox",
    "ddnews.gov.in": "dd news",
    "the times of india": "times of india",
}

def normalize_publisher(source):
    name = str(source.get("source") or "").lower().strip()
    name = re.sub(r"^www\.", "", name)
    return PUBLISHER_ALIASES.get(name, name)


def classify_publisher(source):
    publisher = normalize_publisher(source)
    return {
        "publisher": publisher,
        "source_tier": (
            "PREFERRED"
            if publisher in PREFERRED_PUBLISHERS
            else "UNRATED"
        )
    }




def assess_claim_evidence(claim, sources):
    source_by_id = {
        source["id"]: source
        for source in sources
    }

    supporting_sources = [
        source_by_id[article_id]
        for article_id in claim.get(
            "supporting_article_ids", []
        )
        if article_id in source_by_id
    ]

    publishers = {}
    
    for source in supporting_sources:
        info = classify_publisher(source)
        publishers[info["publisher"]] = info["source_tier"]

    preferred = sorted(
        publisher
        for publisher, tier in publishers.items()
        if tier == "PREFERRED"
    )

    unrated = sorted(
        publisher
        for publisher, tier in publishers.items()
        if tier == "UNRATED"
    )

    agreement = claim.get(
    "headline_agreement", {}
    ).get("status", "UNCLEAR")

    # Defensive check:
    # AGREE requires at least two distinct publishers.
    if agreement == "AGREE" and len(publishers) < 2:
        agreement = "UNCLEAR"

    grounding = claim.get("headline_grounding")
    claim_type = claim.get("claim_type")

    if agreement == "CONFLICT":
        status = "CONFLICTING"

    elif grounding != "SUPPORTED_BY_HEADLINE":
        status = "INSUFFICIENT_EVIDENCE"

    elif claim_type == "forecast":
        status = "FORECAST"

    elif claim_type == "opinion":
        status = "OPINION"

    elif claim_type == "mixed":
        status = "MIXED_CLAIM"

    elif (
        agreement == "AGREE"
        and len(preferred) >= 2
    ):
        status = "CORROBORATED"

    elif preferred and agreement in (
        "AGREE", "SINGLE_SUPPORT"
    ):
        status = "CREDIBLE_SINGLE_SOURCE"

    else:
        status = "INSUFFICIENT_EVIDENCE"

    return {
        "evidence_status": status,
        "headline_agreement": agreement,
        "preferred_publishers": preferred,
        "unrated_publishers": unrated,
        "supporting_source_count": len(
            supporting_sources
        ),
        "limitations": [
            "Based on supplied RSS headlines only.",
            "Publisher preference does not prove accuracy.",
            "Independent original reporting is not confirmed.",
            "Claims have not been independently fact-checked."
        ]
    }


def apply_evidence_assessment(fact_result):
    for event in fact_result["assessed_events"]:
        claims = event["fact_assessment"]["claims"]

        for claim in claims:
            claim["evidence_assessment"] = (
                assess_claim_evidence(
                    claim,
                    event["sources"]
                )
            )

            claim["verification_status"] = "UNVERIFIED"

        event["fact_assessment"][
            "assessment_method"
        ] = "HEADLINE_SOURCE_AND_AGREEMENT"

        event["fact_assessment"][
            "verification_status"
        ] = "UNVERIFIED"

    return fact_result




if __name__ == "__main__":
    from news import fetch_news
    from news_intelligence import group_news_into_events
    from event_quality import assess_event_quality

    news_data = fetch_news()

    grouped_news = group_news_into_events(news_data)

    quality_result = assess_event_quality(grouped_news)

    result = verify_event_facts(quality_result)
    result = assess_headline_agreement(result)
    result = apply_evidence_assessment(result)

    print("Collected articles:", news_data["article_count"])
    print("Grouped events:", len(grouped_news["events"]))
    print(
        "Accepted events:",
        len(quality_result["accepted_events"])
    )
    print(
        "Assessed events:",
        len(result["assessed_events"])
    )

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ))
