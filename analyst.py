
import json
import math
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = """
You are a financial market research analyst specializing
in the Indian Nifty 50 index.

Analyze the supplied market data and qualified news evidence
to produce:

1. Next trading session outlook
2. Next 2-6 weeks market outlook

Your objective is to explain the likely market implications
of technical indicators, domestic developments, and global
events.

Do not simply summarize financial news.


==================================================
1. EVIDENCE RULES
==================================================

Use ONLY the supplied market data and qualified news claims.

The news pipeline has already filtered unsupported claims.
Do not reconstruct excluded information from article titles,
event categories, general knowledge, or assumptions.

Every headline-derived claim remains UNVERIFIED.

Evidence classifications:

CORROBORATED:
Multiple preferred publishers report compatible headlines.
This is NOT independent factual verification.

CREDIBLE_SINGLE_SOURCE:
A preferred publisher reports the claim.
Attribute the information to the reporting source.

FORECAST:
A prediction or expectation, not an established outcome.

OPINION:
Commentary or interpretation, not a confirmed fact.

MIXED_CLAIM:
Contains a mixture of reporting and interpretation.
Do not treat the entire claim as an established fact.

CONFLICTING or INSUFFICIENT_EVIDENCE:
Must not be used to support directional conclusions.

Do not claim that information has been independently verified.


==================================================
2. FACTS VS INTERPRETATIONS
==================================================

Separate three kinds of information:

A. MARKET OBSERVATION

A numerical observation explicitly provided by the
market-data collector.

Example:
"Nifty's latest reported daily change was -0.61%."

B. ATTRIBUTED NEWS REPORT

A qualified claim reported by a source.

Example:
"Economists quoted in a report expect a possible
RBI rate hike."

C. ANALYST INTERPRETATION

A possible economic or market implication.

Example:
"Higher interest rates could increase borrowing
costs and pressure equity valuations."

Never turn an interpretation into an established fact.

Never invent the cause of an observed market movement.

For example:

SUPPORTED:
"Banking stocks reportedly rallied."

POSSIBLE INTERPRETATION:
"Banking strength could support the broader index."

NOT SUPPORTED:
"Banking stocks rallied because of strong earnings,
improved liquidity, or FII buying."

Do not assert causation unless the supplied evidence
explicitly establishes it.


==================================================
3. CRITICAL MACROECONOMIC RULES
==================================================

INFLATION:

Do not claim that Indian inflation is high, rising,
falling, or above RBI tolerance unless qualified
evidence or numerical market data establishes it.

A forecast of an RBI rate hike does NOT establish
the current inflation rate or its trend.

RBI POLICY:

Distinguish:
- Confirmed RBI decisions
- Economists' expectations
- Market speculation

A possible rate hike is not a confirmed rate hike.

FII/DII FLOWS:

Do not claim that institutional investors are buying,
selling, or driving market movements unless qualified
evidence explicitly supports those claims.

Banking-sector strength does not prove FII buying.

US TREASURY YIELDS:

A headline reporting a historical yield level does not
establish the current live yield.

Describe the observation using the source date.
Do not infer an ongoing rise without current yield data.

CRUDE OIL:

Use the supplied Brent market-data observation for
the latest available price and change.

Do not infer current oil direction from older headlines.

GLOBAL EQUITIES:

Use the supplied S&P 500 and Nasdaq observations
for the latest available market performance.

Do not describe an older news headline as a current
market development when newer market data differs.


==================================================
4. MARKET DATA AND RECENCY
==================================================

Consider available:
- Nifty price and daily change
- SMA20 and SMA50
- RSI14
- Five-session return
- India VIX
- USD/INR
- Brent crude
- S&P 500
- Nasdaq

Use each market instrument's own as_of date.

Never assume all instruments were updated at the
same time.

The latest daily-bar observation is not necessarily
a confirmed final closing price.

Unless the input explicitly confirms session completion,
use wording such as:

"Latest available Nifty observation"

or:

"Nifty's reported daily-bar value"

Do not describe a value as simultaneously intraday
and a confirmed closing price.

Do not assume a market is currently trading.

Do not interpret RSI above 70 as proof of an
imminent correction.

Technical indicators describe market conditions,
not guaranteed future movements.


==================================================
5. EVENT ANALYSIS
==================================================

Prioritize developments that could materially affect
Nifty over the next session or the next 2-6 weeks.

Consider only evidence supplied for:
- RBI policy and inflation
- Government and regulatory developments
- Institutional activity
- Banking and major sectors
- US monetary policy and bond yields
- Global equity performance
- Crude oil and geopolitical developments
- Currency movements

For each important news-driven event:

1. State the qualified development.
2. Identify whether it is reported, forecast, or opinion.
3. Explain its possible impact on Indian equities.
4. Include supporting article titles and URLs.

Never invent sources.

Do not treat separate developments as one confirmed
causal sequence simply because they belong to the
same news category.

If evidence is uncertain, explain that uncertainty.

Avoid listing low-impact news just to increase the
number of events.


==================================================
6. NEXT TRADING SESSION
==================================================

Consider:
- Nifty momentum and technical conditions
- India VIX
- Available global market observations
- Qualified domestic and global developments
- Relevant upcoming risks, only if supplied

Provide:
- Upward scenario percentage
- Downward scenario percentage
- Sideways scenario percentage

The percentages must total exactly 100.

Do not assume the next calendar day is a trading day.

Do not invent an exact next-session date.


==================================================
7. NEXT 2-6 WEEKS
==================================================

Evaluate potential changes in the broader market regime.

Distinguish:
- Short-term technical movements
- Sustained macroeconomic developments
- Forecasts and expectations
- Unconfirmed risks

Explain both bullish and bearish factors.

Do not let a single speculative headline dominate
the medium-term outlook.


==================================================
8. PROBABILITIES AND UNCERTAINTY
==================================================

Scenario percentages are subjective analytical estimates.

They are NOT statistically calibrated probabilities.

For each horizon, all three percentages must total 100.

Allowed outlook labels:
- Bullish
- Bearish
- Sideways
- Mixed

Use Mixed when competing scenarios are similarly
plausible and there is no clear directional advantage.

Do not manufacture confidence when evidence is weak.

If important information is unavailable, identify it
in missing_information.

Do not claim missing information was analyzed.


==================================================
9. SOURCE AND CLAIM RESTRICTIONS
==================================================

Use only the claims under qualified_news.events[].claims
as news evidence.

Category labels are organizational metadata, not facts.

Source titles and URLs are citation metadata.
Do not extract additional factual claims from their
wording beyond the associated qualified claim.

An article title may contain unsupported details.
Those details must not enter the analysis independently.

Do not use qualified news sources to support claims
that they do not actually establish.

When qualified news is empty, analyze the market data
alone and explicitly identify the news coverage gap.


==================================================
10. OUTPUT REQUIREMENTS
==================================================

Return ONLY valid JSON.

No Markdown or explanations outside JSON.

Use exactly this structure:

{
  "market_summary": "Current market environment",

  "important_events": [
    {
      "event": "Important development",
      "impact": "Positive",
      "reason": "Evidence and potential Nifty impact",
      "sources": [
        {
          "title": "Supplied article title",
          "url": "Supplied article URL"
        }
      ]
    }
  ],

  "next_day": {
    "outlook": "Mixed",
    "up_probability": 35,
    "down_probability": 40,
    "sideways_probability": 25,
    "explanation": "Evidence-based next-session outlook"
  },

  "next_2_6_weeks": {
    "outlook": "Mixed",
    "bullish_probability": 35,
    "bearish_probability": 40,
    "sideways_probability": 25,
    "explanation": "Evidence-based medium-term outlook"
  },

  "key_risks": [
    "Risk that could materially change the outlook"
  ],

  "missing_information": [
    "Important unavailable or unverified input"
  ]
}

The example percentages are placeholders.
Also include upcoming_high_impact_events as specified in section 12.
Derive your own scenario estimates from the supplied
evidence.

Impact must be one of:
Positive, Negative, Mixed, Uncertain.

Prioritize accuracy and meaningful interpretation
over the number of reported events.

Never invent facts to make the report appear complete.
"""

SYSTEM_PROMPT += """

==================================================
11. TEMPORAL EVIDENCE PRIORITY
==================================================

Use the analysis timestamp and each market instrument's
as_of date to establish which observations are newer.

RULE 1: COMPLETED OBSERVATIONS OVERRIDE PRE-MARKET SIGNALS

If a news headline predicts how a market may open on a
particular date, but the supplied market-data collector
already contains an observation for that date or later,
do not use the old pre-market prediction as a forward-
looking signal.

Example:

A November 13 headline predicts a positive US market open.
The collector also supplies November 13 S&P 500 and Nasdaq
observations.

Use the collector's observations to assess the market.
Do not use the earlier opening prediction as evidence
of positive sentiment for the next trading session.

A pre-market forecast is not the final market outcome.


RULE 2: HISTORICAL LEVELS ARE NOT CURRENT LEVELS

A headline reporting a Treasury yield, inflation figure,
currency rate, or index level on an earlier date establishes
only what the source reported at that time.

Do not describe it as the current value without a newer
observation.

For example:

A November 10 headline reports US 10-year yields above 5%.

Acceptable:
"A November 10 report indicated yields had exceeded 5%."

Not acceptable:
"US Treasury yields are currently above 5%."

If the current level is unavailable, explicitly say so.


RULE 3: FORECASTS REMAIN CONDITIONAL

A forecast of an RBI rate hike, GDP growth, market rally,
or policy change is not confirmation of the predicted event.

Do not infer the underlying economic conditions solely
from the forecast.

Explain the potential impact conditionally.


RULE 4: MARKET SESSION LANGUAGE

Use "latest available observation" or "reported daily-bar
value" unless the input explicitly confirms the market's
final closing price.

Do not say Nifty is currently trading when the observation
comes from an earlier session.

Do not describe the market as open or closed based solely
on an old daily-bar observation.


RULE 5: DIRECTIONAL REASONING

Prefer newer numerical market observations when evaluating
recent price direction.

Historical news may provide macroeconomic context, but
must not override newer market observations for the same
instrument and period.

When current information is missing, acknowledge the gap
rather than extending historical observations into the
present.
"""



SYSTEM_PROMPT += """

==================================================
12. UPCOMING HIGH-IMPACT EVENTS — WATCHLIST ONLY
==================================================

Return upcoming_high_impact_events as a separate list.
This is an informational caution/watchlist, NOT a forecast.

Use ONLY qualified_news.events[].claims and their attached
sources. Include an event only when a qualified claim explicitly
supports BOTH its scheduled calendar date and its occurrence.
Do not infer event dates from general knowledge, or invent
future meetings, announcements, or earnings dates.

Use analysis_time_ist as the report date. Express event dates
as YYYY-MM-DD in the Asia/Kolkata calendar. Include an event
when event_date >= report date, including events happening today.
Exclude events whose event_date is before the report date.
Do not use the event's time of day to exclude it.

Include only material events, such as scheduled FOMC or RBI
policy decisions, major macroeconomic releases, or important
Nifty constituent earnings. Explain briefly why to watch.
Do not predict the outcome or treat it as already completed.

Scheduled future event outcomes must NOT influence the
next_day or next_2_6_weeks directional percentages. Do not
change those probabilities based solely on the calendar.
Existing, qualified market expectations remain separate evidence.

Every listed event must include at least one source attached
to a retained qualified claim. When none qualify, return [].

Add this top-level JSON property:
"upcoming_high_impact_events": [
  {
    "event": "Scheduled event name",
    "date": "YYYY-MM-DD",
    "why_watch": "Potential relevance, without predicting outcome",
    "sources": [
      {"title": "Supplied article title", "url": "Supplied URL"}
    ]
  }
]
"""


ALLOWED_EVIDENCE_STATUSES = {
    "CORROBORATED",
    "CREDIBLE_SINGLE_SOURCE",
    "FORECAST",
    "OPINION",
    "MIXED_CLAIM",
}

ALLOWED_OUTLOOKS = {
    "Bullish",
    "Bearish",
    "Sideways",
    "Mixed",
}

ALLOWED_IMPACTS = {
    "Positive",
    "Negative",
    "Mixed",
    "Uncertain",
}


def is_superseded_premarket_signal(claim, market_data, sources):
    """
    Exclude an old US equity pre-market opening signal
    when same-day or newer US index data is available.

    This intentionally handles only a narrow, identifiable
    case. Other news remains subject to the analyst's
    temporal-evidence instructions.
    """

    claim_text = claim.get("claim", "").lower()

    is_us_premarket_signal = (
        any(
            term in claim_text
            for term in (
                "futures",
                "pre-market",
                "premarket",
            )
        )
        and any(
            term in claim_text
            for term in (
                "dow jones",
                "s&p 500",
                "nasdaq",
                "wall street",
                "us market",
                "u.s. market",
            )
        )
        and any(
            term in claim_text
            for term in (
                "open",
                "opening",
            )
        )
    )

    if not is_us_premarket_signal:
        return False

    markets = market_data.get("markets", {})

    observation_dates = []

    for symbol in ("sp500", "nasdaq"):
        observation = markets.get(symbol, {})
        date_text = observation.get("as_of")

        if not isinstance(date_text, str):
            continue

        try:
            observation_dates.append(
                datetime.fromisoformat(
                    date_text.replace("Z", "+00:00")
                ).date()
            )
        except ValueError:
            continue

    if not observation_dates:
        return False

    latest_observation_date = max(observation_dates)

    supporting_dates = []

    for article_id in claim.get(
        "supporting_article_ids", []
    ):
        source = sources.get(article_id)

        if not source:
            continue

        published_at = source.get("published_at_utc")

        if not published_at:
            return False

        try:
            published_date = datetime.fromisoformat(
                published_at.replace("Z", "+00:00")
            ).date()
        except (ValueError, TypeError):
            return False

        supporting_dates.append(published_date)

    if not supporting_dates:
        return False

    return all(
        published_date <= latest_observation_date
        for published_date in supporting_dates
    )


def prepare_qualified_news(fact_result, market_data):
    """
    Convert assessed news into conservative claim-level
    evidence for the analyst.

    Exclude unsupported or conflicting claims.
    """

    qualified_events = []
    excluded_claims = 0

    for event in fact_result.get("assessed_events", []):
        sources = {
            source["id"]: source
            for source in event.get("sources", [])
        }

        qualified_claims = []

        claims = event.get(
            "fact_assessment", {}
        ).get("claims", [])

        for claim in claims:
            if is_superseded_premarket_signal(
                claim,
                market_data,
                sources
            ):
                excluded_claims += 1
                continue
            assessment = claim.get(
                "evidence_assessment", {}
            )

            status = assessment.get(
                "evidence_status",
                "INSUFFICIENT_EVIDENCE"
            )

            grounding = claim.get(
                "headline_grounding"
            )

            if (
                status not in ALLOWED_EVIDENCE_STATUSES
                or grounding != "SUPPORTED_BY_HEADLINE"
            ):
                excluded_claims += 1
                continue

            supporting_sources = []

            for article_id in claim.get(
                "supporting_article_ids", []
            ):
                source = sources.get(article_id)

                if not source:
                    continue

                supporting_sources.append({
                    "title": source["title"],
                    "url": source["url"],
                    "published_at_utc": source[
                        "published_at_utc"
                    ],
                    "publisher": source.get(
                        "source", ""
                    ),
                })

            if not supporting_sources:
                excluded_claims += 1
                continue

            qualified_claims.append({
                "claim": claim["claim"],
                "claim_type": claim["claim_type"],
                "evidence_status": status,
                "verification_status": "UNVERIFIED",
                "sources": supporting_sources,
            })

        if qualified_claims:
            qualified_events.append({
                "category": event.get("category"),
                "relevance": event.get(
                    "relevance_assessment", {}
                ).get("relevance"),
                "claims": qualified_claims,
            })

    return {
        "events": qualified_events,
        "excluded_claim_count": excluded_claims,
        "limitations": [
            "News is assessed using RSS headlines only.",
            "Headline agreement is not independent "
            "fact verification.",
            "Unsupported and conflicting claims are "
            "excluded from directional reasoning.",
        ],
    }


def validate_probability_group(
    section,
    probability_fields,
    section_name
):
    """
    Validate the three scenario percentages.
    """

    values = []

    for field in probability_fields:
        value = section.get(field)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"{section_name}.{field} must be numeric"
            )

        if not 0 <= value <= 100:
            raise ValueError(
                f"{section_name}.{field} must be "
                "between 0 and 100"
            )

        values.append(value)

    if not math.isclose(
        sum(values),
        100,
        rel_tol=0,
        abs_tol=0.000001
    ):
        raise ValueError(
            f"{section_name} probabilities must total 100. "
            f"Received {sum(values)}"
        )


def claim_explicitly_supports_date(claim_text, event_date):
    """Conservative literal-date check; does not infer relative dates."""
    if not isinstance(claim_text, str):
        return False

    # Match a complete ISO date, not a substring of another date.
    iso_pattern = rf"(?<![\d-]){re.escape(event_date.isoformat())}(?![\d-])"
    if re.search(iso_pattern, claim_text):
        return True

    # Explicit English dates with a four-digit year, in either order.
    month_names = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
    month = month_names[event_date.month - 1]
    day = str(event_date.day)
    year = str(event_date.year)
    patterns = (
        rf"(?<!\w){day}(?:st|nd|rd|th)?\s+{month}\s*,?\s*{year}(?!\d)",
        rf"(?<!\w){month}\s+{day}(?:st|nd|rd|th)?\s*,?\s*{year}(?!\d)",
    )
    return any(re.search(pattern, claim_text, re.IGNORECASE) for pattern in patterns)


def validate_analysis(analysis, qualified_news, market_data=None, report_date=None):
    """
    Validate JSON structure, probabilities and citations.

    This is structural validation, not independent
    verification of the analyst's narrative.
    """

    if not isinstance(analysis, dict):
        raise ValueError(
            "AI analysis must be a JSON object"
        )

    required_fields = {
        "market_summary",
        "important_events",
        "next_day",
        "next_2_6_weeks",
        "key_risks",
        "missing_information",
        "upcoming_high_impact_events",
    }

    missing = required_fields - analysis.keys()

    if missing:
        raise ValueError(
            f"Missing analysis fields: {sorted(missing)}"
        )

    if not isinstance(
        analysis["market_summary"], str
    ):
        raise ValueError(
            "market_summary must be a string"
        )

    for field in (
        "important_events",
        "key_risks",
        "missing_information",
        "upcoming_high_impact_events",
    ):
        if not isinstance(analysis[field], list):
            raise ValueError(
                f"{field} must be a list"
            )

    for field in (
        "key_risks",
        "missing_information",
    ):
        if not all(
            isinstance(item, str)
            for item in analysis[field]
        ):
            raise ValueError(
                f"{field} must contain strings"
            )

    next_day = analysis["next_day"]
    medium_term = analysis["next_2_6_weeks"]

    for section_name, section in (
        ("next_day", next_day),
        ("next_2_6_weeks", medium_term),
    ):
        if not isinstance(section, dict):
            raise ValueError(
                f"{section_name} must be an object"
            )

        if section.get("outlook") not in ALLOWED_OUTLOOKS:
            raise ValueError(
                f"Invalid {section_name} outlook"
            )

        if not isinstance(
            section.get("explanation"), str
        ):
            raise ValueError(
                f"{section_name}.explanation must be a string"
            )

    validate_probability_group(
        next_day,
        (
            "up_probability",
            "down_probability",
            "sideways_probability",
        ),
        "next_day",
    )

    validate_probability_group(
        medium_term,
        (
            "bullish_probability",
            "bearish_probability",
            "sideways_probability",
        ),
        "next_2_6_weeks",
    )

    # Only source pairs attached to retained claims
    # may appear in the final analysis.
    allowed_sources = set()

    for event in qualified_news["events"]:
        for claim in event["claims"]:
            for source in claim["sources"]:
                allowed_sources.add((
                    source["title"],
                    source["url"],
                ))

    for index, event in enumerate(
        analysis["important_events"]
    ):
        if not isinstance(event, dict):
            raise ValueError(
                f"important_events[{index}] must be an object"
            )

        if not isinstance(event.get("event"), str):
            raise ValueError(
                f"important_events[{index}].event "
                "must be a string"
            )

        if not isinstance(event.get("reason"), str):
            raise ValueError(
                f"important_events[{index}].reason "
                "must be a string"
            )

        if event.get("impact") not in ALLOWED_IMPACTS:
            raise ValueError(
                f"Invalid impact for event {index}"
            )

        sources = event.get("sources")

        if not isinstance(sources, list):
            raise ValueError(
                f"Event {index} sources must be a list"
            )

        if not sources:
            raise ValueError(
                f"Event {index} has no supporting sources"
            )

        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(
                    f"Invalid source object in event {index}"
                )

            source_pair = (
                source.get("title"),
                source.get("url"),
            )

            if source_pair not in allowed_sources:
                raise ValueError(
                    "AI used a source that is not present "
                    "in the qualified news evidence: "
                    f"{source_pair}"
                )
    # Date-based watchlist filtering, independent of the model's
    # interpretation of whether today's event has finished.
    report_date = (
        report_date or datetime.now(ZoneInfo("Asia/Kolkata")).date()
    )
    retained_upcoming = []
    for index, event in enumerate(analysis["upcoming_high_impact_events"]):
        if not isinstance(event, dict):
            raise ValueError(f"Upcoming event {index} must be an object")
        for field in ("event", "date", "why_watch"):
            if not isinstance(event.get(field), str) or not event[field].strip():
                raise ValueError(f"Upcoming event {index}: invalid {field}")
        try:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"Upcoming event {index}: invalid date") from exc
        if event_date.isoformat() != event["date"]:
            raise ValueError(f"Upcoming event {index}: date must be YYYY-MM-DD")
        sources = event.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Upcoming event {index} requires sources")
        for source in sources:
            if not isinstance(source, dict) or (
                source.get("title"), source.get("url")
            ) not in allowed_sources:
                raise ValueError(f"Upcoming event {index}: unqualified source")
        # The SAME retained claim must contain the date and cite one
        # of this event's sources. Source membership alone is insufficient.
        cited_pairs = {(source["title"], source["url"]) for source in sources}
        date_supported = any(
            claim_explicitly_supports_date(claim.get("claim"), event_date)
            and any(
                (source.get("title"), source.get("url")) in cited_pairs
                for source in claim.get("sources", [])
            )
            for qualified_event in qualified_news["events"]
            for claim in qualified_event["claims"]
        )
        if not date_supported:
            raise ValueError(
                f"Upcoming event {index}: scheduled date not supported "
                "by a cited qualified claim"
            )
        if event_date >= report_date:
            retained_upcoming.append(event)
    analysis["upcoming_high_impact_events"] = retained_upcoming

    if market_data is not None:
        missing_markets = set(
            market_data.get("errors", {})
        )

        reported_missing = " ".join(
            analysis["missing_information"]
        ).lower()

        for market_name in missing_markets:
            readable_name = market_name.replace("_", " ")

            if (
                market_name.lower() not in reported_missing
                and readable_name.lower() not in reported_missing
            ):
                raise ValueError(
                    f"AI did not acknowledge missing "
                    f"market input: {market_name}"
                )    
    
    return analysis


def analyze_market(market_data, fact_result):
    """
    Generate the Nifty analysis using market data
    and qualified news evidence.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    now_ist = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    market_session_note = (
        "Market-data values are daily-bar observations. "
        "Their as_of dates must be respected. "
        "Do not call a value a confirmed final close "
        "unless session completion is established."
    )

    qualified_news = prepare_qualified_news(
    fact_result,
    market_data
    )

    evidence = {
        "analysis_time_ist": now_ist.isoformat(),
        "market_session_note": market_session_note,
        "market_data": market_data,
        "qualified_news": qualified_news,
    }

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=os.getenv(
            "OPENAI_MODEL",
            "gpt-4.1-mini"
        ),
        instructions=SYSTEM_PROMPT,
        input=(
            "Analyze the supplied Nifty market evidence. "
            "Return the required JSON object.\n\n"
            + json.dumps(
                evidence,
                ensure_ascii=False,
            )
        ),
        text={
            "format": {
                "type": "json_object"
            }
        },
    )

    analysis = json.loads(
        response.output_text
    )

    return validate_analysis(
        analysis,
        qualified_news,
        market_data,
        now_ist.date(),
    )
