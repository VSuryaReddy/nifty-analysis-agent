
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from market_data import MARKETS

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime.now(timezone.utc)

MARKET_TIMEZONES = {
    "nifty_50": "Asia/Kolkata",
    "india_vix": "Asia/Kolkata",
    "usd_inr": "UTC",
    "brent_crude": "America/New_York",
    "sp500": "America/New_York",
    "nasdaq": "America/New_York",
}


def inspect_market(name, symbol):
    print(f"\n=== {name.upper()} ({symbol}) ===")

    data = yf.Ticker(symbol).history(
        period="14d",
        interval="1d",
        auto_adjust=True,
    )

    if data.empty:
        print("ERROR: No data returned")
        return

    closes = data["Close"].dropna()

    if closes.empty:
        print("ERROR: No valid closing prices")
        return

    print("Source timezone:", data.index.tz)
    print("Observation count:", len(closes))
    print("Duplicate dates:", closes.index.has_duplicates)
    print("Dates sorted:", closes.index.is_monotonic_increasing)

    latest_ts = closes.index[-1]
    latest_price = float(closes.iloc[-1])

    print("Latest raw timestamp:", latest_ts)
    print("Latest date label:", latest_ts.date())
    print("Latest price:", round(latest_price, 2))
    print("Latest timestamp in IST:",
          latest_ts.tz_convert(IST))

    market_tz = ZoneInfo(MARKET_TIMEZONES[name])
    print("Date in reference timezone:",
          latest_ts.tz_convert(market_tz).date())

    # Diagnostic only: daily bars are date labels,
    # not necessarily exact market close instants.
    age_days = (
        NOW.date() - latest_ts.tz_convert(timezone.utc).date()
    ).days

    print("Calendar-day difference (UTC):", age_days)
    print("Note: Not a trading-session freshness verdict")

    print("\nLast 5 observations:")
    for timestamp, price in closes.tail(5).items():
        print(
            f"  {timestamp.isoformat()} | "
            f"{float(price):.2f}"
        )

    print("\nCalendar-day gaps:")
    dates = list(closes.index.date)

    for previous, current in zip(dates, dates[1:]):
        gap = (current - previous).days
        if gap > 3:
            print(
                f"  {previous} -> {current}: "
                f"{gap} calendar days"
            )


def main():
    print("=== MDV-05: CROSS-MARKET DATE VALIDATION ===")
    print("Validation time (UTC):", NOW.isoformat())
    print("Validation time (IST):",
          NOW.astimezone(IST).isoformat())

    for name, symbol in MARKETS.items():
        try:
            inspect_market(name, symbol)
        except Exception as exc:
            print(f"\nERROR inspecting {name}: {exc}")

    print("\nMDV-05 diagnostic completed.")
    print("Review observations before assigning PASS/FAIL.")


if __name__ == "__main__":
    main()
