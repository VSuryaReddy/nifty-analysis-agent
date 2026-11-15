
import json
import math
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


MARKETS = {
    "nifty_50": "^NSEI",
    "india_vix": "^INDIAVIX",
    "usd_inr": "INR=X",
    "brent_crude": "BZ=F",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
}

MAX_AGE_DAYS = 5


def fetch_history(symbol):
    """Fetch six months of daily market data."""
    data = yf.Ticker(symbol).history(
        period="6mo",
        interval="1d",
        auto_adjust=True,
    )

    if data.empty or "Close" not in data.columns:
        raise ValueError(f"No closing-price data for {symbol}")

    closes = data["Close"].copy()

    if not isinstance(closes.index, pd.DatetimeIndex):
        raise ValueError("Invalid date index")

    if closes.index.tz is None:
        raise ValueError("Missing source timezone")

    if closes.index.has_duplicates:
        raise ValueError("Duplicate observation timestamps")

    if not closes.index.is_monotonic_increasing:
        raise ValueError("Observation dates are not sorted")

    # Preserve the existing behavior for missing values,
    # but reject invalid numeric observations.
    closes = closes.dropna()

    if closes.empty:
        raise ValueError("No valid closing prices")

    if not all(
        math.isfinite(float(price)) and float(price) > 0
        for price in closes
    ):
        raise ValueError("Invalid closing price detected")

    return closes


def select_eligible_history(name, closes):
    """Exclude unconfirmed Sunday USD/INR daily bars."""
    if name == "usd_inr":
        closes = closes[
            closes.index.weekday != 6
        ]

    if len(closes) < 2:
        raise ValueError("Insufficient eligible price history")

    return closes


def validate_freshness(closes, now=None):
    """Check observation age using source-local dates."""
    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("Current time must be timezone-aware")

    source_timezone = closes.index.tz
    today = now.astimezone(source_timezone).date()
    latest_date = closes.index[-1].date()

    age_days = (today - latest_date).days

    if age_days < 0:
        raise ValueError(
            f"Future observation date: {latest_date}"
        )

    if age_days > MAX_AGE_DAYS:
        raise ValueError(
            f"Stale observation: {latest_date} "
            f"({age_days} calendar days old)"
        )

    return age_days


def calculate_rsi(closes, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return None

    for gain, loss in zip(
        gains.iloc[period + 1:],
        losses.iloc[period + 1:],
    ):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    if avg_gain == 0:
        return 0.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def fetch_market_snapshot():
    snapshot = {
        "collected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "markets": {},
        "errors": {},
    }

    for name, symbol in MARKETS.items():
        try:
            closes = fetch_history(symbol)
            closes = select_eligible_history(name, closes)
            validate_freshness(closes)

            latest = float(closes.iloc[-1])
            previous = float(closes.iloc[-2])

            market = {
                "symbol": symbol,
                "as_of": closes.index[-1].date().isoformat(),
                "close": round(latest, 2),
                "daily_change_pct": round(
                    (latest / previous - 1) * 100, 2
                ),
            }

            if name == "nifty_50":
                if len(closes) < 50:
                    raise ValueError(
                        "Insufficient history for SMA50"
                    )

                market.update({
                    "sma20": round(
                        float(closes.tail(20).mean()), 2
                    ),
                    "sma50": round(
                        float(closes.tail(50).mean()), 2
                    ),
                    "rsi14": calculate_rsi(closes),
                    "five_session_return_pct": round(
                        (
                            latest / float(closes.iloc[-6])
                            - 1
                        ) * 100,
                        2,
                    ),
                })

            snapshot["markets"][name] = market

        except Exception as exc:
            snapshot["errors"][name] = str(exc)
            print(f"Failed to fetch {name}: {exc}")

    return snapshot


if __name__ == "__main__":
    result = fetch_market_snapshot()
    print(json.dumps(result, indent=2))
