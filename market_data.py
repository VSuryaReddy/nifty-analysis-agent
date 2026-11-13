
import json
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


def fetch_history(symbol):
    """Fetch six months of daily market data."""
    data = yf.Ticker(symbol).history(
        period="6mo",
        interval="1d",
        auto_adjust=True,
    )

    if data.empty:
        raise ValueError(f"No data returned for {symbol}")

    return data["Close"].dropna()


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

            if len(closes) < 2:
                raise ValueError("Insufficient price history")

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
                        (latest / float(closes.iloc[-6]) - 1)
                        * 100,
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
