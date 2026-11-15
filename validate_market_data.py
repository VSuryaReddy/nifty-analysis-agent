
import math
import pandas as pd
import yfinance as yf

from market_data import fetch_history, calculate_rsi

SYMBOL = "^NSEI"


def main():
    closes = fetch_history(SYMBOL)

    print("=== NIFTY MARKET DATA VALIDATION ===")
    print("Observations:", len(closes))
    print("First date:", closes.index[0])
    print("Latest date:", closes.index[-1])
    print("Missing prices:", closes.isna().sum())
    print("Duplicate dates:", closes.index.duplicated().sum())
    print("Dates sorted:", closes.index.is_monotonic_increasing)

    if len(closes) < 51:
        raise ValueError("Need at least 51 observations")

    latest = float(closes.iloc[-1])
    previous = float(closes.iloc[-2])

    sma20 = sum(float(x) for x in closes.iloc[-20:]) / 20
    sma50 = sum(float(x) for x in closes.iloc[-50:]) / 50
    daily_return = (latest / previous - 1) * 100
    five_return = (latest / float(closes.iloc[-6]) - 1) * 100

    # Independent RSI implementation using price changes
    changes = [
        float(closes.iloc[i] - closes.iloc[i - 1])
        for i in range(1, len(closes))
    ]
    gains = [max(x, 0.0) for x in changes]
    losses = [max(-x, 0.0) for x in changes]

    avg_gain = sum(gains[:14]) / 14
    avg_loss = sum(losses[:14]) / 14

    for gain, loss in zip(gains[14:], losses[14:]):
        avg_gain = (avg_gain * 13 + gain) / 14
        avg_loss = (avg_loss * 13 + loss) / 14

    if avg_loss == 0:
        independent_rsi = 100.0 if avg_gain > 0 else 50.0
    elif avg_gain == 0:
        independent_rsi = 0.0
    else:
        rs = avg_gain / avg_loss
        independent_rsi = 100 - 100 / (1 + rs)

    print("\n=== CALCULATED VALUES ===")
    print(f"Latest close: {latest:.2f}")
    print(f"SMA20: {sma20:.2f}")
    print(f"SMA50: {sma50:.2f}")
    print(f"Daily return: {daily_return:.2f}%")
    print(f"Five-session return: {five_return:.2f}%")
    print(f"RSI14 independent: {independent_rsi:.2f}")
    print(f"RSI14 application: {calculate_rsi(closes):.2f}")

    assert closes.index.is_monotonic_increasing
    assert not closes.index.has_duplicates
    assert all(math.isfinite(float(x)) and x > 0 for x in closes)
    assert math.isclose(
        independent_rsi, calculate_rsi(closes), abs_tol=0.011
    )

    print("\nPASS: Internal calculations and series checks")

    print("\n=== LATEST 5 OBSERVATIONS ===")
    print(closes.tail(5).to_string())


if __name__ == "__main__":
    main()
