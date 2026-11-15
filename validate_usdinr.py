
import yfinance as yf

SYMBOL = "INR=X"


def inspect(interval, period):
    print(f"\n=== USD/INR: {interval} ===")

    ticker = yf.Ticker(SYMBOL)
    data = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=True,
    )

    if data.empty:
        print("No observations returned")
        return

    print("Source timezone:", data.index.tz)
    print("Observation count:", len(data))

    print("\nLatest observations:")
    print(
        data[["Open", "High", "Low", "Close", "Volume"]]
        .tail(12)
        .to_string()
    )

    print("\nObservations from November 13 onward:")
    selected = data[
        data.index.date >=
        __import__("datetime").date(2026, 11, 13)
    ]

    print(
        selected[["Open", "High", "Low", "Close"]]
        .to_string()
    )


def main():
    print("=== MDV-05: USD/INR SUNDAY CHECK ===")

    inspect(interval="1d", period="14d")
    inspect(interval="1h", period="5d")

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
