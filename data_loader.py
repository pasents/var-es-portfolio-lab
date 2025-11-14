# data_loader.py

import numpy as np
import pandas as pd
import yfinance as yf


def get_prices_and_returns(start="2015-01-01", end=None):
    """
    Downloads BTC/EUR, Gold (USD), IWDA.AS, and EURUSD.
    Converts Gold to EUR and returns (prices_eur, returns).
    """

    BTC_TICKER = "BTC-EUR"      # Bitcoin in EUR
    IWDA_TICKER = "IWDA.AS"     # MSCI World ETF in EUR
    GOLD_TICKER = "GC=F"        # Gold futures in USD
    EURUSD_TICKER = "EURUSD=X"  # FX rate: USD per 1 EUR

    tickers = [BTC_TICKER, IWDA_TICKER, GOLD_TICKER, EURUSD_TICKER]

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # Use Adj Close if available, else Close
    if "Adj Close" in data.columns:
        prices_raw = data["Adj Close"].copy()
    else:
        prices_raw = data["Close"].copy()

    prices = prices_raw.rename(columns={
        BTC_TICKER: "BTC_EUR",
        IWDA_TICKER: "IWDA_EUR",
        GOLD_TICKER: "GOLD_USD",
        EURUSD_TICKER: "EURUSD",
    }).dropna(how="all")

    # 🔁 Convert Gold from USD to EUR:
    # GOLD_EUR = GOLD_USD / EURUSD
    prices["GOLD_EUR"] = prices["GOLD_USD"] / prices["EURUSD"]

    # Keep only BTC, GOLD (EUR), IWDA in EUR
    prices_eur = prices[["BTC_EUR", "GOLD_EUR", "IWDA_EUR"]].dropna()

    # Daily log-returns
    returns = np.log(prices_eur / prices_eur.shift(1)).dropna()

    return prices_eur, returns
if __name__ == "__main__":
    prices, returns = get_prices_and_returns()
    print(prices.tail())
    print(returns.tail())
