# assumptions/run_acerbi_szekely.py

import os
import sys
import numpy as np

# --- Make parent folder importable ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_loader import get_prices_and_returns
from config import WEIGHTS, START_DATE, END_DATE
from backtesting import (
    rolling_var_es_forecast,
    acerbi_szekely_unconditional,
)


def main():
    # 1. Load historical returns
    prices, returns_hist = get_prices_and_returns(start=START_DATE, end=END_DATE)

    # 2. Portfolio returns and losses
    w = np.array(WEIGHTS, dtype=float)
    w = w / w.sum()
    port_ret = returns_hist.dot(w).dropna()

    conf_level = 0.99
    window = 250

    # 3. Rolling VaR/ES forecasts
    var_es_forecast = rolling_var_es_forecast(
        port_ret,
        conf_level=conf_level,
        window=window,
    )

    # Losses aligned with forecast
    losses = -port_ret.loc[var_es_forecast.index]

    # 4. Run Acerbi–Szekely ES backtest
    res = acerbi_szekely_unconditional(
        losses=losses,
        var_es_forecast=var_es_forecast,
        conf_level=conf_level,
    )

    print("=== Acerbi–Szekely ES Backtest (Unconditional) ===")
    print(f"Confidence level (ES): {conf_level:.3f}")
    print(f"T (obs): {res['T']}")
    print(f"Z_bar:   {res['Z_bar']:.6f}")
    print(f"Z_score: {res['Z_score']:.4f}")
    print(f"p-value (one-sided): {res['p_value']:.4f}")
    print("\nInterpretation:")
    print("  - H0: ES model is correct (no systematic underestimation).")
    print("  - H1: ES is too low (underestimation of tail risk).")
    print("  - Small p-value → reject H0 → your ES model is too optimistic.")


if __name__ == "__main__":
    main()
