# assumptions/run_kupiec.py

import os
import sys
import numpy as np

# --- Make parent folder importable (so we can import data_loader, config, etc.) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_loader import get_prices_and_returns
from config import WEIGHTS, START_DATE, END_DATE
from backtesting import rolling_var_es_forecast, kupiec_pof_test


def main():
    # 1. Load historical returns
    prices, returns_hist = get_prices_and_returns(start=START_DATE, end=END_DATE)

    # 2. Build portfolio returns for current weights
    w = np.array(WEIGHTS, dtype=float)
    w = w / w.sum()
    port_ret = returns_hist.dot(w).dropna()   # daily portfolio returns

    # 3. Rolling historical VaR/ES forecasts
    conf_level = 0.99      # 99% VaR
    window = 250           # 1-year rolling window

    var_es_forecast = rolling_var_es_forecast(
        port_ret,
        conf_level=conf_level,
        window=window,
    )

    # 4. Compute violation series (loss > VaR)
    losses = -port_ret.loc[var_es_forecast.index]
    violations = (losses > var_es_forecast["VaR"]).astype(int).values

    # 5. Run Kupiec POF test
    res = kupiec_pof_test(violations, conf_level=conf_level)

    print("=== Kupiec POF Test (Unconditional Coverage) ===")
    print(f"Confidence level (VaR): {conf_level:.3f}")
    print(f"Expected violation rate: {res['viol_rate_expected']:.4%}")
    print(f"Empirical violation rate: {res['viol_rate_empirical']:.4%}")
    print(f"T (obs): {res['T']}")
    print(f"N (violations): {res['N']}")
    print(f"LR_uc: {res['LR_uc']:.4f}")
    print(f"p-value: {res['p_value']:.4f}")
    print("\nInterpretation:")
    print("  - Low p-value → reject H0 (model has wrong unconditional coverage).")
    print("  - High p-value → cannot reject H0 (coverage is consistent with model).")


if __name__ == "__main__":
    main()
