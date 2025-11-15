# assumptions/run_christoffersen.py

import os
import sys
import numpy as np

# --- Make parent folder importable ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_loader import get_prices_and_returns
from config import WEIGHTS, START_DATE, END_DATE
from backtesting import rolling_var_es_forecast, christoffersen_test


def main():
    # 1. Load historical returns
    prices, returns_hist = get_prices_and_returns(start=START_DATE, end=END_DATE)

    # 2. Portfolio returns for current weights
    w = np.array(WEIGHTS, dtype=float)
    w = w / w.sum()
    port_ret = returns_hist.dot(w).dropna()

    # 3. Rolling VaR/ES forecasts
    conf_level = 0.99
    window = 250

    var_es_forecast = rolling_var_es_forecast(
        port_ret,
        conf_level=conf_level,
        window=window,
    )

    # 4. Violations series
    losses = -port_ret.loc[var_es_forecast.index]
    violations = (losses > var_es_forecast["VaR"]).astype(int).values

    # 5. Run Christoffersen test
    res = christoffersen_test(violations, conf_level=conf_level)

    print("=== Christoffersen Conditional Coverage Test ===")
    print(f"Confidence level (VaR): {conf_level:.3f}")
    print(f"Transition counts:")
    print(f"  n00: {res['n00']}  (no→no)")
    print(f"  n01: {res['n01']}  (no→yes)")
    print(f"  n10: {res['n10']}  (yes→no)")
    print(f"  n11: {res['n11']}  (yes→yes)")
    print()
    print(f"Unconditional coverage:")
    print(f"  LR_uc: {res['LR_uc']:.4f}, p_uc: {res['p_uc']:.4f}")
    print()
    print(f"Independence:")
    print(f"  LR_ind: {res['LR_ind']:.4f}, p_ind: {res['p_ind']:.4f}")
    print()
    print(f"Conditional coverage (joint):")
    print(f"  LR_cc: {res['LR_cc']:.4f}, p_cc: {res['p_cc']:.4f}")
    print("\nInterpretation:")
    print("  - p_ind tests clustering in violations (independence).")
    print("  - p_cc combines unconditional coverage + independence.")
    print("  - Low p-values → the VaR model fails the conditional coverage test.")


if __name__ == "__main__":
    main()
