# var_es.py

import numpy as np
import pandas as pd


def portfolio_var_es(returns_df,
                     weights,
                     conf_levels=(0.95, 0.99),
                     horizon_days=1):
    """
    Historical VaR & ES for a portfolio.

    returns_df : DataFrame of asset log-returns (columns = assets)
    weights    : array-like, same length as number of columns
    conf_levels: iterable of confidence levels (e.g. [0.95, 0.99])
    horizon_days: holding period (scales results by sqrt(time))

    Returns: pandas DataFrame with rows = conf levels, columns = ['VaR', 'ES']
    """
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()  # normalize

    # Portfolio returns
    port_ret = returns_df.dot(weights).dropna().values
    n = len(port_ret)
    scale = np.sqrt(horizon_days)

    results = []

    for cl in conf_levels:
        alpha = 1 - cl  # tail prob (e.g. 0.05 for 95% VaR)

        # Sort returns from worst to best
        sorted_ret = np.sort(port_ret)

        k = int(np.floor(alpha * n))
        k = max(1, k)

        # VaR: quantile of loss, as positive number
        var_ret = sorted_ret[k - 1]
        var_value = -var_ret * scale

        # ES: mean of worst alpha% returns, also positive
        tail = sorted_ret[:k]
        es_ret = tail.mean()
        es_value = -es_ret * scale

        results.append({"conf_level": cl, "VaR": var_value, "ES": es_value})

    return pd.DataFrame(results).set_index("conf_level")
