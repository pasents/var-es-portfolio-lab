import numpy as np
from scipy.optimize import minimize

from var_es import portfolio_var_es


def minimize_es_weights(returns_df,
                        conf_level=0.95,
                        horizon_days=1,
                        long_only=True):
    """
    Existing ES minimization function (keep this as you have it).
    """
    n_assets = returns_df.shape[1]

    def objective(w):
        res = portfolio_var_es(
            returns_df=returns_df,
            weights=w,
            conf_levels=[conf_level],
            horizon_days=horizon_days
        )
        es_value = res.loc[conf_level, "ES"]
        return es_value

    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    },)

    if long_only:
        bounds = [(0.0, 1.0)] * n_assets
    else:
        bounds = None

    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result


def maximize_sharpe_weights(returns_df,
                            risk_free_rate=0.0,
                            annualization_factor=252,
                            long_only=True):
    """
    Find portfolio weights that maximize the annualized Sharpe ratio.

    returns_df : DataFrame of asset log-returns
    risk_free_rate : annual risk-free rate (set 0 for simplicity)
    annualization_factor : 252 for daily data
    long_only : if True, enforce weights >= 0

    Returns: scipy.optimize result; result.x are the optimal weights.
    """

    n_assets = returns_df.shape[1]

    # Convert annual rf to daily rf (approx)
    rf_daily = risk_free_rate / annualization_factor

    def sharpe_to_maximize(w):
        w = np.array(w)
        w = w / w.sum()  # normalize, just in case
        port_ret = returns_df.dot(w).dropna()
        mu = port_ret.mean()
        sigma = port_ret.std()

        if sigma == 0:
            return 0.0  # avoid division by zero

        sharpe_daily = (mu - rf_daily) / sigma
        sharpe_annual = sharpe_daily * np.sqrt(annualization_factor)
        return sharpe_annual

    # We minimize negative Sharpe to maximize Sharpe
    def objective(w):
        return -sharpe_to_maximize(w)

    # Sum weights = 1
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    },)

    # Bounds
    if long_only:
        bounds = [(0.0, 1.0)] * n_assets
    else:
        bounds = None

    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    # Attach the Sharpe value at optimum for convenience
    if result.success:
        result.sharpe_annual = sharpe_to_maximize(result.x)
    else:
        result.sharpe_annual = None

    return result
