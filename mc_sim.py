# mc_sim.py

import numpy as np
import pandas as pd


def simulate_student_t_returns(historical_returns,
                               n_days=252 * 5,
                               df=12):
    """
    Simulate multivariate Student-t returns with heavy tails,
    calibrated to historical mean & covariance.

    Parameters
    ----------
    historical_returns : DataFrame
        Historical log-returns for the assets (columns = assets).
    n_days : int
        Number of simulated days.
    df : int or float
        Degrees of freedom for the Student-t distribution.
        Lower df -> heavier tails (e.g. 3–7 is typical).

    Returns
    -------
    DataFrame
        Simulated log-returns with same columns as historical_returns.
    """

    cols = historical_returns.columns
    n_assets = len(cols)

    # Estimate mean and covariance from historical data
    mu = historical_returns.mean().values        # shape (n_assets,)
    cov = historical_returns.cov().values       # shape (n_assets, n_assets)

    # Cholesky factor for covariance (for correlated normals)
    L = np.linalg.cholesky(cov)

    # 1) Draw standard normals: shape (n_days, n_assets)
    z = np.random.normal(size=(n_days, n_assets))

    # 2) Correlated normals: each row ~ N(0, cov)
    z_corr = z @ L.T

    # 3) Draw chi-square variables for t-scaling
    #    g ~ chi2(df) / df  -> 1/sqrt(g) gives heavy tails
    g = np.random.chisquare(df, size=n_days) / df
    scale = 1.0 / np.sqrt(g)          # shape (n_days,)

    # 4) Build multivariate Student-t:
    #    r_t = mu + z_corr_t * scale_t
    r = mu + z_corr * scale[:, None]

    # Wrap in DataFrame
    sim_returns = pd.DataFrame(r, columns=cols)

    return sim_returns
