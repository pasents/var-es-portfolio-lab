import numpy as np
import pandas as pd
from scipy.stats import norm

def ewma_volatility(returns, lam=0.94):
    """
    Compute EWMA volatility time series (RiskMetrics model).
    
    returns : pd.Series of portfolio returns
    lam     : decay factor (lambda)
    """
    returns = returns.dropna()
    ewma_var = np.zeros(len(returns))
    
    # initialize with unconditional variance
    ewma_var[0] = returns.var()

    for t in range(1, len(returns)):
        ewma_var[t] = lam * ewma_var[t-1] + (1-lam) * returns.iloc[t-1]**2

    ewma_vol = np.sqrt(ewma_var)
    return pd.Series(ewma_vol, index=returns.index)


def ewma_var_es(returns, alpha=0.95, lam=0.94):
    """
    Compute EWMA VaR and ES for the latest return.

    returns : pd.Series of portfolio returns
    alpha   : confidence level
    lam     : decay factor
    """
    vol_series = ewma_volatility(returns, lam=lam)
    sigma_t = vol_series.iloc[-1]

    z = norm.ppf(alpha)
    var = z * sigma_t

    # ES formula for normal model
    es = sigma_t * norm.pdf(z) / (1 - alpha)

    return var, es, sigma_t
