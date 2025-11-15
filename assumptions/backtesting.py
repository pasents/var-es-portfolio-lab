# backtesting.py
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


# ============================================================
# 1) Rolling Historical VaR & ES Forecasts
# ============================================================

def rolling_var_es_forecast(
    returns,
    conf_level=0.99,
    window=250,
):
    """
    Compute rolling historical VaR & ES using a fixed-size window.

    returns   : Series of daily portfolio returns
    conf_level: VaR/ES confidence (e.g. 0.99)
    window    : rolling window length
    """
    returns = returns.dropna()
    T = len(returns)

    var_list = []
    es_list = []
    index_list = []

    for t in range(window, T):
        window_data = returns.iloc[t - window:t].values

        # Losses = -returns
        losses = -window_data
        losses_sorted = np.sort(losses)

        k = int(np.floor((1 - conf_level) * window))

        VaR = losses_sorted[-k - 1]
        ES = losses_sorted[-k:].mean()

        var_list.append(VaR)
        es_list.append(ES)
        index_list.append(returns.index[t])

    df = pd.DataFrame(
        {"VaR": var_list, "ES": es_list},
        index=index_list,
    )

    return df


# ============================================================
# 2) Kupiec POF Test — Unconditional Coverage
# ============================================================

def kupiec_pof_test(violations, conf_level=0.99):
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    violations : array of 0/1 indicators for VaR breaches
    conf_level : VaR confidence (e.g. 0.99)
    """
    violations = np.array(violations)
    T = len(violations)
    N = violations.sum()

    pi_expected = 1 - conf_level
    pi_empirical = N / T

    # Likelihood ratio
    LR = -2 * (
        (T - N) * np.log(1 - pi_expected) +
        N * np.log(pi_expected)
        - (T - N) * np.log(1 - pi_empirical)
        - N * np.log(pi_empirical)
    )

    p_value = 1 - chi2.cdf(LR, df=1)

    return {
        "T": T,
        "N": N,
        "viol_rate_expected": pi_expected,
        "viol_rate_empirical": pi_empirical,
        "LR_uc": LR,
        "p_value": p_value,
    }


# ============================================================
# 3) Christoffersen Test — Independence + Conditional Coverage
# ============================================================

def christoffersen_test(violations, conf_level=0.99):
    """
    Christoffersen (1998) independence & conditional coverage tests.

    violations: array of 0/1 VaR violation indicators
    """
    v = np.array(violations)
    T = len(v)

    # Count transitions
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))

    # Transition probabilities for 2-state Markov chain
    pi0 = n01 / (n00 + n01 + 1e-12)
    pi1 = n11 / (n10 + n11 + 1e-12)

    # Unconditional probability
    pi_hat = (n01 + n11) / (T - 1)

    # Unconditional coverage LR
    LR_uc = -2 * (
        (n00 + n01) * np.log(1 - (1 - conf_level)) +
        (n10 + n11) * np.log(1 - conf_level)
        - (n00 * np.log(1 - pi_hat) + (n01 + n11) * np.log(pi_hat))
    )

    # Independence LR
    L_ind = (
        n00 * np.log(1 - pi0 + 1e-12) +
        n01 * np.log(pi0 + 1e-12) +
        n10 * np.log(1 - pi1 + 1e-12) +
        n11 * np.log(pi1 + 1e-12)
    )

    L_uc = (
        (n00 + n10) * np.log(1 - pi_hat + 1e-12) +
        (n01 + n11) * np.log(pi_hat + 1e-12)
    )

    LR_ind = -2 * (L_uc - L_ind)

    # Joint conditional coverage
    LR_cc = LR_uc + LR_ind

    return {
        "n00": n00, "n01": n01,
        "n10": n10, "n11": n11,
        "LR_uc": LR_uc,
        "p_uc": 1 - chi2.cdf(LR_uc, 1),
        "LR_ind": LR_ind,
        "p_ind": 1 - chi2.cdf(LR_ind, 1),
        "LR_cc": LR_cc,
        "p_cc": 1 - chi2.cdf(LR_cc, 2),
    }


# ============================================================
# 4) Acerbi–Szekely ES Backtest (Unconditional)
# ============================================================

def acerbi_szekely_unconditional(losses, var_es_forecast, conf_level=0.975):
    """
    Acerbi–Szekely (2014) ES backtest (unconditional version).

    losses          : Series of actual losses (aligned with forecast)
    var_es_forecast : DataFrame with 'VaR' and 'ES' forecasts
    """
    losses = np.array(losses)
    VaR = np.array(var_es_forecast["VaR"])
    ES = np.array(var_es_forecast["ES"])

    T = len(losses)

    # Indicator of violation
    I = (losses > VaR).astype(float)

    # Z_t statistic
    Z = I * ((losses - ES) / ES)

    Z_bar = Z.mean()
    Z_std = Z.std(ddof=1)
    Z_score = np.sqrt(T) * Z_bar / (Z_std + 1e-12)

    # One-sided test: H1 = ES underestimated → Z > 0
    p_value = 1 - norm.cdf(Z_score)

    return {
        "T": T,
        "Z_bar": Z_bar,
        "Z_score": Z_score,
        "p_value": p_value,
    }
