# plotting.py

import numpy as np
import matplotlib.pyplot as plt

from var_es import portfolio_var_es


# ============================================================
# 1) Plot return distribution with VaR & ES cutoffs
# ============================================================

def plot_portfolio_var_es(returns_df,
                          weights,
                          conf_level=0.95,
                          horizon_days=1,
                          bins=60):
    """
    Plot portfolio return distribution with VaR & ES lines.

    returns_df : DataFrame of asset log-returns
    weights    : array-like, e.g. [w_BTC, w_GOLD, w_IWDA]
    conf_level : e.g. 0.95 or 0.99
    horizon_days : holding period
    bins       : number of histogram bins
    """

    # --- Portfolio returns ---
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    port_ret = returns_df.dot(w).dropna()

    # --- VaR & ES ---
    res = portfolio_var_es(
        returns_df=returns_df,
        weights=weights,
        conf_levels=[conf_level],
        horizon_days=horizon_days
    )

    var_value = res.loc[conf_level, "VaR"]
    es_value  = res.loc[conf_level, "ES"]

    # Convert to return levels
    var_cut = -var_value
    es_cut  = -es_value

    # --- Plot histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(port_ret, bins=bins, density=True, alpha=0.6)

    cl_pct = int(conf_level * 100)

    # VaR & ES lines
    plt.axvline(var_cut, linestyle="--", linewidth=2,
                label=f"{cl_pct}% VaR ({var_value:.2%})")
    plt.axvline(es_cut, linestyle=":", linewidth=2,
                label=f"{cl_pct}% ES ({es_value:.2%})")

    # Labels
    plt.title(f"Portfolio Return Distribution with {cl_pct}% VaR & ES")
    plt.xlabel("Daily log return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()



# ============================================================
# 2) Plot ES–Sharpe efficient frontier (random portfolios)
# ============================================================

def plot_sharpe_vs_es_frontier(returns_df,
                               n_portfolios=2000,
                               conf_level=0.95,
                               horizon_days=1,
                               risk_free_rate=0.0,
                               annualization_factor=252,
                               es_opt_weights=None,
                               sharpe_opt_weights=None):
    """
    ES–Sharpe frontier using random long-only portfolios.

    returns_df : DataFrame of asset returns
    n_portfolios : number of random weight vectors
    conf_level : e.g. 0.95 for ES
    horizon_days : holding period
    risk_free_rate : annual RF (0 default)
    es_opt_weights : optional (highlight ES-opt)
    sharpe_opt_weights : optional (highlight Sharpe-opt)
    """

    n_assets = returns_df.shape[1]
    rf_daily = risk_free_rate / annualization_factor

    es_list = []
    sharpe_list = []

    # --- 1) Sample random portfolios (Dirichlet enforces sum=1 and w>=0) ---
    weights_mat = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)

    for w in weights_mat:
        port_ret = returns_df.dot(w).dropna()
        mu = port_ret.mean()
        sigma = port_ret.std()

        if sigma == 0:
            continue

        # Sharpe
        sharpe_daily = (mu - rf_daily) / sigma
        sharpe_annual = sharpe_daily * np.sqrt(annualization_factor)

        # ES
        es_df = portfolio_var_es(
            returns_df=returns_df,
            weights=w,
            conf_levels=[conf_level],
            horizon_days=horizon_days
        )
        es_value = es_df.loc[conf_level, "ES"]

        sharpe_list.append(sharpe_annual)
        es_list.append(es_value)

    es_arr = np.array(es_list)
    sharpe_arr = np.array(sharpe_list)

    # --- 2) Plot the random frontier ---
    plt.figure(figsize=(10, 6))
    plt.scatter(es_arr, sharpe_arr, alpha=0.4, s=10,
                label="Random portfolios")

    cl_pct = int(conf_level * 100)

    # --- 3) ES-optimal marker ---
    if es_opt_weights is not None:
        port_ret_es = returns_df.dot(es_opt_weights).dropna()
        mu_es = port_ret_es.mean()
        sigma_es = port_ret_es.std()
        sharpe_es = ((mu_es - rf_daily) / sigma_es) * np.sqrt(annualization_factor)

        es_df_es = portfolio_var_es(
            returns_df=returns_df,
            weights=es_opt_weights,
            conf_levels=[conf_level],
            horizon_days=horizon_days
        )
        es_val_es = es_df_es.loc[conf_level, "ES"]

        plt.scatter([es_val_es], [sharpe_es],
                    marker="x", s=120, linewidths=2, color="red",
                    label="ES-optimal")

    # --- 4) Sharpe-optimal marker ---
    if sharpe_opt_weights is not None:
        port_ret_sh = returns_df.dot(sharpe_opt_weights).dropna()
        mu_sh = port_ret_sh.mean()
        sigma_sh = port_ret_sh.std()
        sharpe_sh = ((mu_sh - rf_daily) / sigma_sh) * np.sqrt(annualization_factor)

        es_df_sh = portfolio_var_es(
            returns_df=returns_df,
            weights=sharpe_opt_weights,
            conf_levels=[conf_level],
            horizon_days=horizon_days
        )
        es_val_sh = es_df_sh.loc[conf_level, "ES"]

        plt.scatter([es_val_sh], [sharpe_sh],
                    marker="D", s=120, color="green",
                    label="Sharpe-optimal")

    # --- 5) Labels & formatting ---
    plt.xlabel(f"{cl_pct}% ES (expected tail loss)")
    plt.ylabel("Annualized Sharpe ratio")
    plt.title(f"ES–Sharpe Frontier ({cl_pct}% ES)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_hist_vs_sim_frontier(returns_hist,
                              returns_sim,
                              n_portfolios=2000,
                              conf_level=0.95,
                              horizon_days=1,
                              risk_free_rate=0.0,
                              annualization_factor=252,
                              es_opt_weights=None,
                              sharpe_opt_weights=None):
    """
    Compare ES–Sharpe frontier for historical vs simulated returns
    on a single plot.

    returns_hist : DataFrame of historical returns
    returns_sim  : DataFrame of simulated returns (same columns)
    es_opt_weights, sharpe_opt_weights : typically the optimals
                                         from the SIMULATED world.
    """

    n_assets = returns_hist.shape[1]
    rf_daily = risk_free_rate / annualization_factor

    es_hist_list, sharpe_hist_list = [], []
    es_sim_list,  sharpe_sim_list  = [], []

    # 1) Sample random long-only weights
    weights_mat = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)

    for w in weights_mat:
        # ----- Historical -----
        port_hist = returns_hist.dot(w).dropna()
        mu_h = port_hist.mean()
        sigma_h = port_hist.std()
        if sigma_h > 0:
            sharpe_h_daily = (mu_h - rf_daily) / sigma_h
            sharpe_h = sharpe_h_daily * np.sqrt(annualization_factor)

            es_h_df = portfolio_var_es(
                returns_df=returns_hist,
                weights=w,
                conf_levels=[conf_level],
                horizon_days=horizon_days
            )
            es_h = es_h_df.loc[conf_level, "ES"]

            sharpe_hist_list.append(sharpe_h)
            es_hist_list.append(es_h)

        # ----- Simulated -----
        port_sim = returns_sim.dot(w).dropna()
        mu_s = port_sim.mean()
        sigma_s = port_sim.std()
        if sigma_s > 0:
            sharpe_s_daily = (mu_s - rf_daily) / sigma_s
            sharpe_s = sharpe_s_daily * np.sqrt(annualization_factor)

            es_s_df = portfolio_var_es(
                returns_df=returns_sim,
                weights=w,
                conf_levels=[conf_level],
                horizon_days=horizon_days
            )
            es_s = es_s_df.loc[conf_level, "ES"]

            sharpe_sim_list.append(sharpe_s)
            es_sim_list.append(es_s)

    es_hist_arr = np.array(es_hist_list)
    sharpe_hist_arr = np.array(sharpe_hist_list)
    es_sim_arr = np.array(es_sim_list)
    sharpe_sim_arr = np.array(sharpe_sim_list)

    cl_pct = int(conf_level * 100)

    plt.figure(figsize=(10, 6))

    # Historical frontier
    plt.scatter(es_hist_arr, sharpe_hist_arr,
                alpha=0.35, s=10, label="Historical", color="C0")

    # Simulated frontier
    plt.scatter(es_sim_arr, sharpe_sim_arr,
                alpha=0.35, s=10, label="Simulated (Student-t)", color="C1")

    # ES-optimal / Sharpe-optimal from SIMULATED world
    rf_daily = risk_free_rate / annualization_factor

    if es_opt_weights is not None:
        port_es = returns_sim.dot(es_opt_weights).dropna()
        mu_es = port_es.mean()
        sigma_es = port_es.std()
        sharpe_es = ((mu_es - rf_daily) / sigma_es) * np.sqrt(annualization_factor)

        es_es_df = portfolio_var_es(
            returns_df=returns_sim,
            weights=es_opt_weights,
            conf_levels=[conf_level],
            horizon_days=horizon_days
        )
        es_es = es_es_df.loc[conf_level, "ES"]

        plt.scatter([es_es], [sharpe_es],
                    marker="x", s=120, linewidths=2, color="red",
                    label="ES-optimal (sim)")

    if sharpe_opt_weights is not None:
        port_sh = returns_sim.dot(sharpe_opt_weights).dropna()
        mu_sh = port_sh.mean()
        sigma_sh = port_sh.std()
        sharpe_sh = ((mu_sh - rf_daily) / sigma_sh) * np.sqrt(annualization_factor)

        es_sh_df = portfolio_var_es(
            returns_df=returns_sim,
            weights=sharpe_opt_weights,
            conf_levels=[conf_level],
            horizon_days=horizon_days
        )
        es_sh = es_sh_df.loc[conf_level, "ES"]

        plt.scatter([es_sh], [sharpe_sh],
                    marker="D", s=120, color="green",
                    label="Sharpe-optimal (sim)")

    plt.xlabel(f"{cl_pct}% ES (expected tail loss)")
    plt.ylabel("Annualized Sharpe ratio")
    plt.title(f"Historical vs Simulated ES–Sharpe Frontier ({cl_pct}% ES)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
