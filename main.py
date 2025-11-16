# main.py

import numpy as np
import os

from data_loader import get_prices_and_returns
from var_es import portfolio_var_es
from plotting import (
    plot_portfolio_var_es,
    plot_sharpe_vs_es_frontier,
    plot_hist_vs_sim_frontier,
)
from optimizer import minimize_es_weights, maximize_sharpe_weights
from config import WEIGHTS, CONF_LEVELS, HORIZON_DAYS, START_DATE, END_DATE
from mc_sim import simulate_student_t_returns
from ewma import ewma_var_es   # <-- NEW IMPORT


def main():
    # Optional: reproducible simulation
    #np.random.seed(42)

    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)

    # 1. Historical prices and returns
    prices, returns_hist = get_prices_and_returns(start=START_DATE, end=END_DATE)

    print("Assets:", list(returns_hist.columns))
    print("Sample of HISTORICAL returns:")
    print(returns_hist.tail(), "\n")

    # --- Visual 1: historical return distribution + VaR/ES ---
    plot_portfolio_var_es(
        returns_df=returns_hist,
        weights=WEIGHTS,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
        save_path="figures/hist_distribution.png"
    )

    # 2. Monte Carlo Student-t simulation
    returns = simulate_student_t_returns(
        historical_returns=returns_hist,
        n_days=returns_hist.shape[0] * 5,
        df=5
    )

    print("Sample of SIMULATED Student-t returns:")
    print(returns.tail(), "\n")

    # --- Visual 2: simulated return distribution + VaR/ES ---
    plot_portfolio_var_es(
        returns_df=returns,
        weights=WEIGHTS,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
        save_path="figures/sim_distribution.png"
    )

    # 3. Risk for current weights under simulated data (historical VaR/ES style)
    base_risk = portfolio_var_es(
        returns_df=returns,
        weights=WEIGHTS,
        conf_levels=CONF_LEVELS,
        horizon_days=HORIZON_DAYS
    )

    print("Portfolio VaR & ES with current weights (simulated, historical method)")
    print(base_risk.to_string(float_format=lambda x: f"{x:.4%}"))
    print()

    # --- 3b. EWMA risk metrics for current weights (historical + simulated) ---
    w = np.array(WEIGHTS, dtype=float)
    w = w / w.sum()

    port_hist = returns_hist.dot(w).dropna()
    port_sim = returns.dot(w).dropna()

    # EWMA at 95% confidence, lambda 0.94
    ewma_var_hist, ewma_es_hist, sigma_hist = ewma_var_es(
        port_hist, alpha=0.95, lam=0.94
    )
    ewma_var_sim, ewma_es_sim, sigma_sim = ewma_var_es(
        port_sim, alpha=0.95, lam=0.94
    )

    print("EWMA (RiskMetrics-style) tail risk for CURRENT weights (95%):")
    print(f"Historical EWMA volatility: {sigma_hist:.4%}")
    print(f"Historical EWMA VaR(95%):  {ewma_var_hist:.4%}")
    print(f"Historical EWMA ES(95%):   {ewma_es_hist:.4%}\n")

    print(f"Simulated EWMA volatility: {sigma_sim:.4%}")
    print(f"Simulated EWMA VaR(95%):  {ewma_var_sim:.4%}")
    print(f"Simulated EWMA ES(95%):   {ewma_es_sim:.4%}\n")

    # 4. ES-minimizing weights
    es_opt_res = minimize_es_weights(
        returns_df=returns,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
        long_only=True
    )
    es_opt_weights = es_opt_res.x

    print("Optimal long-only weights to minimize 95% ES (simulated):")
    for name, w_i in zip(returns.columns, es_opt_weights):
        print(f"  {name}: {w_i:.2%}")
    print()

    # 5. Sharpe-maximizing weights
    sharpe_opt_res = maximize_sharpe_weights(
        returns_df=returns,
        risk_free_rate=0.0,
        annualization_factor=252,
        long_only=True
    )
    sharpe_opt_weights = sharpe_opt_res.x

    print("Sharpe-maximizing long-only weights (simulated):")
    for name, w_i in zip(returns.columns, sharpe_opt_weights):
        print(f"  {name}: {w_i:.2%}")
    print(f"Annualized Sharpe: {sharpe_opt_res.sharpe_annual:.3f}\n")

    # --- Visual 3: historical ES–Sharpe frontier ---
    plot_sharpe_vs_es_frontier(
        returns_df=returns_hist,
        n_portfolios=2000,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
        risk_free_rate=0.0,
        annualization_factor=252,
        save_path="figures/frontier_historical.png"
    )

    # --- Visual 4: simulated ES–Sharpe frontier ---
    plot_sharpe_vs_es_frontier(
        returns_df=returns,
        n_portfolios=2000,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
        risk_free_rate=0.0,
        annualization_factor=252,
        es_opt_weights=es_opt_weights,
        sharpe_opt_weights=sharpe_opt_weights,
        save_path="figures/frontier_simulated.png"
    )

    # --- Visual 5: combined frontier ---
    plot_hist_vs_sim_frontier(
        returns_hist=returns_hist,
        returns_sim=returns,
        n_portfolios=2000,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
        risk_free_rate=0.0,
        annualization_factor=252,
        es_opt_weights=es_opt_weights,
        sharpe_opt_weights=sharpe_opt_weights,
        save_path="figures/frontier_compare.png"
    )


if __name__ == "__main__":
    main()
