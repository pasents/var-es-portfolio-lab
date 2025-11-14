# main.py

import numpy as np

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


def main():
    # Optional: make results reproducible
    np.random.seed(42)

    # 1. Get historical prices and returns
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
    )

    # 2. Simulate heavy-tailed Student-t returns based on historical stats
    returns = simulate_student_t_returns(
        historical_returns=returns_hist,
        n_days=returns_hist.shape[0] * 5,  # e.g. 5x the historical length
        df=5                                # lower df ⇒ fatter tails
    )

    print("Sample of SIMULATED Student-t returns:")
    print(returns.tail(), "\n")

    # --- Visual 2: Monte Carlo return distribution + VaR/ES ---
    plot_portfolio_var_es(
        returns_df=returns,
        weights=WEIGHTS,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
    )

    # 3. VaR & ES for your current weights (on simulated data)
    base_risk = portfolio_var_es(
        returns_df=returns,
        weights=WEIGHTS,
        conf_levels=CONF_LEVELS,
        horizon_days=HORIZON_DAYS
    )

    print("Portfolio VaR & ES with current weights (on simulated data)")
    print(f"Current weights (BTC, GOLD, IWDA): {WEIGHTS}")
    print(base_risk.to_string(float_format=lambda x: f"{x:.4%}"))
    print()

    # 4. ES-minimizing weights (95% ES) on simulated data
    es_opt_res = minimize_es_weights(
        returns_df=returns,
        conf_level=0.95,
        horizon_days=HORIZON_DAYS,
        long_only=True
    )
    es_opt_weights = es_opt_res.x

    print("Optimal long-only weights to minimize 95% ES (simulated):")
    for name, w in zip(returns.columns, es_opt_weights):
        print(f"  {name}: {w:.2%}")
    print(f"ES optimization success: {es_opt_res.success}, message: {es_opt_res.message}\n")

    es_opt_risk = portfolio_var_es(
        returns_df=returns,
        weights=es_opt_weights,
        conf_levels=CONF_LEVELS,
        horizon_days=HORIZON_DAYS
    )

    print("VaR & ES for ES-optimal weights (simulated):")
    print(es_opt_risk.to_string(float_format=lambda x: f"{x:.4%}"))
    print()

    # 5. Sharpe-maximizing weights on simulated data
    sharpe_opt_res = maximize_sharpe_weights(
        returns_df=returns,
        risk_free_rate=0.0,
        annualization_factor=252,
        long_only=True
    )
    sharpe_opt_weights = sharpe_opt_res.x

    print("Sharpe-maximizing long-only weights (simulated):")
    for name, w in zip(returns.columns, sharpe_opt_weights):
        print(f"  {name}: {w:.2%}")
    print(f"Sharpe optimization success: {sharpe_opt_res.success}, message: {sharpe_opt_res.message}")
    print(f"Annualized Sharpe at optimum: {sharpe_opt_res.sharpe_annual:.3f}\n")

    sharpe_opt_risk = portfolio_var_es(
        returns_df=returns,
        weights=sharpe_opt_weights,
        conf_levels=CONF_LEVELS,
        horizon_days=HORIZON_DAYS
    )

    print("VaR & ES for Sharpe-optimal weights (simulated):")
    print(sharpe_opt_risk.to_string(float_format=lambda x: f"{x:.4%}"))
    print()

    # 6. Tail comparison for current weights: 95 / 99 / 99.5, hist vs simulated
    tail_levels = [0.95, 0.99, 0.995]

    hist_tail = portfolio_var_es(
        returns_df=returns_hist,
        weights=WEIGHTS,
        conf_levels=tail_levels,
        horizon_days=HORIZON_DAYS
    )
    sim_tail = portfolio_var_es(
        returns_df=returns,
        weights=WEIGHTS,
        conf_levels=tail_levels,
        horizon_days=HORIZON_DAYS
    )

    print("Tail comparison for CURRENT weights (historical vs simulated)")
    print("Historical VaR & ES:")
    print(hist_tail.to_string(float_format=lambda x: f"{x:.4%}"))
    print("\nSimulated VaR & ES:")
    print(sim_tail.to_string(float_format=lambda x: f"{x:.4%}"))
    print()

    # 7. Historical vs Simulated frontier on the SAME plot
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
    )


if __name__ == "__main__":
    main()
