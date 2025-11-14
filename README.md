# VaR & Expected Shortfall Portfolio Lab

Python project to learn and demonstrate **market risk** concepts using a
3-asset portfolio: **Bitcoin (BTC/EUR), Gold (EUR), and IWDA (MSCI World ETF)**.

The project estimates **historical** and **Monte Carlo (Student-t)**:
- Value at Risk (VaR)
- Expected Shortfall (ES, a.k.a. CVaR)
- ES–Sharpe efficient frontiers
- Optimal portfolios (ES-minimizing and Sharpe-maximizing)

---

## Features

### 1. Data loader (`data_loader.py`)
- Downloads price data from Yahoo Finance:
  - `BTC-EUR` – Bitcoin in EUR  
  - `GC=F` – Gold futures in USD (converted to EUR using `EURUSD=X`)  
  - `IWDA.AS` – iShares Core MSCI World UCITS ETF (EUR)
- Converts Gold prices from USD to EUR.
- Computes **daily log returns** for all three assets.

### 2. Risk engine (`var_es.py`)
- Computes **historical VaR & ES** using the empirical distribution of portfolio returns.
- Supports:
  - arbitrary confidence levels (e.g. 95%, 99%, 99.5%)
  - arbitrary horizons via √time scaling (1-day, 10-day, 100-day, ...)

### 3. Optimization (`optimizer.py`)
- **ES minimization**: finds long-only weights that minimize 95% ES.  
- **Sharpe maximization**: finds long-only weights that maximize annualized Sharpe ratio.  
- Uses `scipy.optimize.minimize` with SLSQP under:
  - sum of weights = 1
  - weights ≥ 0 (long-only)

### 4. Monte Carlo heavy-tailed simulation (`mc_sim.py`)
- Simulates **multivariate Student-t** returns calibrated to the historical mean and covariance matrix.
- Degree of freedom `df` controls tail heaviness (default `df = 5`).
- Used to study how **tail risk explodes** under heavy-tailed assumptions.

### 5. Plotting (`plotting.py`)
- `plot_portfolio_var_es`:  
  - Histogram of portfolio returns with **VaR & ES vertical lines**.
- `plot_sharpe_vs_es_frontier`:  
  - ES–Sharpe frontier from random long-only portfolios in a single world (historical or simulated).
- `plot_hist_vs_sim_frontier`:  
  - **Historical vs Simulated** ES–Sharpe frontiers on one plot,
    highlighting:
    - ES-optimal portfolio (simulated)
    - Sharpe-optimal portfolio (simulated)

### 6. Orchestration (`main.py`)
- End-to-end pipeline:
  1. Load historical data and compute returns.
  2. Plot historical distribution + VaR/ES.
  3. Simulate Student-t returns.
  4. Plot simulated distribution + VaR/ES.
  5. Compute VaR & ES for:
     - current weights
     - ES-optimal weights
     - Sharpe-optimal weights
  6. Compare tails at **95% / 99% / 99.5%** for historical vs simulated worlds.
  7. Plot:
     - historical frontier
     - simulated frontier with optimal portfolios
     - **combined** historical vs simulated frontier.

---

## Project Structure

```text
var_es_project/
├─ data_loader.py       # Download prices, convert to EUR, compute log returns
├─ var_es.py            # Historical VaR & ES calculations
├─ optimizer.py         # ES-min and Sharpe-max optimizers
├─ mc_sim.py            # Monte Carlo Student-t simulation
├─ plotting.py          # Histograms, ES–Sharpe frontiers, comparison plots
├─ config.py            # Weights, confidence levels, horizon, date range
├─ main.py              # Orchestrates full workflow
└─ README.md            # This file
