# **Market Risk Modelling: VaR, Expected Shortfall, EWMA & Backtesting**

This repository implements a complete market-risk workflow for a three-asset portfolio consisting of **BTC/EUR**, **Gold (EUR)**, and **IWDA (MSCI World, EUR)**. The project covers:

- Historical and Monte Carlo (Student-t) **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**  
- **EWMA (RiskMetrics)** volatility, VaR and ES  
- **Portfolio optimisation** under tail-risk (ES-minimisation) and risk-adjusted return (Sharpe maximisation)  
- **Backtesting** using industry-standard statistical tests:
  - Kupiec POF (Unconditional Coverage)  
  - Christoffersen Independence & Conditional Coverage  
  - Acerbi–Szekely ES Backtest  

All results are visualised with efficient frontiers, distribution plots, and model validation summaries.

---

## **1. Objectives**

This project was developed to demonstrate key market-risk concepts used in trading desks, risk control units, and regulatory reporting:

- Behaviour of tail-risk under different distributional assumptions  
- Sensitivity of portfolio construction to ES vs Sharpe objectives  
- Comparison of **historical** vs **heavy-tailed** return environments  
- Statistical validation of risk measures under **Basel backtesting frameworks**  

The implementation is intentionally transparent and designed for auditability, teaching, and research.

---

## **2. Methodology Overview**

### **Historical Risk**
- Empirical VaR/ES from daily log returns  
- Multiple confidence levels (95%, 99%, 99.5%)  
- Horizon scaling via √time  

### **EWMA (RiskMetrics)**
- λ = 0.94 daily decay parameter  
- Dynamic volatility estimates  
- Parametric VaR/ES under conditional normality  

### **Monte Carlo (Student-t) Simulation**
- Multivariate Student-t with ν = 5 degrees of freedom  
- Calibrated to historical mean + covariance  
- Generates heavy-tailed stress scenarios  
- Used to compute VaR/ES and efficient frontiers in a “fat-tail world”  

### **Portfolio Optimisation**
- **ES-minimisation:** robust tail-risk portfolio  
- **Sharpe maximisation:** classical mean–variance strategy  
- Long-only constraints with weights summing to 1  

### **Backtesting**
Performed on rolling 250-day VaR/ES forecasts at 99% confidence:

| Test | Purpose | Interpretation |
|------|---------|----------------|
| **Kupiec POF** | Frequency of VaR breaches | Checks unconditional accuracy |
| **Christoffersen** | Clustering of breaches | Tests independence & conditional coverage |
| **Acerbi–Szekely** | ES accuracy | Detects ES underestimation |

---

## **3. Key Results (Summary)**

### **Backtesting Outcomes (99% level)**

- **Kupiec POF:**  
  - p-value ≈ 0.26 → *unconditional coverage not rejected*  

- **Christoffersen Independence:**  
  - p-value ≈ 0.03 → *breaches show time clustering*  

- **Christoffersen Conditional Coverage:**  
  - p-value ≈ 0.04 → *model fails joint conditional coverage*  

- **Acerbi–Szekely ES Backtest:**  
  - p-value ≈ 0.18 → *no evidence of ES underestimation*  

**Interpretation:**  
The historical VaR/ES model is *accurate on average*, but *slow to react to volatility regime shifts* — consistent with real-world model risk. Heavy-tailed simulation further emphasises structural tail-risk understatement under classical assumptions.

---

## **4. Figures**

All figures are generated automatically and stored in `figures/`.

<div align="center">
  <img src="figures/frontier_compare.png" width="650">
</div>

Additional figures include:

- Historical vs Simulated return distributions  
- Historical ES–Sharpe frontier  
- Simulated ES–Sharpe frontier  
- Combined frontier comparison  

---

## **5. Project Architecture**

```
var_es_project/
├── config.py              # Parameters (weights, CLs, horizon, date range)
├── data_loader.py         # Price download, FX conversion, log returns
├── var_es.py              # Historical VaR/ES computation
├── ewma.py                # EWMA volatility, VaR & ES
├── optimizer.py           # ES-min and Sharpe-max optimizers
├── mc_sim.py              # Student-t Monte Carlo simulation
├── plotting.py            # Histograms, frontiers, comparison plots
├── main.py                # Full orchestrated pipeline
│
├── assumptions/           # Backtesting & econometric tests
│   ├── backtesting.py         # Rolling VaR/ES, Kupiec, Christoffersen, Acerbi–Szekely
│   ├── run_kupiec.py          # Runs Kupiec POF test
│   ├── run_christoffersen.py  # Runs independence & conditional coverage tests
│   └── run_acerbi_szekely.py  # Runs ES backtest
│
├── figures/               # Auto-generated plots for the README
│   ├── hist_distribution.png
│   ├── sim_distribution.png
│   ├── frontier_historical.png
│   ├── frontier_simulated.png
│   └── frontier_compare.png
│
├── requirements.txt
└── README.md
```
## **6. Running the Pipeline**

### Full analysis (historical + MC + EWMA + plots):

```bash
python main.py
```
### Individual backtest scripts:
```bash
python assumptions/run_kupiec.py
python assumptions/run_christoffersen.py
python assumptions/run_acerbi_szekely.py
```
