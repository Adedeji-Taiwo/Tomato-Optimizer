# TomatoPro – Stochastic Portfolio Optimiser for Tomato Processing (Nigeria)

> An interactive decision-support tool built in Python to help tomato
> processing plant managers allocate fresh tomatoes across three product channels under
> supply and price uncertainty.

**Developed as part of the Advanced Analytics for Agribusiness course, MSc in Agribusiness & Innovation, UM6P.**

---

## Live Demo

[Launch App on Streamlit Cloud](https://tomato-optimizer.streamlit.app/)

---

## Problem Statement

Tomato processors in Nigeria face three compounding uncertainties:

1. **Yield** — seasonal rainfall drives yields between 10 and 20 t/ha on the same contracted farmland.
2. **Spoilage** — post-harvest losses range from 5% to 20% depending on season and logistics conditions.
3. **Market prices** — fresh-market, paste, and dried-tomato prices follow distinct seasonal dynamics.

Static, deterministic planning systematically underperforms under these conditions. This tool provides a mathematically principled recommendation engine based on stochastic programming.

---

## Solution Architecture

### Two-Stage Stochastic Linear Programme

- **Stage 1 — here-and-now:** Reserve processing capacity for each product *before* the season.
- **Stage 2 — wait-and-see:** Observe the realised scenario and optimise production within reserved capacity.

### Objective Function

```
max (1 − λ) · E[Profit] + λ · CVaR_α[Profit]
```

- `λ = 0` — risk-neutral: maximise expected profit.
- `λ = 1` — fully risk-averse: maximise CVaR.
- `α` — tail probability (e.g., 0.10 = worst 10% of scenarios).

CVaR is linearised following Rockafellar & Uryasev (2000), preserving LP structure.

---

## Features

| Feature | Description |
|---|---|
| Interactive UI | Sliders for λ and α |
| Three-scenario model | Good / Average / Bad rainfall, calibrated to Nigerian data |
| CVaR risk measure | Linearised formulation (Rockafellar & Uryasev, 2000) |
| Efficient frontier | Parametric sweep over λ |
| Scenario analysis | Production and profit charts by scenario |
| Revenue waterfall | Revenue vs. processing cost by product |
| Downloadable plan | Export production plan as CSV |
| Methodology tab | Full mathematical write-up with references |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Optimisation | Pyomo + GLPK |
| Data | NumPy, Pandas |
| Visualisation | Plotly |
| Web Interface | Streamlit |
| Deployment | Streamlit Cloud |

---

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/tomato-optimiser
cd tomato-optimiser
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Install GLPK: sudo apt-get install glpk-utils (Ubuntu) / brew install glpk (Mac)
streamlit run app.py
```

---

## References

Birge, J. R., & Louveaux, F. (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer.

Hart, W. E., et al. (2017). *Pyomo — Optimization Modeling in Python* (2nd ed.). Springer.

FAOSTAT (2023). *Crops and livestock products: Nigeria*. FAO.

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.

---

*Advanced Analytics for Agribusiness · MSc Agribusiness & Innovation · UM6P*