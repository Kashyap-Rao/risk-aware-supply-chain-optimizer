# Risk-Aware Supply Chain Replenishment & Routing Engine

**Live Interactive Simulator:** [Launch Streamlit App](https://kashyap-rao-risk-aware-supply-chain-optimizer-app-lhic48.streamlit.app/)

## Overview
Traditional inventory policies rely on static point forecasts and ignore supplier lead-time variability, resulting in costly stockouts or bloated safety stock. This project implements a **Predict-then-Optimize** pipeline that pairs probabilistic machine learning with Mixed-Integer Linear Programming (MILP) to dynamically allocate replenishment orders across Standard and Expedited shipping modes under warehouse and working capital constraints.

## System Architecture
1. **Cloud Data Pipeline (PostgreSQL / Neon):** Ingested and normalized 50,000+ historical order records from the DataCo Supply Chain dataset; engineered time-series demand lags, rolling volatility features, and historical carrier delay rates via SQL.
2. **Dual-Uncertainty Forecasting (LightGBM):**
   - **Demand Uncertainty:** Trained Quantile Regression models ($P_{10}, P_{50}, P_{90}$) to capture demand surges and downside risk.
   - **Lead-Time Risk:** Trained a gradient-boosted classifier predicting late-delivery probabilities across product categories.
3. **Constrained Optimization Engine (Google OR-Tools / SCIP):** Formulated a Mixed-Integer Linear Program minimizing total expected freight, holding, SLA delay penalties, and stockout costs subject to daily warehouse storage capacity and working capital limits.
4. **Interactive Decision Simulator (Streamlit & Plotly):** Real-time sensitivity analysis allowing planners to stress-test freight spikes, capacity bottlenecks, and risk-aversion buffers.

## Key Backtest Insights (30-Day Out-of-Sample Evaluation)
- **89% Reduction in Unmet Demand:** Reduced stockout units from `909.3` (static 7-day rolling baseline) to `100.0` under identical warehouse and budget constraints.
- **Dynamic Mode Switching:** Automatically routes high-margin, high-delay-risk SKUs via Expedited Freight only when expected stockout and SLA penalties outweigh the freight premium.
