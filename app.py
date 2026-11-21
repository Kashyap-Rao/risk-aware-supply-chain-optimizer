import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ortools.linear_solver import pywraplp

st.set_page_config(page_title="Risk-Aware Supply Chain Optimizer", layout="wide")

st.title("Risk-Aware Supply Chain Replenishment & Routing Engine")
st.markdown(
    "**Predict-then-Optimize Architecture:** Combining **LightGBM Quantile Regression** "
    "($P_{10}, P_{50}, P_{90}$ demand & lead-time delay risk) with **Google OR-Tools MILP** "
    "to optimize order quantities and shipping modes under warehouse and budget constraints."
)

@st.cache_data
def load_data():
    df = pd.read_csv("test_predictions.csv")
    return df

df = load_data()
available_days = sorted(df["order_day"].unique())

# --- Sidebar Controls ---
st.sidebar.header("Operational Constraints & Costs")
selected_day = st.sidebar.selectbox("Simulation Date", available_days, index=min(10, len(available_days) - 1))
warehouse_cap = st.sidebar.slider("Warehouse Capacity (Units)", min_value=200, max_value=800, value=450, step=25)
budget_limit = st.sidebar.slider("Daily Working Capital Budget ($)", min_value=20000, max_value=120000, value=65000, step=5000)
service_weight = st.sidebar.slider("Risk-Aversion / Service Level Buffer", min_value=0.0, max_value=1.5, value=0.6, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Freight & SLA Parameters")
freight_std = st.sidebar.number_input("Standard Freight ($/unit)", value=3.0, step=0.5)
freight_exp = st.sidebar.number_input("Expedited Freight ($/unit)", value=7.5, step=0.5)
sla_penalty_rate = st.sidebar.number_input("Late Delivery SLA Penalty ($/unit)", value=10.0, step=1.0)

def solve_day(day_snapshot, cap, budget, weight, f_std, f_exp, sla_fee):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    n = len(day_snapshot)

    p50 = day_snapshot["pred_q50"].values
    p90 = day_snapshot["pred_q90"].values
    delay_prob = day_snapshot["pred_delay_prob"].values
    unit_cost = day_snapshot["unit_price"].values * 0.60
    stockout_pen = np.maximum(15.0, day_snapshot["unit_profit"].values * 1.5)
    holding_cost = unit_cost * 0.05

    q_std = [solver.NumVar(0, 500, f"q_std_{i}") for i in range(n)]
    q_exp = [solver.NumVar(0, 500, f"q_exp_{i}") for i in range(n)]
    shortage = [solver.NumVar(0, 500, f"short_{i}") for i in range(n)]
    overstock = [solver.NumVar(0, 500, f"over_{i}") for i in range(n)]

    target_buffer = p50 + weight * (p90 - p50) * (0.5 + 0.5 * delay_prob)

    for i in range(n):
        eff_supply = (1.0 - 0.25 * delay_prob[i]) * q_std[i] + 0.98 * q_exp[i]
        solver.Add(eff_supply + shortage[i] - overstock[i] == target_buffer[i])

    solver.Add(sum(q_std[i] + q_exp[i] for i in range(n)) <= cap)
    solver.Add(sum((unit_cost[i] + f_std) * q_std[i] + (unit_cost[i] + f_exp) * q_exp[i] for i in range(n)) <= budget)

    obj = solver.Objective()
    for i in range(n):
        # Standard shipping incurs expected SLA delay penalty proportional to delay risk
        obj.SetCoefficient(q_std[i], float(f_std + delay_prob[i] * sla_fee))
        obj.SetCoefficient(q_exp[i], float(f_exp))
        obj.SetCoefficient(overstock[i], float(holding_cost[i]))
        obj.SetCoefficient(shortage[i], float(stockout_pen[i]))
    obj.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None

    res = day_snapshot.copy()
    res["order_std"] = [round(q_std[i].solution_value(), 1) for i in range(n)]
    res["order_exp"] = [round(q_exp[i].solution_value(), 1) for i in range(n)]
    res["total_ordered"] = res["order_std"] + res["order_exp"]
    return res

# --- Solve for Selected Day ---
snap = df[df["order_day"] == selected_day].reset_index(drop=True)
plan = solve_day(snap, warehouse_cap, budget_limit, service_weight, freight_std, freight_exp, sla_penalty_rate)

if plan is None:
    st.error("No feasible solution found under current constraints. Try increasing Budget or Warehouse Capacity.")
else:
    # KPIs
    actual = plan["target_demand"].values
    unit_cost = plan["unit_price"].values * 0.60
    penalty = np.maximum(15.0, plan["unit_profit"].values * 1.5)
    hold = unit_cost * 0.05
    late_fee = plan["late_rate"].values * sla_penalty_rate

    ml_supply = plan["total_ordered"].values
    ml_unmet = np.maximum(0, actual - ml_supply)
    ml_leftover = np.maximum(0, ml_supply - actual)
    ml_cost = np.sum(plan["order_std"].values * (freight_std + late_fee) + plan["order_exp"].values * freight_exp + ml_unmet * penalty + ml_leftover * hold)

    trad_raw = plan["rolling_mean_7"].values
    scale = min(1.0, warehouse_cap / max(1e-5, np.sum(trad_raw)), budget_limit / max(1e-5, np.sum(trad_raw * (unit_cost + freight_std))))
    trad_supply = trad_raw * scale
    trad_unmet = np.maximum(0, actual - trad_supply)
    trad_leftover = np.maximum(0, trad_supply - actual)
    trad_cost = np.sum(trad_supply * (freight_std + late_fee) + trad_unmet * penalty + trad_leftover * hold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML-MILP Total Cost", f"${ml_cost:,.2f}", f"{((ml_cost - trad_cost)/trad_cost)*100:+.1f}% vs Static")
    c2.metric("Unmet Demand (Stockouts)", f"{np.sum(ml_unmet):.1f} units", f"{np.sum(ml_unmet) - np.sum(trad_unmet):+.1f} vs Static", delta_color="inverse")
    c3.metric("Expedited Share", f"{(plan['order_exp'].sum() / max(1, plan['total_ordered'].sum()))*100:.1f}%")
    c4.metric("Warehouse Utilization", f"{(plan['total_ordered'].sum() / warehouse_cap)*100:.1f}%")

    # Visual 1: Probabilistic Demand Forecast vs Optimal Order Plan
    st.subheader(f"Replenishment Allocation & Uncertainty Intervals ({selected_day})")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Standard Shipping Order", x=plan["category_name"], y=plan["order_std"], marker_color="#2b5c8f"))
    fig.add_trace(go.Bar(name="Expedited Shipping Order", x=plan["category_name"], y=plan["order_exp"], marker_color="#f28e2b"))
    fig.add_trace(go.Scatter(
        name="ML Forecast (P50 with P10–P90 Interval)",
        x=plan["category_name"],
        y=plan["pred_q50"],
        mode="markers",
        marker=dict(color="#111111", size=10, symbol="diamond"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=plan["pred_q90"] - plan["pred_q50"],
            arrayminus=plan["pred_q50"] - plan["pred_q10"],
            color="#555555",
            thickness=1.5
        )
    ))
    fig.add_trace(go.Scatter(
        name="Actual Realized Demand",
        x=plan["category_name"],
        y=plan["target_demand"],
        mode="markers",
        marker=dict(color="#d62728", size=11, symbol="x")
    ))
    fig.update_layout(barmode="stack", yaxis_title="Units", xaxis_title="Product Category", height=430)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    st.subheader("Category-Level Decision Breakdown")
    display_cols = ["category_name", "pred_q10", "pred_q50", "pred_q90", "pred_delay_prob", "order_std", "order_exp", "total_ordered", "target_demand"]
    st.dataframe(plan[display_cols].round(2), use_container_width=True)
