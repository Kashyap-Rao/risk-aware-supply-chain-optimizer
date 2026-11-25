import streamlit as st
import plotly.graph_objects as go
from database import load_forecast_data
from optimizer import DEFAULT_PARAMS, solve_replenishment_milp
from llm_copilot import parse_sop_scenario, generate_sop_debrief

st.set_page_config(page_title="Risk-Aware Supply Chain Optimizer", layout="wide")

# Initialize shared session state for manual sliders + LLM Copilot
for k, v in DEFAULT_PARAMS.items():
    if k not in st.session_state:
        st.session_state[k] = v

df, data_source = load_forecast_data()
available_days = sorted(df["order_day"].unique())

st.title("Risk-Aware Supply Chain Replenishment & Routing Engine")
st.markdown(
    "**Predict-then-Optimize Architecture:** Combining **LightGBM Quantile Regression** "
    "($P_{10}, P_{50}, P_{90}$ demand & lead-time delay risk) with **Google OR-Tools MILP** "
    "to optimize order quantities and shipping modes under warehouse and working-capital constraints."
)
st.caption(f"Active Telemetry Source: **{data_source}** | Rows Loaded: **{len(df):,}**")

# --- Sidebar Controls ---
st.sidebar.header("Operational Constraints & Costs")
selected_day = st.sidebar.selectbox("Simulation Date", available_days, index=min(10, len(available_days) - 1))
st.session_state["warehouse_cap"] = st.sidebar.slider("Warehouse Capacity (Units)", 100, 2000, int(st.session_state["warehouse_cap"]), 25)
st.session_state["budget_limit"] = st.sidebar.slider("Daily Working Capital Budget ($)", 5000, 500000, int(st.session_state["budget_limit"]), 5000)
st.session_state["service_weight"] = st.sidebar.slider("Risk-Aversion / Service Level Buffer", 0.0, 2.0, float(st.session_state["service_weight"]), 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Freight & SLA Parameters")
st.session_state["freight_std"] = st.sidebar.number_input("Standard Freight ($/unit)", value=float(st.session_state["freight_std"]), step=0.5)
st.session_state["freight_exp"] = st.sidebar.number_input("Expedited Freight ($/unit)", value=float(st.session_state["freight_exp"]), step=0.5)
st.session_state["sla_penalty_rate"] = st.sidebar.number_input("Late Delivery SLA Penalty ($/unit)", value=float(st.session_state["sla_penalty_rate"]), step=1.0)

current_params = {k: st.session_state[k] for k in DEFAULT_PARAMS}
snap = df[df["order_day"] == selected_day].reset_index(drop=True)
res = solve_replenishment_milp(snap, current_params)

# --- S&OP Disruption Copilot ---
with st.expander("S&OP Disruption Copilot (Natural Language What-If Simulator)", expanded=True):
    col_in, col_btn = st.columns([5, 1])
    user_prompt = col_in.text_input(
        "Describe a supply shock, freight surge, or S&OP policy shift:",
        placeholder="e.g., Port congestion doubled standard freight to $6/unit and finance cut daily budget to $45,000",
        label_visibility="collapsed",
    )
    if col_btn.button("Simulate Shock", use_container_width=True) and user_prompt.strip():
        new_p, reasoning = parse_sop_scenario(user_prompt, current_params)
        after_res = solve_replenishment_milp(snap, new_p)
        if after_res is not None:
            for k, v in new_p.items():
                st.session_state[k] = v
            st.session_state["last_reasoning"] = reasoning
            if res is not None:
                st.session_state["last_debrief"] = generate_sop_debrief(user_prompt, res, after_res, new_p)
            st.rerun()
        else:
            st.error(f"Scenario infeasible under updated constraints ({reasoning}). Try relaxing capacity or budget.")

    if "last_reasoning" in st.session_state:
        st.info(f"**Constraint Mapping:** {st.session_state['last_reasoning']}")
    if "last_debrief" in st.session_state:
        st.markdown(st.session_state["last_debrief"])

if res is None:
    st.error("No feasible solution found under current constraints. Try increasing Budget or Warehouse Capacity.")
else:
    plan = res["plan"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ML-MILP Total Cost", f"${res['ml_cost']:,.2f}", f"{((res['ml_cost'] - res['trad_cost'])/max(1e-5, res['trad_cost']))*100:+.1f}% vs Static")
    c2.metric("Unmet Demand", f"{res['ml_unmet']:.1f} units", f"{res['ml_unmet'] - res['trad_unmet']:+.1f} vs Static", delta_color="inverse")
    c3.metric("Expedited Share", f"{res['expedited_share']:.1f}%")
    c4.metric("Warehouse Util.", f"{res['warehouse_util']:.1f}%")
    c5.metric("Working Capital Used", f"${res['capital_used']:,.0f}")

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
            thickness=1.5,
        ),
    ))
    fig.add_trace(go.Scatter(
        name="Actual Realized Demand",
        x=plan["category_name"],
        y=plan["target_demand"],
        mode="markers",
        marker=dict(color="#d62728", size=11, symbol="x"),
    ))
    fig.update_layout(barmode="stack", yaxis_title="Units", xaxis_title="Product Category", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category-Level Purchase Order & Routing Breakdown")
    display_cols = [
        "category_name", "pred_q10", "pred_q50", "pred_q90", "pred_delay_prob",
        "order_std", "order_exp", "total_ordered", "allocated_capital", "target_demand",
    ]
    st.dataframe(plan[display_cols].round(2), use_container_width=True)

    csv_export = plan[["order_day"] + display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Daily ERP Purchase Orders (CSV)",
        data=csv_export,
        file_name=f"erp_replenishment_orders_{selected_day}.csv",
        mime="text/csv",
    )
