import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DEFAULT_PARAMS = {
    "warehouse_cap": 450,
    "budget_limit": 65000,
    "service_weight": 0.6,
    "freight_std": 3.0,
    "freight_exp": 7.5,
    "sla_penalty_rate": 10.0,
}

PARAM_BOUNDS = {
    "warehouse_cap": (100, 2000, int),
    "budget_limit": (5000, 500000, int),
    "service_weight": (0.0, 2.0, float),
    "freight_std": (0.5, 50.0, float),
    "freight_exp": (1.0, 100.0, float),
    "sla_penalty_rate": (0.0, 100.0, float),
}


def sanitize_params(raw_params: dict) -> dict:
    """Clamps user or LLM-parsed operational parameters within safe physical bounds."""
    clean = DEFAULT_PARAMS.copy()
    for k, (low, high, cast_type) in PARAM_BOUNDS.items():
        if k in raw_params and raw_params[k] is not None:
            try:
                val = cast_type(raw_params[k])
                clean[k] = cast_type(max(low, min(high, val)))
            except (ValueError, TypeError):
                pass
    return clean


def solve_replenishment_milp(day_snapshot: pd.DataFrame, params: dict) -> dict | None:
    """Solves daily risk-aware replenishment & shipping mode allocation via Google OR-Tools."""
    p = sanitize_params(params)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None

    n = len(day_snapshot)
    p50 = day_snapshot["pred_q50"].values
    p90 = day_snapshot["pred_q90"].values
    delay_prob = day_snapshot["pred_delay_prob"].values
    unit_cost = day_snapshot["unit_price"].values * 0.60
    stockout_pen = np.maximum(15.0, day_snapshot["unit_profit"].values * 1.5)
    holding_cost = unit_cost * 0.05

    q_std = [solver.NumVar(0, 1000, f"q_std_{i}") for i in range(n)]
    q_exp = [solver.NumVar(0, 1000, f"q_exp_{i}") for i in range(n)]
    shortage = [solver.NumVar(0, 1000, f"short_{i}") for i in range(n)]
    overstock = [solver.NumVar(0, 1000, f"over_{i}") for i in range(n)]

    target_buffer = p50 + p["service_weight"] * (p90 - p50) * (0.5 + 0.5 * delay_prob)

    for i in range(n):
        eff_supply = (1.0 - 0.25 * delay_prob[i]) * q_std[i] + 0.98 * q_exp[i]
        solver.Add(eff_supply + shortage[i] - overstock[i] == float(target_buffer[i]))

    solver.Add(sum(q_std[i] + q_exp[i] for i in range(n)) <= p["warehouse_cap"])
    solver.Add(
        sum(
            (unit_cost[i] + p["freight_std"]) * q_std[i]
            + (unit_cost[i] + p["freight_exp"]) * q_exp[i]
            for i in range(n)
        )
        <= p["budget_limit"]
    )

    obj = solver.Objective()
    for i in range(n):
        obj.SetCoefficient(q_std[i], float(p["freight_std"] + delay_prob[i] * p["sla_penalty_rate"]))
        obj.SetCoefficient(q_exp[i], float(p["freight_exp"]))
        obj.SetCoefficient(overstock[i], float(holding_cost[i]))
        obj.SetCoefficient(shortage[i], float(stockout_pen[i]))
    obj.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None

    plan = day_snapshot.copy()
    plan["order_std"] = [round(q_std[i].solution_value(), 1) for i in range(n)]
    plan["order_exp"] = [round(q_exp[i].solution_value(), 1) for i in range(n)]
    plan["total_ordered"] = (plan["order_std"] + plan["order_exp"]).round(1)
    plan["allocated_capital"] = (
        plan["order_std"] * (unit_cost + p["freight_std"])
        + plan["order_exp"] * (unit_cost + p["freight_exp"])
    ).round(2)

    # Realized KPI evaluation against Static 7-Day Rolling Mean baseline
    actual = plan["target_demand"].values
    late_fee = plan["late_rate"].values * p["sla_penalty_rate"]

    ml_supply = plan["total_ordered"].values
    ml_unmet = np.maximum(0.0, actual - ml_supply)
    ml_leftover = np.maximum(0.0, ml_supply - actual)
    ml_cost = float(
        np.sum(
            plan["order_std"].values * (p["freight_std"] + late_fee)
            + plan["order_exp"].values * p["freight_exp"]
            + ml_unmet * stockout_pen
            + ml_leftover * holding_cost
        )
    )

    trad_raw = plan["rolling_mean_7"].values
    scale = min(
        1.0,
        p["warehouse_cap"] / max(1e-5, float(np.sum(trad_raw))),
        p["budget_limit"] / max(1e-5, float(np.sum(trad_raw * (unit_cost + p["freight_std"])))),
    )
    trad_supply = trad_raw * scale
    trad_unmet = np.maximum(0.0, actual - trad_supply)
    trad_leftover = np.maximum(0.0, trad_supply - actual)
    trad_cost = float(
        np.sum(
            trad_supply * (p["freight_std"] + late_fee)
            + trad_unmet * stockout_pen
            + trad_leftover * holding_cost
        )
    )

    return {
        "plan": plan,
        "ml_cost": ml_cost,
        "trad_cost": trad_cost,
        "ml_unmet": float(np.sum(ml_unmet)),
        "trad_unmet": float(np.sum(trad_unmet)),
        "expedited_share": float((plan["order_exp"].sum() / max(1.0, plan["total_ordered"].sum())) * 100.0),
        "warehouse_util": float((plan["total_ordered"].sum() / p["warehouse_cap"]) * 100.0),
        "capital_used": float(plan["allocated_capital"].sum()),
    }
