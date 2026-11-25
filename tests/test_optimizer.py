import pandas as pd
import pytest
from optimizer import DEFAULT_PARAMS, PARAM_BOUNDS, sanitize_params, solve_replenishment_milp
from database import validate_supply_chain_schema


def make_synthetic_snapshot(num_categories: int = 5) -> pd.DataFrame:
    """Creates a compact synthetic daily forecast snapshot for fast unit testing."""
    rows = []
    for i in range(1, num_categories + 1):
        rows.append({
            "order_day": "2026-10-01",
            "category_name": f"Category-{i}",
            "pred_q10": 15.0 * i,
            "pred_q50": 25.0 * i,
            "pred_q90": 40.0 * i,
            "pred_delay_prob": min(0.95, 0.15 * i),
            "unit_price": 50.0 + 10.0 * i,
            "unit_profit": 15.0 + 3.0 * i,
            "late_rate": min(0.90, 0.10 * i),
            "rolling_mean_7": 22.0 * i,
            "target_demand": 26.0 * i,
        })
    return pd.DataFrame(rows)


def test_parameter_guardrails_clamp_extremes():
    """Verifies out-of-bounds LLM or slider inputs get clamped safely into valid ranges."""
    bad_inputs = {
        "warehouse_cap": -50,
        "budget_limit": 99999999,
        "service_weight": -2.5,
        "freight_std": 0.0,
    }
    clean = sanitize_params(bad_inputs)
    assert clean["warehouse_cap"] == PARAM_BOUNDS["warehouse_cap"][0]
    assert clean["budget_limit"] == PARAM_BOUNDS["budget_limit"][1]
    assert clean["service_weight"] == PARAM_BOUNDS["service_weight"][0]
    assert clean["freight_std"] == PARAM_BOUNDS["freight_std"][0]


def test_milp_respects_warehouse_and_budget_constraints():
    """Verifies total ordered units and allocated capital never exceed operational limits."""
    snap = make_synthetic_snapshot(num_categories=6)
    params = {**DEFAULT_PARAMS, "warehouse_cap": 200, "budget_limit": 15000}

    result = solve_replenishment_milp(snap, params)
    assert result is not None

    plan = result["plan"]
    assert (plan["order_std"] >= -1e-6).all(), "Negative standard order quantity!"
    assert (plan["order_exp"] >= -1e-6).all(), "Negative expedited order quantity!"
    assert plan["total_ordered"].sum() <= 200.0 + 1e-4, "Warehouse capacity exceeded!"
    assert result["capital_used"] <= 15000.0 + 1e-2, "Daily working capital budget exceeded!"


def test_schema_validation_catches_missing_columns():
    """Verifies database schema validation raises ValueError when required columns are missing."""
    broken_df = pd.DataFrame({"order_day": ["2026-10-01"], "category_name": ["Electronics"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_supply_chain_schema(broken_df)
