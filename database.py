import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

REQUIRED_COLUMNS = {
    "order_day",
    "category_name",
    "pred_q10",
    "pred_q50",
    "pred_q90",
    "pred_delay_prob",
    "unit_price",
    "unit_profit",
    "late_rate",
    "rolling_mean_7",
    "target_demand",
}


def validate_supply_chain_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces data contracts before passing forecasts to the MILP solver."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Forecast telemetry missing required columns: {sorted(missing)}")

    clean = df.copy()
    clean["pred_delay_prob"] = clean["pred_delay_prob"].clip(0.0, 1.0)
    clean["late_rate"] = clean["late_rate"].clip(0.0, 1.0)
    return clean


def _get_db_url() -> str | None:
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
        if "postgres" in st.secrets and "url" in st.secrets["postgres"]:
            return st.secrets["postgres"]["url"]
    except Exception:
        pass
    return os.getenv("DATABASE_URL")


@st.cache_data(ttl=600)
def load_forecast_data(fallback_csv: str = "test_predictions.csv") -> tuple[pd.DataFrame, str]:
    """Loads quantile demand & delay forecasts from Neon PostgreSQL with CSV fallback."""
    db_url = _get_db_url()
    if db_url:
        try:
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as conn:
                df = pd.read_sql(text("SELECT * FROM supply_chain_predictions"), conn)
            return validate_supply_chain_schema(df), "Neon PostgreSQL (Live Cloud DB)"
        except Exception as exc:
            st.warning(f"Neon DB unavailable ({exc}); falling back to local CSV.")

    df = pd.read_csv(fallback_csv)
    return validate_supply_chain_schema(df), "Local Snapshot (test_predictions.csv)"
