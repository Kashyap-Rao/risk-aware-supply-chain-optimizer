import json
import os
import streamlit as st
from groq import Groq
from optimizer import sanitize_params

SYSTEM_PROMPT = """
You are an S&OP Supply Chain Optimization Copilot. Translate the planner's natural-language
disruption or policy change into updated JSON solver parameters.

Current parameters:
{current_params}

Valid keys & meanings:
- warehouse_cap (int): Max receiving warehouse capacity in units (100 to 2000)
- budget_limit (int): Daily working capital budget in USD (5000 to 500000)
- service_weight (float): Risk-aversion buffer multiplier for P90 demand & delay risk (0.0 to 2.0)
- freight_std (float): Standard sea/ground freight cost per unit in USD (0.5 to 50.0)
- freight_exp (float): Expedited air freight cost per unit in USD (1.0 to 100.0)
- sla_penalty_rate (float): Late delivery SLA penalty per unit in USD (0.0 to 100.0)

Return ONLY a valid JSON object with:
1. "updated_params": dictionary containing ALL 6 keys with updated values.
2. "reasoning": 1-sentence explanation of how the disruption was mapped to solver constraints.
"""


def _get_groq_client() -> Groq | None:
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


def parse_sop_scenario(prompt: str, current_params: dict) -> tuple[dict, str]:
    client = _get_groq_client()
    if not client:
        return current_params, "GROQ_API_KEY not found in Secrets; using existing parameters."

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(current_params=json.dumps(current_params))},
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(resp.choices[0].message.content)
        sanitized = sanitize_params(payload.get("updated_params", {}))
        return sanitized, payload.get("reasoning", "Scenario applied.")
    except Exception as exc:
        return current_params, f"Copilot fallback ({exc}); parameters unchanged."


def generate_sop_debrief(prompt: str, before: dict, after: dict, new_params: dict) -> str:
    client = _get_groq_client()
    if not client:
        return (
            f"**Updated Plan:** Total Cost `${after['ml_cost']:,.2f}` "
            f"(vs. `${before['ml_cost']:,.2f}` prior) | "
            f"Expedited Share `{after['expedited_share']:.1f}%` | "
            f"Warehouse Util `{after['warehouse_util']:.1f}%`."
        )

    summary_ctx = (
        f"Planner Request: {prompt}\n"
        f"New Constraints: {json.dumps(new_params)}\n"
        f"Before -> Total Cost: ${before['ml_cost']:,.2f}, Unmet Demand: {before['ml_unmet']:.1f} units, "
        f"Expedited Share: {before['expedited_share']:.1f}%, Capital Used: ${before['capital_used']:,.2f}\n"
        f"After -> Total Cost: ${after['ml_cost']:,.2f}, Unmet Demand: {after['ml_unmet']:.1f} units, "
        f"Expedited Share: {after['expedited_share']:.1f}%, Capital Used: ${after['capital_used']:,.2f}"
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior Supply Chain & S&OP Director. Write a crisp 3-bullet executive debrief explaining the cost, modal shift (standard vs expedited), and working-capital impact based strictly on the provided numbers.",
                },
                {"role": "user", "content": summary_ctx},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        return f"**Re-Optimized Cost:** `${after['ml_cost']:,.2f}` | **Expedited Share:** `{after['expedited_share']:.1f}%`."
