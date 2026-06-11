from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import math
import os
import pandas as pd

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


class TemporalSchema(BaseModel):
    temporal_risk: float = Field(
        description="Risk score between 0 and 1",
        ge=0,
        le=1
    )
    temporal_label: str = Field(
        description="Low | Medium | High"
    )
    temporal_reason: str = Field(
        description="Short explanation"
    )


structured_model = model.with_structured_output(TemporalSchema)


def circular_hour_distance(h1: float, h2: float) -> float:
    """d = min(|h1 - h2|, 24 - |h1 - h2|)"""
    diff = abs(h1 - h2)
    return min(diff, 24.0 - diff)


def exponential_decay_risk(deviation_hours: float, decay_lambda: float = 0.3) -> float:
    """risk = 1 - e^(-λ · d)"""
    return 1.0 - math.exp(-decay_lambda * deviation_hours)


def hour_frequency_score(txn_hour: int, hour_counts: pd.Series) -> float:
    """freq = count_at_hour / total_count"""
    total = hour_counts.sum()
    if total == 0:
        return 0.0
    return hour_counts.get(txn_hour, 0) / total


def temporal_agent(state: dict) -> dict:

    state.setdefault("nodes", [])
    txn = state.get("txn") or state.get("transaction") or {}
    history_source = (
        state.get("customer_txns")
        if state.get("customer_txns") is not None
        else state.get("transaction_history", [])
    )

    txn_timestamp = txn.get("timestamp")

    if isinstance(history_source, pd.DataFrame):
        history_df = history_source.copy()
    else:
        history_df = pd.DataFrame(history_source)

    # Prefer the explicit localHour field from the Microsoft dataset
    local_hour_raw = txn.get("localHour")
    txn_dt = pd.to_datetime(txn_timestamp, errors="coerce")

    if local_hour_raw is not None and str(local_hour_raw).strip() != "":
        try:
            txn_hour = int(float(local_hour_raw))
        except (ValueError, TypeError):
            txn_hour = txn_dt.hour if txn_dt is not None and not pd.isna(txn_dt) else None
    else:
        txn_hour = txn_dt.hour if txn_dt is not None and not pd.isna(txn_dt) else None

    # Determine which hour column to use for history
    hist_hour_col = None
    if not history_df.empty:
        if "localHour" in history_df.columns and history_df["localHour"].notna().any():
            history_df["_hour"] = pd.to_numeric(history_df["localHour"], errors="coerce")
            hist_hour_col = "_hour"
        elif "timestamp" in history_df.columns:
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
            history_df["_hour"] = history_df["timestamp"].dt.hour
            hist_hour_col = "_hour"

    if hist_hour_col and not history_df.empty and history_df[hist_hour_col].notna().any():
        hours = history_df[hist_hour_col].dropna()
        typical_hours = hours.mode().tolist()
        avg_hour = hours.mean()
        hour_counts = hours.astype(int).value_counts()

        if txn_hour is not None:
            deviation = circular_hour_distance(txn_hour, avg_hour)
            decay_risk = exponential_decay_risk(deviation)
            freq = hour_frequency_score(txn_hour, hour_counts)
            freq_risk = 1.0 - freq

            composite_temporal = round(0.6 * decay_risk + 0.4 * freq_risk, 4)

            stats_summary = (
                f"Transaction Hour: {txn_hour}:00 (local), Avg Customer Hour: {avg_hour:.1f}\n"
                f"        Circular Deviation: {deviation:.1f}h\n"
                f"        Exponential Decay Risk: 1 - e^(-0.3 × {deviation:.1f}) = {decay_risk:.2f}\n"
                f"        Hour Frequency: {freq:.0%} of historical txns at this hour\n"
                f"        Composite: 0.6×{decay_risk:.2f} + 0.4×{freq_risk:.2f} = {composite_temporal:.2f}"
            )
        else:
            composite_temporal = 0.3
            stats_summary = "No transaction hour — default moderate risk."

        history_summary = f"""
        Typical Active Hours: {typical_hours}
        Average Active Hour: {round(avg_hour, 2)}
        Total Historical Transactions: {len(history_df)}
        """
    else:
        history_summary = "No historical timestamp data available."
        stats_summary = "No history to compute temporal deviation."
        composite_temporal = 0.3

    messages = [
        SystemMessage(
            content="""
You are a senior fraud risk analyst specializing in temporal fraud detection.

Evaluate:
- Is the transaction occurring at an unusual hour?
- Is timing inconsistent with historical behavior?
- Use the pre-computed temporal analysis as strong signals.

Respond strictly in JSON format:
{
    "temporal_risk": float between 0 and 1,
    "temporal_label": "Low" | "Medium" | "High",
    "temporal_reason": "short explanation"
}
"""
        ),
        HumanMessage(
            content=f"""
Current Transaction Timestamp: {txn_timestamp}
Local Hour: {txn_hour}

Customer Temporal Summary:
{history_summary}

Pre-computed Temporal Analysis:
{stats_summary}

Analyze whether this transaction is temporally suspicious.
"""
        )
    ]

    response = structured_model.invoke(messages)

    state["temporal_risk"] = response.temporal_risk
    state["temporal_label"] = response.temporal_label
    state["temporal_reason"] = response.temporal_reason
    state["temporal_decay_risk"] = round(composite_temporal, 4)
    state["nodes"].append(
        {
            "id": "temporal_agent",
            "name": "Temporal Agent",
            "risk": response.temporal_risk,
            "label": response.temporal_label,
            "reason": response.temporal_reason,
        }
    )
    return state
