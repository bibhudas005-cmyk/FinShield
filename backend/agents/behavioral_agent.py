from langchain_groq import ChatGroq
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


class BehaviouralSchema(BaseModel):
    behavioral_risk: float = Field(
        description="Risk score between 0 and 1",
        ge=0,
        le=1
    )
    behavioral_label: str = Field(
        description="Low | Medium | High"
    )
    behavioral_reason: str = Field(
        description="Short explanation for assigned risk"
    )


structured_model = model.with_structured_output(BehaviouralSchema)


def z_score(value: float, mean: float, std: float) -> float:
    """Z = (x - μ) / σ"""
    if std == 0:
        return 0.0
    return (value - mean) / std


def sigmoid(x: float) -> float:
    """σ(x) = 1 / (1 + e^(-x))"""
    return 1.0 / (1.0 + math.exp(-x))


def statistical_risk_from_zscore(z: float, threshold: float = 1.5) -> float:
    """risk = σ(|Z| - threshold)"""
    return sigmoid(abs(z) - threshold)


def cvv_risk(txn: dict) -> tuple[float, str]:
    """
    CVV verification result risk scoring:
      M (Match) → 0.0
      N (No Match) → 0.9
      P (Not Processed) → 0.4
      S (Should be present but missing) → 0.7
      U (Uncertified) → 0.3
      Empty → 0.5
    """
    result = (txn.get("cvvResult") or "").strip().upper()
    scores = {"M": 0.0, "N": 0.9, "P": 0.4, "S": 0.7, "U": 0.3}
    score = scores.get(result, 0.5)
    return score, f"CVV={result or 'empty'}(risk={score:.1f})"


def account_age_risk(txn: dict) -> tuple[float, str]:
    """
    Account age risk: newer accounts are higher risk.
    risk = σ(-(age - 30) / 20)
    At age=0 → 0.82, age=30 → 0.50, age=90 → 0.05
    """
    age = int(txn.get("accountAge") or 0)
    risk = sigmoid(-(age - 30) / 20.0)
    return risk, f"AcctAge={age}d(risk={risk:.2f})"


def recent_rejects_risk(txn: dict) -> tuple[float, str]:
    """
    Recent payment rejection risk:
      0 rejects → 0.0
      1 reject → 0.4
      2+ rejects → 0.8
    """
    rejects = int(txn.get("recentRejects") or 0)
    if rejects == 0:
        return 0.0, "Rejects=0"
    if rejects == 1:
        return 0.4, "Rejects=1(risk=0.4)"
    return 0.8, f"Rejects={rejects}(risk=0.8)"


def behavioral_agent(state: dict) -> dict:
    state.setdefault("nodes", [])
    txn = state.get("txn") or state.get("transaction") or {}
    history_source = (
        state.get("customer_txns")
        if state.get("customer_txns") is not None
        else state.get("transaction_history", [])
    )

    if isinstance(history_source, pd.DataFrame):
        history_df = history_source.copy()
    else:
        history_df = pd.DataFrame(history_source)

    if "amount" not in history_df.columns:
        history_df["amount"] = pd.Series(dtype=float)

    current_amount = float(txn.get("amount", 0))

    if history_df.empty or history_df["amount"].isna().all():
        z = 0.0
        stat_risk = 0.5
        history_summary = "No previous transaction history available."
        stats_summary = "Z-Score: N/A (no history), Statistical Risk: 0.50"
    else:
        mean_amt = history_df["amount"].mean()
        std_amt = history_df["amount"].std()
        z = z_score(current_amount, mean_amt, std_amt)
        stat_risk = statistical_risk_from_zscore(z)

        history_summary = f"""
        Total Transactions: {len(history_df)}
        Average Amount (μ): {mean_amt:.2f}
        Std Deviation (σ): {std_amt:.2f}
        Maximum Amount: {history_df['amount'].max():.2f}
        Minimum Amount: {history_df['amount'].min():.2f}
        """
        stats_summary = (
            f"Z-Score: {z:.2f} = ({current_amount:.0f} - {mean_amt:.0f}) / {std_amt:.0f}, "
            f"Statistical Risk: σ(|{z:.2f}| - 1.5) = {stat_risk:.2f}"
        )

    cvv_score, cvv_detail = cvv_risk(txn)
    age_score, age_detail = account_age_risk(txn)
    reject_score, reject_detail = recent_rejects_risk(txn)

    enrichment_summary = f"""
    {cvv_detail}
    {age_detail}
    {reject_detail}
    Payment Type: {txn.get('paymentType', 'N/A')}
    Digital Items: {txn.get('digitalItemCount', 0)}, Physical Items: {txn.get('physicalItemCount', 0)}
    Registered Account: {txn.get('isRegistered', 'N/A')}
    """

    prompt = f"""
    You are a senior financial fraud analyst.

    Current Transaction:
    {txn}

    Customer Behavioral History:
    {history_summary}

    Statistical Analysis:
    {stats_summary}

    Enrichment Signals:
    {enrichment_summary}

    The Z-score measures standard deviations from the mean. Z > 2 is highly anomalous.
    CVV mismatch (N) is a strong fraud indicator.
    New accounts (low age) and recent payment rejects elevate risk.

    Provide:
    - Risk score between 0 and 1
    - Behavioral label (Low, Medium, High)
    - Short explanation referencing Z-score and enrichment signals
    """

    response = structured_model.invoke(prompt)

    state["behavioral_risk"] = response.behavioral_risk
    state["behavioral_label"] = response.behavioral_label
    state["behavioral_reason"] = response.behavioral_reason
    state["behavioral_zscore"] = round(z, 4)
    state["behavioral_stat_risk"] = round(stat_risk, 4)
    state["nodes"].append(
        {
            "id": "behavioral_agent",
            "name": "Behavioral Agent",
            "risk": response.behavioral_risk,
            "label": response.behavioral_label,
            "reason": response.behavioral_reason,
        }
    )
    return state
