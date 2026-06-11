from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tools.geo_tool import geo_risk_score
import os
import pandas as pd

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


class GeoSchema(BaseModel):
    geo_risk: float = Field(ge=0, le=1)
    geo_label: str
    geo_reason: str

structured_model = model.with_structured_output(GeoSchema)


def geo_agent(state: dict) -> dict:

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

    tool_risk, tool_reason = geo_risk_score(txn, history_df)

    geo_context = f"""
    IP Country: {txn.get('ipCountry', 'N/A')}
    IP State: {txn.get('ipState', 'N/A')}
    Shipping Country: {txn.get('shippingCountry', 'N/A')}
    Billing Country: {txn.get('billingCountry', 'N/A')}
    Account Country: {txn.get('accountCountry', 'N/A')}
    Location: {txn.get('location', 'N/A')}
    Proxy IP: {txn.get('isProxyIP', 'N/A')}
    """

    messages = [
        SystemMessage(
            content="""
You are a senior fraud analyst specializing in geographic fraud patterns.

You have received a composite geo risk score from a tool that checks:
1. Distance from historical transaction locations (sigmoid-smoothed)
2. Impossible travel velocity detection
3. Country mismatch (IP vs shipping vs billing vs account)

Interpret the tool output and geographic context to produce a final assessment.

Return:
- geo_risk (0 to 1)
- geo_label (Low | Medium | High)
- geo_reason (short explanation)
"""
        ),
        HumanMessage(
            content=f"""
Transaction:
{txn}

Geographic Context:
{geo_context}

Tool Output:
Risk Score: {tool_risk}
Reason: {tool_reason}

Provide final geo fraud assessment.
"""
        )
    ]

    response = structured_model.invoke(messages)

    state["geo_risk"] = response.geo_risk
    state["geo_label"] = response.geo_label
    state["geo_reason"] = response.geo_reason
    state["nodes"].append(
        {
            "id": "geo_agent",
            "name": "Geo Agent",
            "risk": response.geo_risk,
            "label": response.geo_label,
            "reason": response.geo_reason,
        }
    )
    return state
