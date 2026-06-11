from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tools.device_tool import device_risk_score
import os
import pandas as pd

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


class DeviceSchema(BaseModel):
    device_risk: float = Field(ge=0, le=1)
    device_label: str
    device_reason: str


structured_model = model.with_structured_output(DeviceSchema)


def device_agent(state: dict) -> dict:

    state.setdefault("nodes", [])
    txn = state.get("txn") or state.get("transaction") or {}
    customer_txns = state.get("customer_txns")

    tool_risk, tool_reason = device_risk_score(txn, customer_txns)

    device_context = f"""
    Device ID: {txn.get('deviceId', 'N/A')}
    Device Type: {txn.get('deviceType', 'N/A')}
    Browser: {txn.get('browserType', 'N/A')}
    Proxy IP: {txn.get('isProxyIP', 'N/A')}
    IP Address: {txn.get('ipAddress', 'N/A')}
    """

    if isinstance(customer_txns, pd.DataFrame) and not customer_txns.empty:
        known_devices = customer_txns["deviceId"].dropna().unique().tolist()[:10]
        known_types = (
            customer_txns["deviceType"].dropna().unique().tolist()
            if "deviceType" in customer_txns.columns else []
        )
        known_browsers = (
            customer_txns["browserType"].dropna().unique().tolist()
            if "browserType" in customer_txns.columns else []
        )
        history_summary = f"""
        Known Devices: {known_devices}
        Known Device Types: {known_types}
        Known Browsers: {known_browsers}
        Total Historical Transactions: {len(customer_txns)}
        """
    else:
        history_summary = "No device history available for this customer."

    messages = [
        SystemMessage(
            content="""
You are a senior fraud analyst specializing in device-based fraud detection.

You have received a composite device risk score from a tool that checks:
1. Known vs unknown device (Shannon entropy of device diversity)
2. Device frequency ratio
3. Proxy IP detection
4. Browser type consistency
5. Device type consistency

Interpret the tool output along with the device context and history.

Return:
- device_risk (0 to 1)
- device_label (Low | Medium | High)
- device_reason (short explanation)
"""
        ),
        HumanMessage(
            content=f"""
Transaction:
{txn}

Device Context:
{device_context}

Customer Device History:
{history_summary}

Tool Output:
Risk Score: {tool_risk}
Reason: {tool_reason}

Provide final device fraud assessment.
"""
        )
    ]

    try:
        response = structured_model.invoke(messages)
        risk = response.device_risk
        label = response.device_label
        reason = response.device_reason
    except Exception:
        risk = tool_risk
        reason = tool_reason
        if risk < 0.33:
            label = "Low"
        elif risk < 0.66:
            label = "Medium"
        else:
            label = "High"

    state["device_risk"] = risk
    state["device_label"] = label
    state["device_reason"] = reason

    state["nodes"].append({
        "id": "device_agent",
        "name": "Device Agent",
        "risk": risk,
        "label": label,
        "reason": reason,
    })

    return state
