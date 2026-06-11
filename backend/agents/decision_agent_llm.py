from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import json
import re
import os

from agents.behavioral_agent import behavioral_agent
from agents.temporal_agent import temporal_agent
from agents.geo_agent import geo_agent
from agents.device_agent import device_agent

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def decision_agent_llm(state: dict) -> dict:
    """
    LLM Decision Agent (Orchestrator)
    ------------------
    Orchestrates upstream agents, then returns final decision + action.
    """

    # Ensure trace exists
    state.setdefault("trace", [])
    state["trace"].append("🤖 LLM Decision Agent started")
    state.setdefault("nodes", [])

    # Orchestrate the other agents first
    state = behavioral_agent(state)
    state = temporal_agent(state)
    state = geo_agent(state)
    state = device_agent(state)

    txn = state.get("txn") or {}

    messages = [
        SystemMessage(
            content="""
You are a senior banking fraud decision engine performing the final adjudication.

You will receive detailed fraud signals from four specialized agents (behavioral,
temporal, geographic, device) including their risk scores, labels, and reasoning.

Your job is to synthesize ALL signals into a comprehensive verdict.

Return ONLY valid JSON (no markdown, no extra text).

Schema:
{
  "decision": "LOW_RISK" | "MID_RISK" | "HIGH_RISK",
  "action": "ALLOW" | "REVIEW" | "BLOCK",
  "reasoning": "2-3 concise sentences. Cite the top 2-3 risk factors with their scores. End with what action to take and why. No filler, no repetition."
}
""".strip()
        ),
        HumanMessage(
            content=f"""
Transaction under review:
  Amount: ${txn.get('amount', 'N/A')}
  Customer: {txn.get('customerId', 'N/A')}
  Location: {txn.get('location', 'N/A')}
  IP Country: {txn.get('ipCountry', 'N/A')}
  Account Age: {txn.get('accountAge', 'N/A')} days
  CVV Result: {txn.get('cvvResult', 'N/A')}
  Proxy IP: {txn.get('isProxyIP', 'N/A')}
  Recent Rejects: {txn.get('recentRejects', 'N/A')}

Agent Signals:
1. BEHAVIORAL (weight=0.30):
   risk={state.get("behavioral_risk", 0.5)}, label={state.get("behavioral_label", "N/A")}
   Z-score={state.get("behavioral_zscore", "N/A")}, Statistical risk={state.get("behavioral_stat_risk", "N/A")}
   Reason: {state.get("behavioral_reason", "N/A")}

2. TEMPORAL (weight=0.20):
   risk={state.get("temporal_risk", 0.5)}, label={state.get("temporal_label", "N/A")}
   Decay composite={state.get("temporal_decay_risk", "N/A")}
   Reason: {state.get("temporal_reason", "N/A")}

3. GEOGRAPHIC (weight=0.25):
   risk={state.get("geo_risk", 0.5)}, label={state.get("geo_label", "N/A")}
   Reason: {state.get("geo_reason", "N/A")}

4. DEVICE (weight=0.25):
   risk={state.get("device_risk", 0.5)}, label={state.get("device_label", "N/A")}
   Reason: {state.get("device_reason", "N/A")}

Provide the final decision with detailed reasoning.
""".strip()
        ),
    ]

    # Invoke LLM
    try:
        response = llm.invoke(messages)
        raw_text = response.content

        # Extract JSON safely even if LLM adds extra text
        json_match = re.search(r"\{.*\}", raw_text, re.S)
        if not json_match:
            raise ValueError(f"LLM did not return JSON: {raw_text}")

        result = json.loads(json_match.group())

    except Exception as e:
        # Minimal fallback if LLM fails (no hard-coded weighting)
        labels = [
            state.get("behavioral_label"),
            state.get("temporal_label"),
            state.get("geo_label"),
            state.get("device_label"),
        ]
        if "High" in labels:
            result = {"decision": "HIGH_RISK", "action": "BLOCK", "reasoning": "Fallback: at least one agent flagged High risk."}
        elif "Medium" in labels:
            result = {"decision": "MID_RISK", "action": "REVIEW", "reasoning": "Fallback: at least one agent flagged Medium risk."}
        else:
            result = {"decision": "LOW_RISK", "action": "ALLOW", "reasoning": "Fallback: no agent flagged elevated risk."}
        state["trace"].append(f"⚠️ LLM failed, fallback used: {str(e)}")

    # Update trace for observability
    state["trace"].append(f"Decision={result['decision']}, Action={result['action']}")

    state["decision"] = result["decision"]
    state["action"] = result["action"]
    state["llm_reasoning"] = result["reasoning"]
    state["nodes"].append(
        {
            "id": "llm_agent",
            "name": "LLM Decision Agent",
            "decision": result["decision"],
            "action": result["action"],
            "reasoning": result["reasoning"],
        }
    )

    return state