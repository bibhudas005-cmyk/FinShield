import asyncio
import math
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fraud_graph import evaluate


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"


# ── Extended schema matching Microsoft R-Server Fraud Detection Dataset ──

class TransactionRequest(BaseModel):
    transactionId: str
    customerId: str
    amount: float
    timestamp: str
    merchant: Optional[str] = "Unknown"
    location: Optional[str] = ""
    deviceId: Optional[str] = "unknown"
    deviceType: Optional[str] = ""
    ipAddress: Optional[str] = ""
    ipState: Optional[str] = ""
    ipCountry: Optional[str] = ""
    isProxyIP: Optional[str] = "FALSE"
    browserType: Optional[str] = ""
    paymentType: Optional[str] = ""
    cardType: Optional[str] = ""
    cvvResult: Optional[str] = ""
    shippingCountry: Optional[str] = ""
    billingCountry: Optional[str] = ""
    localHour: Optional[int] = None
    transactionType: Optional[str] = ""
    transactionMethod: Optional[str] = ""
    digitalItemCount: Optional[int] = 0
    physicalItemCount: Optional[int] = 0
    accountAge: Optional[int] = 0
    accountCountry: Optional[str] = ""
    isRegistered: Optional[str] = "FALSE"
    recentRejects: Optional[int] = 0


class SimulationRequest(BaseModel):
    transactionId: str
    amount: float
    location: str
    device: str
    customerId: Optional[str] = "CUST01"
    merchant: Optional[str] = "Unknown"
    timestamp: Optional[str] = None
    deviceType: Optional[str] = ""
    ipCountry: Optional[str] = ""
    isProxyIP: Optional[str] = "FALSE"
    browserType: Optional[str] = ""
    paymentType: Optional[str] = ""
    cvvResult: Optional[str] = ""
    shippingCountry: Optional[str] = ""
    billingCountry: Optional[str] = ""
    accountAge: Optional[int] = 0
    recentRejects: Optional[int] = 0


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/fraud/check")
async def check_fraud(txn: TransactionRequest):
    result = await asyncio.to_thread(evaluate, txn.dict())
    return sanitize(result)


AGENT_WEIGHTS = {
    "behavioral_agent": 0.30,
    "temporal_agent": 0.20,
    "geo_agent": 0.25,
    "device_agent": 0.25,
}


def weighted_composite_risk(agent_risks: dict) -> float:
    """
    Weighted Composite Risk Score (stacking ensemble — Paper 4):

        R_composite = Σ(w_i · r_i) / Σ(w_i)
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for agent_id, risk in agent_risks.items():
        w = AGENT_WEIGHTS.get(agent_id, 0.25)
        weighted_sum += w * risk
        total_weight += w
    if total_weight == 0:
        return 0.5
    return weighted_sum / total_weight


def transform_pipeline_result(pipeline_result: dict, timestamp: str) -> dict:
    nodes = pipeline_result.get("nodes", [])

    agent_risk_map = {}
    reasoning_steps = []
    action = "REVIEW"
    llm_explanation = ""

    for node in nodes:
        if node["id"] == "llm_agent":
            action = node.get("action", "REVIEW")
            llm_explanation = node.get("reasoning", "")
            reasoning_steps.append(f"Final Decision: {llm_explanation}")
        else:
            risk = node.get("risk", 0.5)
            label = node.get("label", "Medium")
            reason = node.get("reason", "")
            agent_risk_map[node["id"]] = risk
            reasoning_steps.append(f"{node['name']}: {reason} (Risk: {label})")

    composite = weighted_composite_risk(agent_risk_map) if agent_risk_map else 0.5
    risk_score = int(min(97, composite * 100))

    status_map = {"BLOCK": "FLAGGED", "REVIEW": "REVIEW", "ALLOW": "CLEAR"}
    verdict_map = {
        "FLAGGED": "Fraud Detected",
        "REVIEW": "Manual Review Recommended",
        "CLEAR": "Transaction Cleared",
    }

    status = status_map.get(action, "REVIEW")
    verdict = verdict_map.get(status, "Manual Review Recommended")

    explanation = llm_explanation or {
        "FLAGGED": "Multiple agents flagged high-risk signals across behavioral, temporal, geographic, and device dimensions.",
        "REVIEW": "Some agents detected anomalies that deviate from the customer's established patterns.",
        "CLEAR": "All agent assessments stayed within acceptable risk thresholds.",
    }.get(status, "")

    return {
        "riskScore": risk_score,
        "status": status,
        "reasoning": reasoning_steps,
        "verdict": verdict,
        "confidence": min(96, risk_score + 5),
        "explanation": explanation,
        "processedAt": timestamp or datetime.utcnow().isoformat(),
        "agents": nodes,
        "signals": pipeline_result.get("signals", {}),
        "weights": AGENT_WEIGHTS,
        "compositeRisk": round(composite, 4),
        "pipelineUsed": True,
    }


def build_simulation_response(txn: SimulationRequest):
    amount_risk = 32 if txn.amount >= 45000 else 22 if txn.amount >= 25000 else 12
    device_risk = 30 if "new" in txn.device.lower() else 14
    location_risk = (
        20
        if txn.location.lower() in {"mumbai", "delhi", "dubai", "singapore"}
        else 8
    )

    risk_score = min(97, amount_risk + device_risk + location_risk)

    if risk_score >= 75:
        status, verdict = "FLAGGED", "Fraud Detected"
        explanation = "Large-value transfer from a new device with elevated geographic risk markers."
    elif risk_score >= 45:
        status, verdict = "REVIEW", "Manual Review Recommended"
        explanation = "Multiple signals deviated from the expected customer profile and require review."
    else:
        status, verdict = "CLEAR", "Transaction Cleared"
        explanation = "Agent checks stayed within safe thresholds across device, amount, and location."

    return {
        "riskScore": risk_score,
        "status": status,
        "reasoning": [
            "Cross-checking behavioral pattern",
            "Comparing device fingerprint",
            "Checking geo-velocity anomaly",
        ],
        "verdict": verdict,
        "confidence": min(96, risk_score + 5),
        "explanation": explanation,
        "processedAt": txn.timestamp or datetime.utcnow().isoformat(),
        "pipelineUsed": False,
    }


@app.post("/api/transaction")
async def simulate_transaction(txn: SimulationRequest):
    timestamp = txn.timestamp or datetime.utcnow().isoformat()

    try:
        full_txn = {
            "transactionId": txn.transactionId,
            "customerId": txn.customerId,
            "amount": txn.amount,
            "merchant": txn.merchant,
            "location": txn.location,
            "deviceId": txn.device,
            "timestamp": timestamp,
            "deviceType": txn.deviceType,
            "ipCountry": txn.ipCountry,
            "isProxyIP": txn.isProxyIP,
            "browserType": txn.browserType,
            "paymentType": txn.paymentType,
            "cvvResult": txn.cvvResult,
            "shippingCountry": txn.shippingCountry,
            "billingCountry": txn.billingCountry,
            "accountAge": txn.accountAge,
            "recentRejects": txn.recentRejects,
        }
        pipeline_result = await asyncio.to_thread(evaluate, full_txn)
        return sanitize(transform_pipeline_result(pipeline_result, timestamp))

    except Exception as e:
        print(f"[FinShield] Pipeline failed, using deterministic fallback: {e}")
        traceback.print_exc()
        return build_simulation_response(txn)


@app.get("/api/transactions")
async def list_transactions():
    """Return only the 5 showcase transactions for the frontend picker."""
    from fraud_graph import transaction_history
    import math

    df = transaction_history.copy()
    if "isShowcase" in df.columns:
        df = df[df["isShowcase"] == True]

    txns = []
    for _, row in df.iterrows():
        txn = {}
        for col in df.columns:
            if col == "isShowcase":
                continue
            txn[col] = row[col]
        txns.append(txn)
    return sanitize(txns)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ── Serve the React frontend build in production ──

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
