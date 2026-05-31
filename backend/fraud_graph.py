import json
import math
import pandas as pd
from datetime import datetime
from pathlib import Path

from agents.decision_agent_llm import decision_agent_llm

DATA_DIR = Path(__file__).parent / "data"

_csv_candidates = [
    DATA_DIR / "curated_transactions.csv",
    DATA_DIR / "microsoft_transactions.csv",
    DATA_DIR / "transactions.csv",
]
_csv_path = next((p for p in _csv_candidates if p.exists()), _csv_candidates[-1])
transaction_history = pd.read_csv(_csv_path, low_memory=False)

_COLUMN_ALIASES = {
    "transaction_id": "transactionId",
    "customer_id": "customerId",
    "device": "deviceId",
}
transaction_history = transaction_history.rename(
    columns={k: v for k, v in _COLUMN_ALIASES.items() if k in transaction_history.columns}
)
if "timestamp" in transaction_history.columns:
    transaction_history["timestamp"] = pd.to_datetime(
        transaction_history["timestamp"], errors="coerce"
    )

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def evaluate(txn: dict):
    customer_id = txn.get("customerId") or txn.get("customer_id")
    txn_id = txn.get("transactionId") or txn.get("transaction_id")

    customer_txns = transaction_history[
        transaction_history["customerId"] == customer_id
    ]
    if txn_id is not None and "transactionId" in customer_txns.columns:
        customer_txns = customer_txns[customer_txns["transactionId"] != txn_id]

    nodes = []
    state = {
        "txn": txn,
        "customer_txns": customer_txns,
        "nodes": nodes,
    }

    state = decision_agent_llm(state)

    result = {
        "transaction": txn,
        "nodes": nodes,
        "signals": {
            "behavioral": {
                "z_score": state.get("behavioral_zscore"),
                "statistical_risk": state.get("behavioral_stat_risk"),
            },
            "temporal": {
                "decay_risk": state.get("temporal_decay_risk"),
            },
        },
    }

    _save_analysis(txn, state, result)
    return result


def _save_analysis(txn: dict, state: dict, result: dict):
    txn_id = txn.get("transactionId", "unknown")
    analysis = {
        "meta": {
            "source": str(_csv_path.name),
            "evaluated_at": datetime.utcnow().isoformat(),
            "dataset_rows": len(transaction_history),
        },
        "transaction": txn,
        "agent_signals": {
            "behavioral": {
                "risk": state.get("behavioral_risk"),
                "label": state.get("behavioral_label"),
                "reason": state.get("behavioral_reason"),
                "z_score": state.get("behavioral_zscore"),
                "statistical_risk": state.get("behavioral_stat_risk"),
            },
            "temporal": {
                "risk": state.get("temporal_risk"),
                "label": state.get("temporal_label"),
                "reason": state.get("temporal_reason"),
                "decay_risk": state.get("temporal_decay_risk"),
            },
            "geo": {
                "risk": state.get("geo_risk"),
                "label": state.get("geo_label"),
                "reason": state.get("geo_reason"),
            },
            "device": {
                "risk": state.get("device_risk"),
                "label": state.get("device_label"),
                "reason": state.get("device_reason"),
            },
        },
        "decision": {
            "verdict": state.get("decision"),
            "action": state.get("action"),
            "reasoning": state.get("llm_reasoning"),
        },
        "trace": state.get("trace", []),
        "nodes": result.get("nodes", []),
    }
    out_path = RESULTS_DIR / f"{txn_id}.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
