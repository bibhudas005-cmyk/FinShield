from typing import TypedDict, Dict, Any, List


class FraudState(TypedDict, total=False):
    # ── Incoming transaction (dict with all Microsoft + legacy fields) ──
    txn: Dict[str, Any]

    # ── Customer history (pandas DataFrame) ──
    customer_txns: Any

    # ── Behavioral Agent ──
    behavioral_risk: float
    behavioral_label: str
    behavioral_reason: str
    behavioral_zscore: float
    behavioral_stat_risk: float

    # ── Temporal Agent ──
    temporal_risk: float
    temporal_label: str
    temporal_reason: str
    temporal_decay_risk: float

    # ── Geo Agent ──
    geo_risk: float
    geo_label: str
    geo_reason: str

    # ── Device Agent ──
    device_risk: float
    device_label: str
    device_reason: str

    # ── LLM Decision ──
    decision: str
    action: str
    llm_reasoning: str

    # ── Visualization ──
    nodes: List[Dict[str, Any]]

    # ── Observability ──
    trace: List[str]
