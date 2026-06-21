import math
import pandas as pd


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def shannon_entropy(device_counts: list[int]) -> float:
    """
    Shannon Entropy: H = -Σ p_i · log₂(p_i)

    Measures diversity of device usage. Higher entropy means the customer
    uses many different devices (more unpredictable, higher baseline risk).
    """
    total = sum(device_counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in device_counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def device_frequency_ratio(device_id: str, customer_txns: pd.DataFrame) -> float:
    """
    Frequency Ratio: f_device = n_device / N_total

    How often this specific device appears in the customer's history.
    """
    if customer_txns is None or customer_txns.empty:
        return 0.0
    total = len(customer_txns)
    device_count = (customer_txns["deviceId"] == device_id).sum()
    return device_count / total


def proxy_risk(txn: dict) -> float:
    """Proxy IP detection: isProxyIP=TRUE adds significant risk."""
    val = _safe_str(txn.get("isProxyIP")).upper()
    return 0.85 if val == "TRUE" else 0.0


def browser_consistency_risk(txn: dict, customer_txns) -> float:
    """
    Browser consistency: if the browser type has never been used by this
    customer before, it's a mild risk signal.
    """
    browser = _safe_str(txn.get("browserType"))
    if not browser or customer_txns is None or not hasattr(customer_txns, "empty") or customer_txns.empty:
        return 0.0
    if "browserType" not in customer_txns.columns:
        return 0.0
    known_browsers = customer_txns["browserType"].dropna().unique().tolist()
    if not known_browsers:
        return 0.0
    return 0.0 if browser in known_browsers else 0.4


def device_type_consistency_risk(txn: dict, customer_txns) -> float:
    """
    Device type consistency: if the customer always uses 'PC' and this
    transaction is from 'Mobile', it's a risk signal.
    """
    dtype = _safe_str(txn.get("deviceType"))
    if not dtype or customer_txns is None or not hasattr(customer_txns, "empty") or customer_txns.empty:
        return 0.0
    if "deviceType" not in customer_txns.columns:
        return 0.0
    known_types = customer_txns["deviceType"].dropna().unique().tolist()
    if not known_types:
        return 0.0
    return 0.0 if dtype in known_types else 0.3


def device_risk_score(txn: dict, customer_txns):
    """
    Composite device risk using:
      1. Known/unknown device:      0.25 weight
      2. Shannon Entropy:           0.15 weight
      3. Device frequency ratio:    0.15 weight
      4. Proxy IP detection:        0.20 weight
      5. Browser consistency:       0.10 weight
      6. Device type consistency:   0.15 weight

    R_device = Σ(w_i · signal_i)
    """
    device_id = _safe_str(txn.get("deviceId")) or "unknown"

    if customer_txns is None or (hasattr(customer_txns, "empty") and customer_txns.empty):
        p_risk = proxy_risk(txn)
        base = max(0.5, p_risk)
        return base, f"No device history. Proxy={txn.get('isProxyIP','?')}. Risk={base:.2f}"

    known_devices = customer_txns["deviceId"].dropna().unique().tolist()
    is_known = device_id in known_devices

    known_signal = 0.1 if is_known else 0.9

    device_counts = customer_txns["deviceId"].value_counts().tolist()
    entropy = shannon_entropy(device_counts)
    max_entropy = math.log2(len(known_devices)) if len(known_devices) > 1 else 1.0
    entropy_norm = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0

    freq = device_frequency_ratio(device_id, customer_txns)

    p_risk = proxy_risk(txn)
    b_risk = browser_consistency_risk(txn, customer_txns)
    dt_risk = device_type_consistency_risk(txn, customer_txns)

    composite = round(
        0.25 * known_signal
        + 0.15 * entropy_norm
        + 0.15 * (1 - freq)
        + 0.20 * p_risk
        + 0.10 * b_risk
        + 0.15 * dt_risk,
        4,
    )
    composite = max(0.0, min(1.0, composite))

    parts = [
        f"known={'yes' if is_known else 'no'}",
        f"entropy={entropy:.2f}",
        f"freq={freq:.0%}",
        f"proxy={txn.get('isProxyIP', '?')}",
        f"browser={txn.get('browserType', '?')}",
        f"devType={txn.get('deviceType', '?')}",
    ]
    reason = f"Device signals: {', '.join(parts)}. Composite={composite:.2f}"

    return composite, reason
