import math
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

CITY_COORDS = {
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "tokyo": (35.6895, 139.6917),
    "paris": (48.8566, 2.3522),
    "dubai": (25.2048, 55.2708),
    "singapore": (1.3521, 103.8198),
    "san francisco": (37.7749, -122.4194),
    "los angeles": (34.0522, -118.2437),
    "berlin": (52.5200, 13.4050),
    "moscow": (55.7558, 37.6173),
    "sydney": (33.8688, 151.2093),
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "hong kong": (22.3193, 114.1694),
}


def _safe_str(val) -> str:
    """Convert any value (including NaN) to a clean string."""
    if val is None:
        return ""
    if isinstance(val, float):
        import math as _m
        if _m.isnan(val):
            return ""
        return str(val)
    return str(val).strip()


def resolve_coordinates(txn: dict):
    lat = txn.get("latitude")
    lon = txn.get("longitude")
    if lat and lon:
        try:
            return float(lat), float(lon)
        except (ValueError, TypeError):
            pass

    location = _safe_str(txn.get("location")).lower()
    if location in CITY_COORDS:
        return CITY_COORDS[location]

    for key in ("ipState", "ipCountry", "accountCity", "merchant"):
        val = _safe_str(txn.get(key)).lower()
        if not val:
            continue
        for city, coords in CITY_COORDS.items():
            if city in val:
                return coords

    return None, None


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def distance_to_risk(distance_km: float) -> float:
    """
    Sigmoid-smoothed distance risk:
        risk = σ((d - μ) / s)
    """
    return sigmoid((distance_km - 100.0) / 60.0)


def travel_velocity(txn: dict, history_df: pd.DataFrame, current_lat, current_lon):
    """
    Impossible Travel:  velocity = haversine_distance / Δt  (km/h)
    """
    txn_time = pd.to_datetime(txn.get("timestamp"), errors="coerce")
    if txn_time is None or pd.isna(txn_time):
        return None, "No timestamp for velocity check"

    if history_df.empty or "timestamp" not in history_df.columns:
        return None, "No historical timestamps"

    has_geo = "latitude" in history_df.columns and "longitude" in history_df.columns
    if not has_geo:
        return None, "No historical geo coordinates"

    history_sorted = history_df.dropna(subset=["timestamp", "latitude", "longitude"]).copy()
    history_sorted["timestamp"] = pd.to_datetime(history_sorted["timestamp"], errors="coerce")
    history_sorted = history_sorted.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)

    if history_sorted.empty:
        return None, "No valid historical timestamps"

    last_row = history_sorted.iloc[0]
    delta_hours = (txn_time - last_row["timestamp"]).total_seconds() / 3600.0

    if delta_hours <= 0:
        return None, "Transaction not after last known transaction"

    dist_km = haversine_distance(
        current_lat, current_lon,
        float(last_row["latitude"]), float(last_row["longitude"]),
    )
    velocity_kmh = dist_km / delta_hours

    return velocity_kmh, f"Velocity={velocity_kmh:.0f}km/h over {dist_km:.0f}km in {delta_hours:.1f}h"


def country_mismatch_risk(txn: dict) -> tuple[float, str]:
    """
    Cross-country mismatch detection:
      - IP country vs shipping country
      - Shipping country vs billing country
      - IP country vs account country

    Each mismatch adds 0.3 risk (capped at 0.9).
    """
    ip_country = _safe_str(txn.get("ipCountry")).lower()
    ship_country = _safe_str(txn.get("shippingCountry")).lower()
    bill_country = _safe_str(txn.get("billingCountry")).lower()
    acct_country = _safe_str(txn.get("accountCountry")).lower()

    mismatches = []
    risk = 0.0

    if ip_country and ship_country and ip_country != ship_country:
        mismatches.append(f"IP({ip_country})≠Ship({ship_country})")
        risk += 0.3

    if ship_country and bill_country and ship_country != bill_country:
        mismatches.append(f"Ship({ship_country})≠Bill({bill_country})")
        risk += 0.3

    if ip_country and acct_country and ip_country != acct_country:
        mismatches.append(f"IP({ip_country})≠Acct({acct_country})")
        risk += 0.3

    risk = min(risk, 0.9)

    if mismatches:
        return risk, f"Country mismatches: {'; '.join(mismatches)}"
    return 0.0, "No country mismatches detected"


def geo_risk_score(txn: dict, history_df: pd.DataFrame):
    """
    Composite geographic risk:
      1. Sigmoid-smoothed distance anomaly  (0.35 weight)
      2. Impossible travel velocity          (0.30 weight)
      3. Country mismatch detection          (0.35 weight)

    R_geo = 0.35·dist + 0.30·velocity + 0.35·mismatch
    """
    current_lat, current_lon = resolve_coordinates(txn)
    mismatch_risk, mismatch_detail = country_mismatch_risk(txn)

    has_coords = current_lat is not None and current_lon is not None
    has_history_geo = (
        not history_df.empty
        and "latitude" in history_df.columns
        and "longitude" in history_df.columns
    )

    if not has_coords or not has_history_geo:
        composite = round(0.35 * 0.5 + 0.30 * 0.5 + 0.35 * mismatch_risk, 4)
        reason_parts = ["No geo-distance data available (default 0.5)"]
        if mismatch_detail:
            reason_parts.append(mismatch_detail)
        return max(0.0, min(1.0, composite)), ". ".join(reason_parts)

    distances = []
    for _, row in history_df.iterrows():
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            d = haversine_distance(current_lat, current_lon, float(row["latitude"]), float(row["longitude"]))
            distances.append(d)

    if not distances:
        composite = round(0.35 * 0.5 + 0.30 * 0.5 + 0.35 * mismatch_risk, 4)
        return max(0.0, min(1.0, composite)), f"No historical geo coords. {mismatch_detail}"

    min_distance = min(distances)
    dist_risk = distance_to_risk(min_distance)

    vel_kmh, vel_detail = travel_velocity(txn, history_df, current_lat, current_lon)
    if vel_kmh is not None:
        if vel_kmh > 900:
            vel_risk = 0.95
        elif vel_kmh > 500:
            vel_risk = 0.75
        elif vel_kmh > 200:
            vel_risk = 0.50
        else:
            vel_risk = 0.10
    else:
        vel_risk = 0.0

    composite = round(
        0.35 * dist_risk + 0.30 * vel_risk + 0.35 * mismatch_risk,
        4,
    )

    reason = (
        f"Dist={min_distance:.0f}km(risk={dist_risk:.2f}), "
        f"{vel_detail}, "
        f"{mismatch_detail}. "
        f"Composite={composite:.2f}"
    )

    return max(0.0, min(1.0, composite)), reason
