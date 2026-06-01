"""
Build a curated dataset: 5 showcase transactions + background history.
  1. Indian UPI — Fraud (high amount)
  2. Indian UPI — Legit
  3. Microsoft (Foreign) — Fraud
  4. Microsoft (Foreign) — Legit
  5. Indian UPI — Fraud (different city)
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

upi = pd.read_csv(DATA / "indian_upi_fraud.csv")
ms = pd.read_csv(DATA / "microsoft_transactions.csv", low_memory=False)
print(f"Indian UPI: {len(upi)} rows, {upi['fraud'].sum()} fraud")
print(f"Microsoft:  {len(ms)} rows, {ms['isFraud'].sum()} fraud")

# ── Map Indian UPI rows to FinShield schema ──
def map_upi(row):
    dt = pd.to_datetime(f"{row['Date']} {row['Time']}", format="%d-%m-%Y %I:%M:%S %p", errors="coerce")
    return {
        "transactionId": row["Transaction_ID"],
        "customerId": row["Customer_ID"],
        "amount": row["amount"],
        "timestamp": dt.isoformat() if pd.notna(dt) else "",
        "localHour": dt.hour if pd.notna(dt) else None,
        "deviceId": str(row.get("Device_ID", ""))[:12],
        "deviceType": row.get("Device_OS", ""),
        "ipAddress": row.get("IP_Address", ""),
        "ipState": row.get("Transaction_State", ""),
        "ipCountry": "IN",
        "isProxyIP": "FALSE",
        "browserType": "",
        "paymentType": row.get("Payment_Gateway", ""),
        "cardType": "UPI",
        "cvvResult": "",
        "location": row.get("Transaction_City", ""),
        "shippingCountry": "IN",
        "billingCountry": "IN",
        "merchant": row.get("Merchant_Category", ""),
        "transactionType": row.get("Transaction_Type", ""),
        "transactionMethod": row.get("Transaction_Channel", ""),
        "digitalItemCount": 0,
        "physicalItemCount": 0,
        "accountCity": row.get("Transaction_City", ""),
        "accountCountry": "IN",
        "accountAge": max(1, int(row.get("Days_Since_Last_Transaction", 30))),
        "isRegistered": "TRUE",
        "recentRejects": 0,
        "isFraud": int(row["fraud"]),
        "source": "Indian UPI Fraud Dataset",
    }

# ── Pick 5 showcase transactions ──
# 1. Indian fraud — highest amount
indian_fraud = upi[upi["fraud"] == 1].nlargest(1, "amount").iloc[0]
# 2. Indian legit — mid-range amount
indian_legit = upi[upi["fraud"] == 0].iloc[len(upi[upi["fraud"] == 0]) // 2]
# 5. Indian fraud — different city than #1
other_city_frauds = upi[(upi["fraud"] == 1) & (upi["Transaction_City"] != indian_fraud["Transaction_City"])]
indian_fraud2 = other_city_frauds.nlargest(1, "amount").iloc[0]

showcase_indian = [
    map_upi(indian_fraud),
    map_upi(indian_legit),
    map_upi(indian_fraud2),
]

# 3. Microsoft fraud — pick one
ms_fraud_row = ms[ms["isFraud"] == 1].iloc[0].to_dict()
# 4. Microsoft legit — pick one
ms_legit_row = ms[ms["isFraud"] == 0].iloc[0].to_dict()

showcase_ids = {
    showcase_indian[0]["transactionId"],  # Indian fraud
    showcase_indian[1]["transactionId"],  # Indian legit
    showcase_indian[2]["transactionId"],  # Indian fraud 2
    ms_fraud_row["transactionId"],        # Foreign fraud
    ms_legit_row["transactionId"],        # Foreign legit
}

# ── Build history: 15 Indian UPI rows per city for background ──
history_cities = {
    showcase_indian[0]["location"],
    showcase_indian[1]["location"],
    showcase_indian[2]["location"],
}
indian_history = []
for city in history_cities:
    city_rows = upi[upi["Transaction_City"] == city].head(15)
    for _, r in city_rows.iterrows():
        mapped = map_upi(r)
        mapped["customerId"] = f"IND_{city.upper().replace(' ', '_')[:10]}"
        indian_history.append(mapped)

# Also override the showcase indian txns' customerIds to match their city history
for s in showcase_indian:
    s["customerId"] = f"IND_{s['location'].upper().replace(' ', '_')[:10]}"

# ── Microsoft history: take all rows from the same customers ──
ms_fraud_cust = ms_fraud_row["customerId"]
ms_legit_cust = ms_legit_row["customerId"]
ms_history = ms[ms["customerId"].isin({ms_fraud_cust, ms_legit_cust})].to_dict("records")
for r in ms_history:
    r["source"] = "Microsoft R-Server Fraud Detection Dataset"

# ── Combine everything ──
all_rows = []

# Add Indian history
for r in indian_history:
    r["isShowcase"] = r["transactionId"] in showcase_ids
    all_rows.append(r)

# Add showcase Indian rows if not already in history
existing_ids = {r["transactionId"] for r in all_rows}
for s in showcase_indian:
    if s["transactionId"] not in existing_ids:
        s["isShowcase"] = True
        all_rows.append(s)

# Add Microsoft rows
for r in ms_history:
    r["isShowcase"] = r["transactionId"] in showcase_ids
    all_rows.append(r)

out = pd.DataFrame(all_rows)
out_path = DATA / "curated_transactions.csv"
out.to_csv(out_path, index=False)

showcases = out[out["isShowcase"] == True]
print(f"\nSaved {len(out)} total rows ({len(showcases)} showcase) to {out_path}")
print("\nShowcase transactions:")
for _, s in showcases.iterrows():
    src = "🇮🇳 Indian" if "Indian" in str(s.get("source", "")) else "🌍 Foreign"
    label = "FRAUD" if s.get("isFraud", 0) == 1 else "LEGIT"
    print(f"  {src} | {label:5s} | ₹{s['amount']:.2f} | {s.get('location', '?')} | {s['transactionId'][:12]}...")
