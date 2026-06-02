"""
Data Preparation Script
-----------------------
Source: Microsoft R-Server Fraud Detection Dataset
  https://github.com/microsoft/r-server-fraud-detection

Merges Untagged_Transactions.csv + Fraud_Transactions.csv + Account_Info.csv
into a unified FinShield-schema CSV: microsoft_transactions.csv
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent

untagged = pd.read_csv(RAW_DIR / "data_raw_untagged.csv", low_memory=False)
fraud = pd.read_csv(RAW_DIR / "data_raw_fraud.csv", low_memory=False)
accounts = pd.read_csv(RAW_DIR / "data_raw_accounts.csv", low_memory=False)

print(f"Loaded: {len(untagged)} untagged, {len(fraud)} fraud, {len(accounts)} accounts")

# --- 1. Label fraud transactions ---
fraud_ids = set(fraud["transactionID"].unique())
untagged["isFraud"] = untagged["transactionID"].isin(fraud_ids).astype(int)
print(f"Fraud labels: {untagged['isFraud'].sum()} fraudulent out of {len(untagged)}")

# --- 2. Join account info (latest account snapshot per accountID) ---
accounts_dedup = accounts.sort_values("transactionDate", ascending=False).drop_duplicates(
    subset=["accountID"], keep="first"
)
accounts_dedup = accounts_dedup[
    ["accountID", "accountCity", "accountState", "accountCountry",
     "accountAge", "isUserRegistered", "numPaymentRejects1dPerUser"]
]
merged = untagged.merge(accounts_dedup, on="accountID", how="left")

# --- 3. Pick interesting accounts: those with BOTH fraud and legit txns ---
acct_fraud_counts = merged.groupby("accountID")["isFraud"].sum()
acct_total_counts = merged.groupby("accountID")["isFraud"].count()
mixed_accounts = acct_fraud_counts[(acct_fraud_counts > 0) & (acct_total_counts > 3)].index.tolist()
print(f"Accounts with fraud + history: {len(mixed_accounts)}")

sample_accounts = mixed_accounts[:20]
sample = merged[merged["accountID"].isin(sample_accounts)].copy()
print(f"Sampled {len(sample)} transactions from {len(sample_accounts)} accounts")

# --- 4. Parse timestamps ---
def parse_timestamp(row):
    d = str(row["transactionDate"])
    t = str(row["transactionTime"]).zfill(6)
    try:
        return pd.to_datetime(f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}")
    except Exception:
        return pd.NaT

sample["timestamp"] = sample.apply(parse_timestamp, axis=1)

# --- 5. Map to FinShield schema ---
DEVICE_TYPE_MAP = {"P": "PC", "M": "Mobile", "C": "Console", "O": "Other"}
BROWSER_MAP = {"I": "IE", "C": "Chrome", "F": "Firefox", "O": "Other"}
PAYMENT_MAP = {"C": "CreditCard", "D": "DebitCard", "P": "PayPal", "K": "Check", "H": "Cash", "O": "Other"}
CARD_MAP = {"M": "Magnetic", "C": "Chip"}

out = pd.DataFrame()
out["transactionId"] = sample["transactionID"]
out["customerId"] = sample["accountID"]
out["amount"] = sample["transactionAmountUSD"]
out["timestamp"] = sample["timestamp"]
out["localHour"] = sample["localHour"]

out["deviceId"] = sample["transactionDeviceId"].fillna("unknown")
out["deviceType"] = sample["transactionDeviceType"].map(DEVICE_TYPE_MAP).fillna("Unknown")
out["ipAddress"] = sample["transactionIPaddress"].fillna("")
out["ipState"] = sample["ipState"].fillna("")
out["ipCountry"] = sample["ipCountryCode"].fillna("")
out["isProxyIP"] = sample["isProxyIP"].fillna("FALSE")

out["browserType"] = sample["browserType"].map(BROWSER_MAP).fillna("Unknown")
out["paymentType"] = sample["paymentInstrumentType"].map(PAYMENT_MAP).fillna("Unknown")
out["cardType"] = sample["cardType"].map(CARD_MAP).fillna("Unknown")
out["cvvResult"] = sample["cvvVerifyResult"].fillna("")

out["location"] = sample["shippingCity"].fillna(sample["ipState"]).fillna("")
out["shippingCountry"] = sample["shippingCountry"].fillna("")
out["billingCountry"] = sample["paymentBillingCountryCode"].fillna("")
out["merchant"] = sample["purchaseProductType"].fillna("Unknown")

out["transactionType"] = sample["transactionType"].fillna("")
out["transactionMethod"] = sample["transactionMethod"].fillna("")
out["digitalItemCount"] = sample["digitalItemCount"].fillna(0).astype(int)
out["physicalItemCount"] = sample["physicalItemCount"].fillna(0).astype(int)

out["accountCity"] = sample["accountCity"].fillna("")
out["accountCountry"] = sample["accountCountry"].fillna("")
out["accountAge"] = sample["accountAge"].fillna(0).astype(int)
out["isRegistered"] = sample["isUserRegistered"].fillna("FALSE")
out["recentRejects"] = sample["numPaymentRejects1dPerUser"].fillna(0).astype(int)

out["isFraud"] = sample["isFraud"].values

out = out.sort_values(["customerId", "timestamp"]).reset_index(drop=True)

OUT_PATH = RAW_DIR / "microsoft_transactions.csv"
out.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(out)} transactions to {OUT_PATH}")
print(f"Columns: {list(out.columns)}")
print(f"\nFraud distribution:\n{out['isFraud'].value_counts().to_string()}")
print(f"\nSample accounts:\n{out.groupby('customerId').agg(txns=('transactionId','count'), frauds=('isFraud','sum')).head(20).to_string()}")
