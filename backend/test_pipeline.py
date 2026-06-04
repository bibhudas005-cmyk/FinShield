"""
End-to-end test: pick real transactions from microsoft_transactions.csv
(mix of fraud and legit) and run them through the full agent pipeline.
Results are saved to results/<transactionId>.json automatically.
"""

import json
import pandas as pd
from fraud_graph import evaluate

df = pd.read_csv("microsoft_transactions.csv", low_memory=False)

# Pick 3 fraud + 2 legit transactions from different customers
fraud_sample = df[df["isFraud"] == 1].groupby("customerId").first().head(3).reset_index()
legit_sample = df[df["isFraud"] == 0].groupby("customerId").first().head(2).reset_index()
test_txns = pd.concat([fraud_sample, legit_sample], ignore_index=True)

print(f"Testing {len(test_txns)} transactions ({len(fraud_sample)} fraud, {len(legit_sample)} legit)")
print("=" * 80)

for i, row in test_txns.iterrows():
    txn = row.to_dict()
    # Remove the isFraud label from the input (the pipeline shouldn't see it)
    actual_fraud = txn.pop("isFraud", None)

    print(f"\n{'─' * 80}")
    print(f"[{i+1}/{len(test_txns)}] Transaction: {txn['transactionId']}")
    print(f"  Customer: {txn['customerId']}")
    print(f"  Amount: ${txn['amount']:.2f}")
    print(f"  Location: {txn.get('location', 'N/A')} | IP Country: {txn.get('ipCountry', 'N/A')}")
    print(f"  Device: {txn.get('deviceId', 'N/A')} ({txn.get('deviceType', 'N/A')})")
    print(f"  ACTUAL LABEL: {'FRAUD' if actual_fraud == 1 else 'LEGIT'}")
    print()

    result = evaluate(txn)

    nodes = result.get("nodes", [])
    for node in nodes:
        if node["id"] == "llm_agent":
            print(f"  DECISION: {node.get('decision')} → {node.get('action')}")
            print(f"  REASONING: {node.get('reasoning')}")
        else:
            print(f"  {node['name']:20s} risk={node.get('risk', 0):.2f}  label={node.get('label', '?'):8s}  {node.get('reason', '')[:80]}")

    # Check if result JSON was saved
    json_path = f"results/{txn['transactionId']}.json"
    print(f"  → Full analysis saved to {json_path}")

print(f"\n{'=' * 80}")
print("All tests complete. Check results/ folder for full JSON output.")
