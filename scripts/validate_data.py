"""Stage 1: Data Validation"""
import json
import os
import pandas as pd

RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"

def validate():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH)

    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) < 1000:
        raise ValueError(f"Too few rows: {len(df)}")

    report = {
        "rows": len(df),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "fraud_rate": float(df["Class"].mean()),
        "status": "valid"
    }

    with open(os.path.join(PROCESSED_DIR, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation passed. Rows: {len(df)}, Fraud rate: {report['fraud_rate']:.4%}")

if __name__ == "__main__":
    validate()