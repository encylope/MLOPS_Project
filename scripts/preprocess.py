"""Stage 2: Preprocessing"""
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"

def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH)

    amount_cap = df["Amount"].quantile(0.999)
    df = df[df["Amount"] <= amount_cap].copy()

    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": ["Amount", "Time"]
    }
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f)

    df.to_csv(os.path.join(PROCESSED_DIR, "creditcard_processed.csv"), index=False)
    print(f"Preprocessing done. Rows: {len(df)}")

if __name__ == "__main__":
    preprocess()