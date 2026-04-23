"""Stage 3: Feature Engineering"""
import json
import os
import pandas as pd

PROCESSED_DIR = "data/processed"
FEATURES_DIR = "data/features"

def feature_engineering():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "creditcard_processed.csv"))
    feature_cols = [c for c in df.columns if c != "Class"]

    baseline = {}
    for col in feature_cols:
        baseline[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "p25": float(df[col].quantile(0.25)),
            "p75": float(df[col].quantile(0.75)),
        }

    with open(os.path.join(FEATURES_DIR, "feature_baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    df.to_csv(os.path.join(FEATURES_DIR, "features.csv"), index=False)
    print(f"Feature engineering done. Features: {len(feature_cols)}")

if __name__ == "__main__":
    feature_engineering()