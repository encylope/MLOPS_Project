"""Stage 4: Train/Val/Test Split"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES_DIR = "data/features"

def split():
    df = pd.read_csv(os.path.join(FEATURES_DIR, "features.csv"))
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    for name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        split_df = X_split.copy()
        split_df["Class"] = y_split.values
        path = os.path.join(FEATURES_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"Saved {name}: {len(split_df)} rows, fraud={y_split.sum()}")

if __name__ == "__main__":
    split()