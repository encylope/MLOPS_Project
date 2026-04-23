"""Stage 6: Model Evaluation"""
import json
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, roc_curve
)

FEATURES_DIR = "data/features"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def evaluate():
    os.makedirs("metrics", exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    df = pd.read_csv(os.path.join(FEATURES_DIR, "test.csv"))
    X_test = df.drop("Class", axis=1)
    y_test = df["Class"]

    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='fraud-detector'")
    if not versions:
        print("No model found in registry. Run train.py first.")
        return

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    model = mlflow.sklearn.load_model(f"models:/fraud-detector/{latest.version}")

    probas = model.predict_proba(X_test)[:, 1]
    y_pred = (probas >= 0.5).astype(int)

    # ROC curve — save for DVC plots
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df.to_csv("metrics/roc_curve.csv", index=False)

    metrics = {
        "test_roc_auc": round(float(roc_auc_score(y_test, probas)), 6),
        "test_auc_pr": round(float(average_precision_score(y_test, probas)), 6),
        "test_f1_fraud": round(float(f1_score(y_test, y_pred)), 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ROC-AUC: {metrics['test_roc_auc']}")
    print(f"F1 Fraud: {metrics['test_f1_fraud']}")
    print("Saved metrics/test_metrics.json and metrics/roc_curve.csv")

if __name__ == "__main__":
    evaluate()