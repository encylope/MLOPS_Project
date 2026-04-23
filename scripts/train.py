"""
Model Training Script — Credit Card Fraud Detection
Run via: python scripts/train.py
or via:  mlflow run . -e train

MLflow tracks:
  - Parameters: model type, hyperparameters, SMOTE settings, threshold
  - Metrics: precision, recall, F1, AUC-ROC, AUC-PR, inference latency
  - Artifacts: model, confusion matrix, ROC curve, feature importance plot
  - Tags: dataset version (DVC hash), git commit
"""

import json
import logging
import os
import subprocess
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR = os.getenv("FEATURES_DIR", "data/features")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "credit-card-fraud-detection"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_dvc_hash() -> str:
    try:
        result = subprocess.check_output(
            ["dvc", "status", "--json"], text=True
        ).strip()
        return result[:16] if result else "unknown"
    except Exception:
        return "unknown"


def load_splits():
    """Load train/val/test splits from DVC-tracked feature store."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(FEATURES_DIR, f"{name}.csv")
        df = pd.read_csv(path)
        X = df.drop("Class", axis=1)
        y = df["Class"]
        splits[name] = (X, y)
        logger.info(f"{name}: {len(df)} rows | fraud={y.sum()} ({y.mean():.4%})")
    return splits


def plot_confusion_matrix(cm, labels=["Legit", "Fraud"], path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_roc_curve(fpr, tpr, roc_auc, path="roc_curve.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def train_and_log(model_type: str = "xgboost"):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    splits = load_splits()
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    with mlflow.start_run(run_name=f"{model_type}-{get_git_commit()}") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "git_commit": get_git_commit(),
            "dvc_hash": get_dvc_hash(),
            "dataset": "kaggle-creditcardfraud",
            "model_type": model_type,
            "engineer": os.getenv("USER", "mlops-student"),
        })

        # ── SMOTE oversampling ─────────────────────────────────────────────────
        smote_ratio = 0.1
        mlflow.log_param("smote_sampling_strategy", smote_ratio)
        mlflow.log_param("smote_random_state", 42)

        smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {X_resampled.shape[0]} rows | fraud={y_resampled.sum()}")

        mlflow.log_metric("train_rows_after_smote", len(X_resampled))

        # ── Model selection + hyperparams ──────────────────────────────────────
        if model_type == "xgboost":
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "scale_pos_weight": 10,
                "random_state": 42,
                "eval_metric": "aucpr",
                "use_label_encoder": False,
            }
            model = XGBClassifier(**params)
        else:
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        mlflow.log_params(params)

        # ── Training ───────────────────────────────────────────────────────────
        logger.info("Training model...")
        start = time.time()
        model.fit(X_resampled, y_resampled)
        train_time = time.time() - start
        mlflow.log_metric("training_time_seconds", round(train_time, 2))

        # ── Threshold tuning on validation set ────────────────────────────────
        val_probas = model.predict_proba(X_val)[:, 1]
        best_f1, best_threshold = 0.0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (val_probas >= thresh).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, thresh

        mlflow.log_param("decision_threshold", round(best_threshold, 2))
        logger.info(f"Best threshold: {best_threshold:.2f} (val F1={best_f1:.4f})")

        # ── Test evaluation ────────────────────────────────────────────────────
        test_probas = model.predict_proba(X_test)[:, 1]
        y_pred_test = (test_probas >= best_threshold).astype(int)

        roc_auc = roc_auc_score(y_test, test_probas)
        auc_pr = average_precision_score(y_test, test_probas)
        report = classification_report(y_test, y_pred_test, output_dict=True)
        fraud_metrics = report.get("1", {})

        metrics = {
            "test_roc_auc": round(roc_auc, 6),
            "test_auc_pr": round(auc_pr, 6),
            "test_f1_fraud": round(fraud_metrics.get("f1-score", 0), 6),
            "test_precision_fraud": round(fraud_metrics.get("precision", 0), 6),
            "test_recall_fraud": round(fraud_metrics.get("recall", 0), 6),
            "test_accuracy": round(report.get("accuracy", 0), 6),
        }
        mlflow.log_metrics(metrics)

        # Inference latency (avg over 1000 single predictions)
        sample = X_test.iloc[:1]
        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            model.predict_proba(sample)
            times.append((time.perf_counter() - t0) * 1000)
        avg_latency = round(float(np.mean(times)), 3)
        mlflow.log_metric("avg_inference_latency_ms", avg_latency)
        logger.info(f"Avg inference latency: {avg_latency}ms")

        logger.info(f"Metrics: {metrics}")

        # ── Artifacts ─────────────────────────────────────────────────────────
        os.makedirs("/tmp/mlflow_artifacts", exist_ok=True)

        cm = confusion_matrix(y_test, y_pred_test)
        cm_path = plot_confusion_matrix(cm, path="/tmp/mlflow_artifacts/confusion_matrix.png")
        mlflow.log_artifact(cm_path)

        fpr, tpr, _ = roc_curve(y_test, test_probas)
        roc_path = plot_roc_curve(fpr, tpr, roc_auc, path="/tmp/mlflow_artifacts/roc_curve.png")
        mlflow.log_artifact(roc_path)

        # Feature importance (XGBoost)
        if model_type == "xgboost" and hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=X_train.columns)
            top_features = fi.nlargest(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            top_features.sort_values().plot(kind="barh", ax=ax)
            ax.set_title("Top 15 Feature Importances")
            fi_path = "/tmp/mlflow_artifacts/feature_importance.png"
            plt.tight_layout()
            plt.savefig(fi_path, dpi=100)
            plt.close()
            mlflow.log_artifact(fi_path)

        # Save full classification report
        report_path = "/tmp/mlflow_artifacts/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # ── Log model ─────────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud-detector",
            input_example=X_test.iloc[:1],
        )
        # Save metrics locally for DVC tracking
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Run ID: {run.info.run_id}")

        return run.info.run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="xgboost", choices=["xgboost", "random_forest"])
    args = parser.parse_args()

    run_id = train_and_log(model_type=args.model_type)
    print(f"\nMLflow Run ID: {run_id}")
    print(f"View at: {MLFLOW_URI}/#/experiments")
