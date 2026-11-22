"""Train and save breast-cancer models for the explainable AI dashboard."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = PROJECT_ROOT / "models" / "breast_cancer.joblib"


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Return standard classification metrics for one model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def load_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
    list[str],
]:
    """Load the built-in Breast Cancer dataset and preserve readable names."""
    dataset = load_breast_cancer()
    frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target = pd.Series(dataset.target, name="target")
    feature_names = [str(name) for name in dataset.feature_names]
    class_names = [str(name) for name in dataset.target_names]
    return frame, target, feature_names, class_names


def train_and_save() -> tuple[dict[str, Any], pd.DataFrame]:
    """Train both models, select by ROC-AUC, and save one joblib artifact."""
    X, y, feature_names, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    random_forest.fit(X_train, y_train)
    gradient_boosting.fit(X_train, y_train)

    rf_metrics = evaluate_model(random_forest, X_test, y_test)
    gb_metrics = evaluate_model(gradient_boosting, X_test, y_test)

    metrics = {
        "Random Forest": rf_metrics,
        "Gradient Boosting": gb_metrics,
    }
    best_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    # Random Forest is the app default because it is usually stable with TreeExplainer.
    selected_model = random_forest if best_name == "Random Forest" else gradient_boosting

    artifact = {
        "model": selected_model,
        "feature_names": feature_names,
        "class_names": class_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact, X_test


def print_metrics_table(metrics: dict[str, dict[str, float]]) -> None:
    """Print clean model metric rows for terminal verification."""
    table = pd.DataFrame(metrics).T
    table = table[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))


def load_artifacts() -> dict[str, Any]:
    """Load saved artifacts, training them automatically when absent."""
    if not ARTIFACT_PATH.exists():
        artifact, _ = train_and_save()
        return artifact
    return joblib.load(ARTIFACT_PATH)


def main() -> None:
    """Train models and print their test-set metrics."""
    artifact, _ = train_and_save()
    print("Breast Cancer Model Comparison")
    print_metrics_table(artifact["metrics"])
    print(f"\nArtifact saved to: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
