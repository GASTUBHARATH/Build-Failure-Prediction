"""Train, compare, and save build-failure classification models."""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.preprocess import create_preprocessor, prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "build_failure_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "evaluation_results.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "models" / "confusion_matrix.csv"


def build_models() -> Dict[str, Pipeline]:
    """Return the two required models with a shared preprocessing pipeline."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250, random_state=42, class_weight="balanced"
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[dict, list]:
    """Return the requested metrics and a 2x2 confusion matrix."""
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Success", "Failure"],
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics, matrix.tolist()


def train_and_save() -> dict:
    """Train both models, select by F1-score, and persist the best pipeline."""
    x_train, x_test, y_train, y_test = prepare_data()
    results = {}
    matrices = {}

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics, matrix = evaluate_model(model, x_test, y_test)
        results[name] = metrics
        matrices[name] = matrix
        print(
            f"{name}: accuracy={metrics['accuracy']:.3f}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1_score']:.3f}"
        )

    best_name = max(results, key=lambda name: (results[name]["f1_score"], results[name]["recall"]))
    best_model = build_models()[best_name]
    best_model.fit(x_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    best_matrix = matrices[best_name]
    pd.DataFrame(
        best_matrix,
        index=["Actual Success", "Actual Failure"],
        columns=["Predicted Success", "Predicted Failure"],
    ).to_csv(CONFUSION_MATRIX_PATH)
    output = {
        "dataset": "Simulated educational build data (not production data)",
        "test_records": int(len(x_test)),
        "best_model": best_name,
        "models": results,
        "confusion_matrix": best_matrix,
        "labels": ["Success", "Failure"],
    }
    METRICS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Selected model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation results: {METRICS_PATH}")
    return output


if __name__ == "__main__":
    train_and_save()