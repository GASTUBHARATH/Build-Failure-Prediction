"""Command-line prediction for a single software build."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import FEATURE_COLUMNS


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "build_failure_model.joblib"


def predict_build(
    build_duration: float,
    code_changes: int,
    previous_failures: int,
    test_failures: int,
    dependency_changes: int,
    model_path: Path = MODEL_PATH,
) -> tuple[str, float | None]:
    """Predict build status and return failure probability when available."""
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run `python -m src.train_model` first.")
    model = joblib.load(model_path)
    values = pd.DataFrame(
        [[
            build_duration,
            code_changes,
            previous_failures,
            test_failures,
            dependency_changes,
        ]],
        columns=FEATURE_COLUMNS,
    )
    prediction = int(model.predict(values)[0])
    failure_probability = None
    if hasattr(model, "predict_proba"):
        failure_probability = float(model.predict_proba(values)[0][1])
    return ("Failure" if prediction == 1 else "Success"), failure_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether a software build will fail.")
    parser.add_argument("--build-duration", type=float, required=True)
    parser.add_argument("--code-changes", type=int, required=True)
    parser.add_argument("--previous-failures", type=int, required=True)
    parser.add_argument("--test-failures", type=int, required=True)
    parser.add_argument("--dependency-changes", type=int, required=True)
    args = parser.parse_args()
    status, probability = predict_build(**vars(args))
    print(f"Prediction: {status}")
    if probability is not None:
        print(f"Probability of failure: {probability:.1%}")


if __name__ == "__main__":
    main()