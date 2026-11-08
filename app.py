"""Streamlit dashboard for the saved build-failure prediction model."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "build_failure_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "evaluation_results.json"
FEATURE_COLUMNS = [
    "build_duration",
    "code_changes",
    "previous_failures",
    "test_failures",
    "dependency_changes",
]


@st.cache_resource
def load_model():
    """Load the already-trained model once per Streamlit session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model was not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    """Load the saved evaluation results."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Evaluation file was not found: {METRICS_PATH}")
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    if "best_model" not in metrics or "models" not in metrics:
        raise ValueError("The evaluation file does not contain model results.")
    if "confusion_matrix" not in metrics:
        raise ValueError("The evaluation file does not contain a confusion matrix.")
    return metrics


def display_evaluation(metrics: dict) -> None:
    """Display saved model metrics and the saved confusion matrix."""
    selected_name = metrics["best_model"]
    selected_metrics = metrics["models"].get(selected_name)
    if selected_metrics is None:
        raise ValueError(f"Metrics for selected model '{selected_name}' are missing.")

    st.subheader("Model evaluation")
    st.caption(f"Selected model: **{selected_name}**")
    metric_columns = st.columns(4)
    metric_values = [
        ("Accuracy", selected_metrics["accuracy"]),
        ("Precision", selected_metrics["precision"]),
        ("Recall", selected_metrics["recall"]),
        ("F1-score", selected_metrics["f1_score"]),
    ]
    for column, (label, value) in zip(metric_columns, metric_values):
        column.metric(label, f"{value:.1%}")

    left, right = st.columns(2)
    with left:
        st.write("Comparison of saved model results")
        comparison_rows = []
        for model_name, model_metrics in metrics["models"].items():
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Accuracy": model_metrics["accuracy"],
                    "Precision": model_metrics["precision"],
                    "Recall": model_metrics["recall"],
                    "F1-score": model_metrics["f1_score"],
                }
            )
        comparison = pd.DataFrame(comparison_rows).set_index("Model")
        st.dataframe(comparison.style.format("{:.1%}"), use_container_width=True)

    with right:
        st.write("Confusion matrix")
        matrix = metrics["confusion_matrix"]
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError("The saved confusion matrix must be 2x2.")
        figure, axis = plt.subplots(figsize=(4.5, 3.5))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], labels=["Success", "Failure"])
        axis.set_yticks([0, 1], labels=["Success", "Failure"])
        axis.set_xlabel("Predicted status")
        axis.set_ylabel("Actual status")
        axis.set_title("Confusion matrix")
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    matrix[row_index][column_index],
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index][column_index] > max(map(max, matrix)) / 2 else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        st.pyplot(figure, clear_figure=True)


def display_prediction(model) -> None:
    """Collect build details and display a prediction."""
    st.subheader("Predict a new build")
    st.write("Enter the details of a build to estimate its outcome.")
    with st.form("prediction_form"):
        first, second = st.columns(2)
        build_duration = first.number_input(
            "Build duration (minutes)", min_value=0.0, value=18.0, step=1.0
        )
        code_changes = first.number_input("Code changes", min_value=0, value=25, step=1)
        previous_failures = second.number_input(
            "Previous failures", min_value=0, value=1, step=1
        )
        test_failures = second.number_input("Test failures", min_value=0, value=0, step=1)
        dependency_changes = second.number_input(
            "Dependency changes", min_value=0, value=1, step=1
        )
        submitted = st.form_submit_button("Predict build status", type="primary")

    if not submitted:
        return

    input_data = pd.DataFrame(
        [
            [
                build_duration,
                code_changes,
                previous_failures,
                test_failures,
                dependency_changes,
            ]
        ],
        columns=FEATURE_COLUMNS,
    )
    try:
        prediction = int(model.predict(input_data)[0])
        status = "Failure" if prediction == 1 else "Success"
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1])
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        st.error(f"Prediction could not be completed: {error}")
        return

    if status == "Failure":
        st.error("Prediction: Failure")
        explanation = (
            "The model estimates that this build is more likely to fail. "
            "Review the build inputs, tests, and dependencies before running it."
        )
    else:
        st.success("Prediction: Success")
        explanation = (
            "The model estimates that this build is more likely to succeed. "
            "This is an estimate, not a guarantee."
        )
    st.info(explanation)
    if probability is not None:
        st.metric("Probability of failure", f"{probability:.1%}")


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Build-Failure Prediction",
        page_icon="🔧",
        layout="wide",
    )
    st.title("Build-Failure Prediction Using Machine Learning")
    st.markdown(
        "Use historical build information to estimate whether a new software build "
        "will succeed or fail."
    )
    st.info("The dataset used for this assignment is simulated educational data, not production data.")

    try:
        model = load_model()
        metrics = load_metrics()
        display_evaluation(metrics)
        display_prediction(model)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Dashboard could not load its saved files: {error}")


if __name__ == "__main__":
    main()
