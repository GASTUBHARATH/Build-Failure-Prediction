"""Data loading and leakage-safe preprocessing for build-failure prediction."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "build_data.csv"
FEATURE_COLUMNS = [
    "build_duration",
    "code_changes",
    "previous_failures",
    "test_failures",
    "dependency_changes",
]
TARGET_COLUMN = "build_status"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the simulated build dataset."""
    data = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("string").str.strip().str.title()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].where(
        data[TARGET_COLUMN].isin(["Success", "Failure"])
    )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("build_status must contain only Success or Failure labels.")
    return data


def prepare_data(
    path: Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load data, encode the target, and split before fitting preprocessing."""
    data = load_dataset(path)
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].map({"Failure": 1, "Success": 0}).astype(int)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def create_preprocessor() -> ColumnTransformer:
    """Create a numeric imputation and scaling transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, FEATURE_COLUMNS)],
        remainder="drop",
    )