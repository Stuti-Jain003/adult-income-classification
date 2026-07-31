"""
train_models.py
----------------
Trains six classification models on the UCI Adult Income dataset and saves
each fitted pipeline (preprocessing + estimator) to model/*.joblib.

Models:
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors
    4. Naive Bayes (Gaussian)
    5. Random Forest (ensemble)
    6. Support Vector Machine (SVM)

Also writes:
    - model/metrics.csv        : comparison table of all evaluation metrics
    - test_data.csv            : held-out test split (used by the Streamlit app)

Run from the project root:
    python model/train_models.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
RAW_TRAIN = os.path.join(PROJECT_ROOT, "data", "adult.data")
MODEL_DIR = HERE
TEST_CSV = os.path.join(PROJECT_ROOT, "test_data.csv")
METRICS_CSV = os.path.join(MODEL_DIR, "metrics.csv")

RANDOM_STATE = 42

# Column names for the raw Adult dataset (no header in the UCI file)
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

TARGET = "income"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw Adult data and perform basic cleaning."""
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    # Drop rows with missing values (only ~7% of rows have them)
    df = df.dropna().reset_index(drop=True)

    # Normalise the target label ("<=50K." / ">50K." variants -> clean strings)
    df[TARGET] = df[TARGET].str.replace(".", "", regex=False).str.strip()
    # Binary target: 1 if income > 50K else 0
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics."""
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return preprocessor


def get_models() -> dict:
    """Return the six classifiers keyed by a short slug."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "svm": SVC(probability=True, random_state=RANDOM_STATE),
    }


DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
    "svm": "SVM",
}


def evaluate(model, X_test, y_test) -> dict:
    """Compute the six required evaluation metrics."""
    y_pred = model.predict(X_test)

    # Probability / score for AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main() -> None:
    print("Loading and cleaning data ...")
    df = load_and_clean(RAW_TRAIN)
    print(f"  clean shape: {df.shape}")
    print(f"  features: {df.shape[1] - 1}, instances: {df.shape[0]}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train: {X_train.shape[0]} rows | test: {X_test.shape[0]} rows")

    # Persist the raw (un-preprocessed) test split for the Streamlit app.
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(TEST_CSV, index=False)
    print(f"  wrote test data -> {TEST_CSV}")

    preprocessor = build_preprocessor(X_train)
    models = get_models()

    rows = []
    for slug, estimator in models.items():
        print(f"Training: {DISPLAY_NAMES[slug]} ...")
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", estimator)]
        )
        pipe.fit(X_train, y_train)

        metrics = evaluate(pipe, X_test, y_test)
        metrics_row = {"Model": DISPLAY_NAMES[slug], **metrics}
        rows.append(metrics_row)

        out_path = os.path.join(MODEL_DIR, f"{slug}.joblib")
        joblib.dump(pipe, out_path)
        print(
            f"  saved -> {out_path}  "
            f"(Acc={metrics['Accuracy']:.4f}, AUC={metrics['AUC']:.4f})"
        )

    metrics_df = pd.DataFrame(rows).round(4)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics comparison written -> {METRICS_CSV}")
    print(metrics_df.to_string(index=False))

    # Save the feature column order so the app can validate uploads.
    meta = {
        "target": TARGET,
        "feature_columns": X.columns.tolist(),
        "display_names": DISPLAY_NAMES,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved model/meta.json")


if __name__ == "__main__":
    main()
