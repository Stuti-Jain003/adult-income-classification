"""
Streamlit web app - Adult Income Classification
------------------------------------------------
Interactive demo of six classification models trained on the UCI Adult
Income dataset. Upload a test CSV, pick a model, and view its evaluation
metrics, confusion matrix, and classification report on your data.

Run locally:
    streamlit run app.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="💼",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
    "SVM": "svm.joblib",
}

TARGET = "income"


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(filename: str):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def load_metrics_table() -> pd.DataFrame | None:
    path = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_meta() -> dict:
    path = os.path.join(MODEL_DIR, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown(
    "Upload your **test data** (CSV) and choose a model to evaluate it on."
)

uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

selected_model_name = st.sidebar.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys()),
    index=4,  # Random Forest default (best performer)
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: UCI Adult Income · Target: whether annual income > $50K"
)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("💼 Adult Income Classification")
st.markdown(
    "Predict whether a person earns **more than \\$50K/year** from census "
    "attributes, using six classical machine-learning classifiers. "
    "Upload a labelled test CSV (must contain the target column "
    f"`{TARGET}`) to see live evaluation metrics."
)

# Show the pre-computed comparison table up front.
metrics_table = load_metrics_table()
if metrics_table is not None:
    with st.expander("📊 Model comparison on the original hold-out test set", expanded=False):
        st.dataframe(
            metrics_table.set_index("Model").style.format("{:.4f}").highlight_max(
                axis=0, color="#c6efce"
            ),
            use_container_width=True,
        )

st.markdown("---")


# ----------------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------------
if uploaded is None:
    st.info(
        "👈 Upload a test CSV from the sidebar to begin. "
        "You can use the `test_data.csv` file included in the repository."
    )
    st.stop()

# Read uploaded data
try:
    data = pd.read_csv(uploaded)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

st.subheader("Uploaded data preview")
st.write(f"Shape: **{data.shape[0]} rows × {data.shape[1]} columns**")
st.dataframe(data.head(), use_container_width=True)

if TARGET not in data.columns:
    st.error(
        f"The uploaded CSV must contain the target column `{TARGET}`. "
        f"Columns found: {list(data.columns)}"
    )
    st.stop()

# Split features / target. Normalise target to 0/1 if needed.
X = data.drop(columns=[TARGET])
y_raw = data[TARGET]

if y_raw.dtype == object:
    y = (
        y_raw.astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .eq(">50K")
        .astype(int)
    )
else:
    y = y_raw.astype(int)

# Load selected model and predict
model = load_model(MODEL_FILES[selected_model_name])

try:
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
except Exception as exc:  # noqa: BLE001
    st.error(
        "The model could not run on this data. Make sure the columns match "
        f"the training features.\n\nDetails: {exc}"
    )
    st.stop()

st.markdown("---")
st.subheader(f"Results for: {selected_model_name}")

# --- Evaluation metrics ---
metrics = compute_metrics(y, y_pred, y_score)

cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

# --- Confusion matrix + classification report side by side ---
left, right = st.columns(2)

with left:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("#### Classification Report")
    report = classification_report(
        y, y_pred, target_names=["<=50K", ">50K"], output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

st.success(
    f"Evaluated **{selected_model_name}** on **{len(y)}** rows from your uploaded file."
)
