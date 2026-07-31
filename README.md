# Adult Income Classification — ML Assignment 2

M.Tech (AIML / DSE) · Machine Learning · BITS Pilani WILP

An end-to-end machine-learning project that trains **six classification models**
on the UCI *Adult Income* dataset and serves them through an interactive
**Streamlit** web app. Upload a labelled test CSV, pick a model, and inspect its
evaluation metrics, confusion matrix, and classification report — or compare all
six models side by side.

---

## a. Problem Statement

Given demographic and employment attributes from the 1994 US Census, predict
whether an individual earns **more than \$50,000 per year** (`>50K`) or
**\$50,000 or less** (`<=50K`). This is a supervised **binary classification**
problem. Accurate income prediction is useful for socio-economic studies,
targeted policy design, and credit / marketing segmentation.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository — [Adult Data Set](https://archive.ics.uci.edu/dataset/2/adult) |
| Task | Binary classification |
| Target | `income` → `1` if `>50K`, else `0` |
| Instances (after cleaning) | **30,162** (rows with missing values dropped) |
| Features | **14** (6 numeric + 8 categorical) |
| Class balance | ~24.9% `>50K`, ~75.1% `<=50K` (imbalanced) |

**Features used**

- *Numeric:* `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`
- *Categorical:* `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`

**Preprocessing** (implemented inside each model's scikit-learn `Pipeline`, so it
is applied identically at train and inference time):

- Rows with `?` / missing values are dropped.
- Numeric features are standardised with `StandardScaler`.
- Categorical features are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`.

The data is split **80% train / 20% test** with a fixed `random_state=42` and
stratification on the target. The 20% hold-out split is saved as
[`test_data.csv`](test_data.csv) and is what the Streamlit app evaluates on.

> Minimum requirements satisfied: **14 features ≥ 12** and **30,162 instances ≥ 500**.

---

## c. GitHub Repository Link

> **Repo:** `https://github.com/<your-username>/adult-income-classification`
>
> _(Replace `<your-username>` with your GitHub handle after pushing — see the
> "How to run / deploy" section below.)_

### Repository structure

```
adult-income-classification/
├── app.py                     # Streamlit web application
├── requirements.txt           # Pinned dependencies (deployment-ready)
├── runtime.txt                # Pins Python 3.11 on Streamlit Cloud
├── README.md                  # This file
├── test_data.csv              # 20% hold-out test split (used by the app)
├── .streamlit/
│   └── config.toml            # App theme / upload-size config
└── model/
    ├── train_models.py        # Trains + evaluates + saves all 6 models
    ├── metrics.csv            # Comparison table (generated)
    ├── meta.json              # Feature order + target (generated)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── svm.joblib
```

---

## d. Models Used

Six classifiers are trained on the **same** dataset and preprocessing pipeline:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (ensemble)
6. Support Vector Machine (SVM)

### Comparison Table (on the 20% hold-out test set, 6,033 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8475 | 0.9022 | 0.7354 | 0.6052 | 0.6640 | 0.5711 |
| Decision Tree | 0.8530 | 0.8959 | 0.7681 | 0.5866 | 0.6652 | 0.5817 |
| kNN | 0.8384 | 0.8911 | 0.7007 | 0.6125 | 0.6536 | 0.5510 |
| Naive Bayes | 0.5826 | 0.8018 | 0.3678 | 0.9414 | 0.5290 | 0.3643 |
| **Random Forest (Ensemble)** | **0.8556** | **0.9146** | **0.7850** | 0.5786 | **0.6662** | **0.5877** |
| SVM | 0.8498 | 0.8983 | 0.7500 | 0.5952 | 0.6637 | 0.5750 |

*(All metrics are reproducible via `python model/train_models.py`.)*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong, well-calibrated baseline. Second-highest AUC (0.902) despite a linear decision boundary, showing the classes are largely linearly separable after one-hot encoding + scaling. Balanced precision/recall. |
| **Decision Tree** | Good accuracy (0.853) with `max_depth=10` to curb overfitting. Higher precision than recall — it is conservative about predicting the minority `>50K` class. Slightly lower AUC than the linear/ensemble models. |
| **kNN** | Weakest of the tree/linear family (accuracy 0.838, MCC 0.551). Distance-based methods suffer in the high-dimensional one-hot-encoded space ("curse of dimensionality"), and it is the slowest at inference. |
| **Naive Bayes** | Clear outlier. Its feature-independence assumption is badly violated by correlated + one-hot features, collapsing accuracy to 0.583. However, it achieves the **highest recall (0.941)** — it flags almost every high earner, at the cost of many false positives (low precision 0.368). Useful only when recall matters far more than precision. |
| **Random Forest (Ensemble)** | **Best overall.** Top Accuracy (0.856), AUC (0.915), Precision (0.785), F1 (0.666), and MCC (0.588). Bagging many de-correlated trees handles the mixed feature types and non-linear interactions best while resisting overfitting. |
| **SVM** | Very competitive (accuracy 0.850, AUC 0.898), close behind Random Forest and Logistic Regression. The RBF kernel captures non-linearity well, but it is the most expensive to train and offers no accuracy gain over the ensemble here. |
| **Overall Winner for this dataset** | 🏆 **Random Forest (Ensemble)** — it leads on 5 of the 6 metrics (Accuracy, AUC, Precision, F1, MCC), making it the most reliable choice for this imbalanced, mixed-type dataset. |

---

## Streamlit App Features

The deployed app (`app.py`) implements every required feature:

- **CSV upload** — upload your own labelled test CSV (must contain the `income` target column). The repo's `test_data.csv` can be used directly.
- **Model selection dropdown** — switch between all six trained models.
- **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, and MCC shown as live metric cards for the selected model.
- **Confusion matrix + classification report** — rendered side by side for the selected model.
- **Model comparison** — an expandable panel shows the full six-model comparison table with the best value in each column highlighted.

---

## How to Run Locally

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/adult-income-classification.git
cd adult-income-classification

# 2. (Optional) create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) retrain the models from scratch
#    Requires the raw UCI file at data/adult.data
python model/train_models.py

# 5. Launch the app
streamlit run app.py
```

Then open the CSV uploader, select `test_data.csv`, and choose a model.

---

## How to Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, main file `app.py`.
4. Click **Deploy**. The pinned `requirements.txt` + `runtime.txt` install automatically.
5. Once live, upload `test_data.csv` in the app to see results.

> **Live app:** `https://<your-app-name>.streamlit.app`

---

## Reproducibility

- Fixed `random_state=42` throughout (split + models) → identical metrics on every run.
- Preprocessing lives inside each saved `Pipeline`, so the app applies exactly the
  same transformations that were used during training.
- Dependency versions are pinned in `requirements.txt` and Python is pinned via
  `runtime.txt` to guarantee the `*.joblib` artifacts load correctly on the cloud.
