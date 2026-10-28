"""Train 5 classifiers on a 600-row, 15-feature sample of the
Social Media Screen-time & Mental Health 2026 dataset.

Usage:
    python train.py --source full_dataset.csv [--include-anxiety]

Outputs: dataset_600.csv, train_data.csv, test_data.csv, model/*.joblib,
         model/metrics.csv, model/features.json
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

TARGET = "wellbeing_band"
SEED = 42
N_SAMPLES = 600
N_FEATURES = 15

# Features ranked by mutual information with the target on the full 7,000 rows.
# anxiety_score_0to27 is the top-ranked feature but effectively defines the
# target (see README), so it is excluded unless --include-anxiety is passed.
RANKED = [
    "anxiety_score_0to27", "daily_screen_hours", "low_mood_score_0to27",
    "daily_notifications", "fomo_1to10", "avg_sleep_hours",
    "life_satisfaction_1to10", "self_esteem_1to10", "loneliness_1to10",
    "Age", "social_comparison_1to10", "night_time_use", "occupation",
    "most_used_platform", "Gender", "primary_purpose",
]
CATEGORICAL = {"Gender", "occupation", "most_used_platform", "night_time_use",
               "primary_purpose"}


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=SEED),
    }


def build_pipeline(features, model):
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro",
                             labels=pipe.classes_),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--include-anxiety", action="store_true")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "model").mkdir(exist_ok=True)

    ranked = RANKED if args.include_anxiety else RANKED[1:]
    features = ranked[:N_FEATURES]

    full = pd.read_csv(args.source)
    df = full[features + [TARGET]]
    # Stratified 600-row sample (keeps the original class proportions)
    sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df[TARGET],
                                 random_state=SEED)
    sample = sample.reset_index(drop=True)
    sample.to_csv(out / "dataset_600.csv", index=False)

    train, test = train_test_split(sample, test_size=0.2, stratify=sample[TARGET],
                                   random_state=SEED)
    train.to_csv(out / "train_data.csv", index=False)
    test.to_csv(out / "test_data.csv", index=False)
    Xtr, ytr = train[features], train[TARGET]
    Xte, yte = test[features], test[TARGET]

    rows = {}
    for name, model in build_models().items():
        pipe = build_pipeline(features, model).fit(Xtr, ytr)
        rows[name] = evaluate(pipe, Xte, yte)
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, out / "model" / fname)

    metrics = pd.DataFrame(rows).T.round(4)
    metrics.index.name = "ML Model Name"
    metrics.to_csv(out / "model" / "metrics.csv")
    json.dump({"features": features, "target": TARGET,
               "classes": sorted(sample[TARGET].unique())},
              open(out / "model" / "features.json", "w"), indent=2)
    print(f"features ({len(features)}): {features}")
    print(f"train={len(train)} test={len(test)}")
    print(test[TARGET].value_counts().to_dict())
    print(metrics.to_string())


if __name__ == "__main__":
    main()
