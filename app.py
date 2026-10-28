import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Wellbeing Band Classifier", layout="wide")
st.title("Social Media Mental Health Risk - Wellbeing Band Classifier")
st.caption("Upload a test CSV, pick a trained model, and see its evaluation "
           "metrics. Models are loaded from disk and are NOT retrained.")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def load_config():
    return json.load(open(MODEL_DIR / "features.json"))


cfg = load_config()
features, target = cfg["features"], cfg["target"]

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown(f"**Required columns:** {len(features)} features + `{target}`")
    with st.expander("Feature list"):
        st.write(features)
    st.download_button("Download sample test_data.csv",
                       (BASE / "test_data.csv").read_bytes(),
                       file_name="test_data.csv", mime="text/csv")

if uploaded is None:
    st.info("Upload a CSV in the sidebar (or download the sample test file "
            "there and upload it) to evaluate a model.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"The uploaded CSV is missing required columns: {missing}")
    st.stop()

df = df.dropna(subset=[target])
X, y = df[features], df[target]
st.subheader("Uploaded data preview")
st.write(f"{len(df)} rows")
st.dataframe(df.head(10), width="stretch")

pipe = load_model(model_name)
unseen = sorted(set(y) - set(pipe.classes_))
if unseen:
    st.error(f"Target contains labels the model was not trained on: {unseen}")
    st.stop()

pred = pipe.predict(X)
proba = pipe.predict_proba(X)


def safe_auc():
    try:
        return roc_auc_score(y, proba, multi_class="ovr", average="macro",
                             labels=pipe.classes_)
    except ValueError:  # a class is absent from the uploaded file
        return float("nan")


metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": safe_auc(),
    "Precision": precision_score(y, pred, average="macro", zero_division=0),
    "Recall": recall_score(y, pred, average="macro", zero_division=0),
    "F1": f1_score(y, pred, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")
st.caption("AUC is one-vs-rest macro-averaged; Precision, Recall and F1 are "
           "macro-averaged across the three classes.")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, labels=list(pipe.classes_), cmap="Blues", ax=ax,
        colorbar=False)
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, pred, labels=list(pipe.classes_),
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

st.subheader("All models on this uploaded data")
rows = {}
for name in MODEL_FILES:
    p = load_model(name)
    pr, pb = p.predict(X), p.predict_proba(X)
    try:
        auc = roc_auc_score(y, pb, multi_class="ovr", average="macro",
                            labels=p.classes_)
    except ValueError:
        auc = float("nan")
    rows[name] = {
        "Accuracy": accuracy_score(y, pr), "AUC": auc,
        "Precision": precision_score(y, pr, average="macro", zero_division=0),
        "Recall": recall_score(y, pr, average="macro", zero_division=0),
        "F1": f1_score(y, pr, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pr),
    }
cmp_df = pd.DataFrame(rows).T.round(4)
cmp_df.index.name = "ML Model Name"
st.dataframe(cmp_df, width="stretch")
