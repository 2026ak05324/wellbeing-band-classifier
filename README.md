# Social Media Mental Health Risk Prediction - Wellbeing Band Classification

**GitHub Repository:** GITHUB_URL_PLACEHOLDER
**Live Streamlit App:** STREAMLIT_URL_PLACEHOLDER

## a. Problem Statement

Heavy social media and screen-time use is linked to anxiety, low mood, poor sleep and low self-esteem. The goal of this project is to build and compare five classical machine learning classifiers that predict a person's **wellbeing band** (`At-risk`, `Moderate` or `Good`) from 15 behavioural, psychological and demographic features. This is a **3-class classification** problem. An interactive Streamlit app lets a user upload a test CSV, choose a trained model and inspect its evaluation metrics and confusion matrix, without retraining.

## b. Dataset Description

- **Source:** Kaggle - *Social Media Screen-time & Mental Health 2026* (`social_media_screentime_mental_health_2026.csv`), 7,000 rows and 25 columns in the original file.
- **Target:** `wellbeing_band` with 3 classes: `At-risk`, `Moderate`, `Good`.
- **Sampling (assignment limit: 600 rows):** a stratified random sample of **600 rows** (seed 42), preserving the original class proportions: Moderate 323, Good 194, At-risk 83.
- **Features (assignment limit: 15):** the 15 features below, chosen by ranking all candidate features by mutual information with the target on the full 7,000 rows (identifier `participant_id` excluded) and keeping the top 15.

| Type | Features |
|---|---|
| Numeric (10) | daily_screen_hours, low_mood_score_0to27, daily_notifications, fomo_1to10, avg_sleep_hours, life_satisfaction_1to10, self_esteem_1to10, loneliness_1to10, Age, social_comparison_1to10 |
| Categorical (5) | night_time_use, occupation, most_used_platform, Gender, primary_purpose |

- **Excluded feature - target leakage:** `anxiety_score_0to27` is the single strongest feature, but the class bands follow it almost exactly (Good: scores 0-9, Moderate: about 10-16, At-risk: about 17 and above). Including it lets a decision tree reach 100% on every metric, which says nothing about real predictive ability. It is therefore **excluded by default**. Running `python train.py --source <full_csv> --include-anxiety` reproduces the version with it.
- **Preprocessing (inside each saved pipeline):** median imputation + standard scaling for numeric features (12 missing `avg_sleep_hours` values); most-frequent imputation + one-hot encoding for categorical features (7 missing `Gender` values). Because preprocessing is saved with each model, the app accepts the raw CSV.
- **Train/test split:** stratified 80/20, giving 480 training rows and **120 test rows** (Moderate 65, Good 39, At-risk 16), saved as `test_data.csv`.
- **Metric averaging:** Precision, Recall and F1 are macro-averaged over the three classes; AUC is one-vs-rest, macro-averaged, computed from predicted probabilities; MCC is the multiclass version.

## c. GitHub Repository Link

GITHUB_URL_PLACEHOLDER

```
project-folder/
|-- app.py                  Streamlit app
|-- train.py                data sampling, training, evaluation
|-- requirements.txt
|-- README.md
|-- test_data.csv           120-row test set (15 features + target)
|-- train_data.csv          480-row training set
|-- dataset_600.csv         the 600-row sample
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
|   |-- features.json
|   `-- metrics.csv
```

## d. Models Used

All five models are trained on the same 480-row training set and evaluated on the same 120-row test set: Logistic Regression (max_iter=2000), Decision Tree (max_depth=5), KNN (k=7), Gaussian Naive Bayes, and SVM (RBF kernel, probability=True).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5500 | 0.7029 | 0.6026 | 0.5188 | 0.5484 | 0.1967 |
| Decision Tree | 0.5750 | 0.7192 | 0.5605 | 0.5267 | 0.5374 | 0.2645 |
| KNN | 0.5917 | 0.6803 | 0.6046 | 0.5335 | 0.5550 | 0.2785 |
| Naive Bayes | 0.5500 | 0.6880 | 0.5341 | 0.5065 | 0.5179 | 0.2079 |
| Support Vector Machine | 0.5750 | 0.7164 | 0.7105 | 0.4208 | 0.4158 | 0.1862 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.55 and AUC 0.70 are modest, which suggests the classes are only partly linearly separable once the anxiety score is removed. It was the best at spotting the minority At-risk class (recall 0.50), but weak on Good (0.41). It was the most stable model in 5-fold cross-validation on the 600 rows (accuracy 0.58, macro-F1 0.50). |
| Decision Tree | Highest AUC (0.719) and second-highest MCC (0.265). A depth-5 tree captures non-linear effects of screen time and notifications and is easy to interpret, but it missed most At-risk cases (recall 0.38). Cross-validated accuracy (0.58) matches Logistic Regression. |
| KNN | Best accuracy (0.592), F1 (0.555) and MCC (0.279) on the test split, with reasonable recall on the two larger classes (Moderate 0.66, Good 0.56) but weaker At-risk recall (0.38). Its AUC (0.680) is the lowest because k=7 gives coarse probability estimates, and its cross-validated accuracy (0.55) is lower than on this particular test split, so part of its lead is likely luck. |
| Naive Bayes | Weakest overall in cross-validation (accuracy 0.51, MCC 0.17). The independence assumption is violated because anxiety-related scores, loneliness, self-esteem and life satisfaction are correlated, and one-hot features are not Gaussian. It is fast and has fairly even recall across classes (At-risk 0.44, Good 0.44, Moderate 0.65), but it never leads on any metric. |
| Support Vector Machine | Highest precision (0.711) but the lowest recall (0.421), F1 (0.416) and MCC (0.186). It recognises only 1 of 16 At-risk cases (recall 0.06) and mostly predicts Moderate, the majority class, so the high precision is misleading. Without class weighting or tuning it is a poor fit for this imbalanced problem. |

**Overall winner for this dataset: KNN** - it has the best Accuracy, F1 and MCC on the held-out test set. The margin is small, though: the test set has only 120 rows (16 At-risk), and in 5-fold cross-validation Logistic Regression and Decision Tree score as well or better. All five models land between 0.55 and 0.59 accuracy, so no model is clearly superior here.

Caveat: this is a 600-row sample and the relationship between the remaining features and wellbeing band is weak, which limits every model.

## How to Run

```
pip install -r requirements.txt
python train.py --source social_media_screentime_mental_health_2026.csv   # regenerates data files and models
streamlit run app.py
```

## Streamlit App Features

- Dataset upload (CSV) - upload `test_data.csv` or any CSV with the 15 feature columns and `wellbeing_band`
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Comparison table of all five models on the uploaded data
- Saved models are loaded from `model/`; nothing is retrained
