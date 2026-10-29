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
- **Train/test split:** stratified, giving **500 training rows** and **100 test rows** (Moderate 54, Good 32, At-risk 14), saved as `test_data.csv`.
- **Metric averaging:** Precision, Recall and F1 are macro-averaged over the three classes; AUC is one-vs-rest, macro-averaged, computed from predicted probabilities; MCC is the multiclass version.

## c. GitHub Repository Link

GITHUB_URL_PLACEHOLDER

```
project-folder/
|-- app.py                  Streamlit app
|-- train.py                data sampling, training, evaluation
|-- requirements.txt
|-- README.md
|-- test_data.csv           100-row test set (15 features + target)
|-- train_data.csv          500-row training set
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

All five models are trained on the same 500-row training set and evaluated on the same 100-row test set: Logistic Regression (max_iter=2000), Decision Tree (max_depth=5), KNN (k=7), Gaussian Naive Bayes, and SVM (RBF kernel, probability=True).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5300 | 0.7252 | 0.5779 | 0.4924 | 0.5201 | 0.1686 |
| Decision Tree | 0.5400 | 0.7026 | 0.5665 | 0.5113 | 0.5281 | 0.2141 |
| KNN | 0.5600 | 0.6500 | 0.5699 | 0.5103 | 0.5266 | 0.2371 |
| Naive Bayes | 0.5500 | 0.7011 | 0.5351 | 0.5047 | 0.5170 | 0.2113 |
| Support Vector Machine | 0.5700 | 0.7391 | 0.7056 | 0.4204 | 0.4165 | 0.1797 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-highest AUC (0.725) but the lowest accuracy (0.53) and MCC (0.169), so it ranks classes reasonably well but its hard predictions are weak, which suggests the classes are only partly linearly separable once the anxiety score is removed. Recall was 0.43 for At-risk, 0.44 for Good and 0.61 for Moderate. It was the most stable model in 5-fold cross-validation on the 600 rows (accuracy 0.58, macro-F1 0.50). |
| Decision Tree | Best macro-F1 (0.528) and second-best MCC (0.214) on the test set, with At-risk recall of 0.43 and Good recall of 0.53. A depth-5 tree captures non-linear effects of screen time and notifications and is easy to interpret. It also did best on MCC in cross-validation (0.25, accuracy 0.58), so its test result is consistent. Its AUC (0.703) is only middling because tree probabilities are coarse. |
| KNN | Best MCC (0.237) on the test set and nearly tied for best F1 (0.527), with recall of 0.61 for Moderate and 0.56 for Good but only 0.36 for At-risk. Its AUC (0.650) is the lowest because k=7 gives coarse probability estimates. Its cross-validated accuracy (0.55) is the lowest of the top models, so part of its test lead is likely luck. |
| Naive Bayes | Accuracy 0.55 and MCC 0.211 are competitive on this split, but it was the weakest model in cross-validation (accuracy 0.51, MCC 0.17). The independence assumption is violated because loneliness, self-esteem, life satisfaction and FOMO are correlated, and one-hot features are not Gaussian. It is fast and has fairly even recall (At-risk 0.43, Good 0.44, Moderate 0.65). |
| Support Vector Machine | Highest accuracy (0.57), AUC (0.739) and precision (0.706), but the lowest recall (0.420), F1 (0.417) and MCC (0.180). It predicted At-risk only once in 100 rows (recall 0.07) and labelled 77 rows Moderate, the majority class, so the high precision is misleading. Without class weighting or tuning it is a poor fit for this imbalanced problem. |

**Overall winner for this dataset: Decision Tree** - it has the best macro-F1 on the held-out test set, the second-best MCC, and the best cross-validated MCC, so its performance is consistent across both checks and it is interpretable. The margin is small: KNN has the highest test MCC, and the SVM has the highest accuracy and AUC but misses nearly all At-risk cases. All five models land between 0.53 and 0.57 accuracy, so no model is clearly superior here, and with only 100 test rows (14 At-risk) differences of a few points are within noise.

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
