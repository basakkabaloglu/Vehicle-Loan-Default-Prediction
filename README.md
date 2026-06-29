# Vehicle Loan Default Prediction using Explainable Machine Learning

## Overview

This project develops and evaluates machine learning models for predicting vehicle loan defaults using the L&T Vehicle Loan Default dataset. The study focuses on handling class imbalance, feature engineering, explainable AI (SHAP), and comparing multiple classification algorithms.

The complete workflow includes data preprocessing, exploratory data analysis, feature engineering, model selection, hyperparameter tuning, performance evaluation, and model interpretation.

---

## Dataset

* **Source:** Kaggle – L&T Vehicle Loan Default Prediction
* **Observations:** 233,154
* **Original Features:** 41
* **Target Variable:** `loan_default`
* **Default Rate:** 21.7%

---

## Project Workflow

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Feature Selection
* Cost-Sensitive Learning
* Hyperparameter Tuning
* Model Evaluation
* SHAP Explainability
* Borrower Segmentation (K-Means)

---

## Machine Learning Models

* Logistic Regression
* Decision Tree
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* LightGBM
* CatBoost

---

## Techniques Used

* Missing Value Imputation
* Frequency Encoding
* One-Hot Encoding
* Standardization
* Stratified Train-Test Split
* 5-Fold Cross Validation
* Grid Search
* Cost-Sensitive Learning
* SHAP
* McNemar Test

---

## Best Model Performance

| Model   | AUC       | F1 Score  | Recall    |
| ------- | --------- | --------- | --------- |
| XGBoost | **0.663** | **0.414** | **0.655** |

Gradient boosting models (XGBoost, CatBoost, and LightGBM) achieved the best overall performance.

---

## Feature Importance

SHAP analysis identified the most influential variables as:

* Loan-to-Value Ratio (LTV)
* Credit Score
* Loan Amount
* Branch Frequency
* Employment Type
* Age
* Credit History Features

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* SHAP
* Matplotlib
* Seaborn

---

## Repository Structure

```
Vehicle-Loan-Default-Prediction
│
├── notebooks/
│   └── vehicle_loan_default_prediction.ipynb
│
├── images/
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── shap_summary.png
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Results

The study shows that handling class imbalance through cost-sensitive learning substantially improves recall without significantly reducing AUC. Among all evaluated models, XGBoost achieved the strongest overall performance while SHAP provided interpretable explanations for model predictions.

---

## Author

**Başak Kabaloğlu**

Department of Statistics

Middle East Technical University
