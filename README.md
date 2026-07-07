# Student Risk Screening ML Workflow

An end-to-end machine learning workflow for tabular binary classification, using the Kaggle Student Depression Dataset as a case study.

This project is not intended to diagnose depression or make clinical decisions. It demonstrates a reproducible workflow for risk-screening style classification problems.

## Project Positioning

The goal is not only to predict a label. The goal is to show a complete analysis pipeline:

```text
Raw data
-> scope and anomaly cleaning
-> exploratory data analysis
-> stratified train/test split
-> cross-validation on the training set
-> final held-out test evaluation
-> feature interpretation
```

## Dataset

Place the raw CSV here:

```text
data/raw/student_depression_dataset.csv
```

Target column:

```text
Depression
```

The dataset is treated as a synthetic case-study dataset rather than a clinical survey. Results should be interpreted as predictive patterns inside this dataset, not as medical conclusions.

## Cleaning Rules in v1

Version 1 makes the analysis population stricter and more consistent:

| Rule | Reason |
|---|---|
| Keep only `Profession == Student` | The project scope is student risk screening |
| Remove `CGPA == 0` | A zero CGPA is inconsistent with active student academic variables |
| Remove `Degree == Others` | The category is ambiguous and very rare |
| Remove invalid `Financial Stress` values such as `?` | The value is not a valid numeric stress score |
| Drop `City` | The project does not perform regional or spatial analysis |
| Drop `id` | Identifier columns should not be predictive features |

These cleaning decisions are intentionally conservative because the affected records are a very small part of the dataset and have ambiguous meanings.

The cleaning counts are written to:

```text
reports/cleaning_report.csv
```

## Modeling Workflow in v1

The workflow uses a held-out test set correctly:

```text
1. Split cleaned data into training and test sets.
2. Run cross-validation and hyperparameter tuning only on the training set.
3. Evaluate the selected models once on the held-out test set.
```

Outputs:

```text
reports/cv_model_comparison.csv
reports/test_model_comparison.csv
```

## Preprocessing

Different model families use different preprocessing:

| Models | Preprocessing |
|---|---|
| Logistic, Lasso Logistic, Ridge Logistic, KNN, SVM | numeric imputation + standardization, categorical imputation + one-hot encoding |
| Random Forest, XGBoost | numeric imputation without standardization, categorical imputation + one-hot encoding |

Tree models do not need numeric standardization, but categorical text variables still need encoding in the sklearn workflow.

## Models Included

Version 1 compares:

- Logistic Regression
- L1-regularized Logistic Regression
- L2-regularized Logistic Regression
- KNN
- Linear SVM
- Random Forest
- XGBoost, if installed

## Evaluation Metrics

The held-out test report includes:

- Accuracy
- ROC-AUC
- PR-AUC
- Sensitivity / Recall
- Specificity
- Precision
- F1 score

Sensitivity is included because false negatives matter in screening-style tasks. This project still does not make clinical recommendations.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
python main.py
```

Run tests:

```bash
python -m pytest
```

Generated outputs appear under:

```text
reports/
reports/figures/
```

## Repository Structure

```text
student-risk-screening-ml-workflow-v1/
├── README.md
├── requirements.txt
├── main.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── cleaning.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── interpretation.py
│   ├── modeling.py
│   └── visualization.py
├── reports/
│   └── figures/
└── tests/
    └── test_pipeline.py
```

## Responsible Interpretation

Feature importance and model coefficients should be interpreted as predictive signals or correlates in this dataset. They should not be interpreted as causal effects.
