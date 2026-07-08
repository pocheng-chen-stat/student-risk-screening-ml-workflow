## Results: Model Comparison

The best held-out ROC-AUC is from **XGBoost** (ROC-AUC = 0.922). The highest sensitivity is from **KNN** (sensitivity = 0.907).

The performance gap among XGBoost, logistic regression, ridge logistic, lasso logistic, and linear SVM is small. Therefore, the project does not stop at selecting the highest-scoring model; it also uses interpretable models to explain which variables repeatedly carry predictive signal.

### Held-Out Test Performance

| Model               |   Accuracy |   ROC-AUC |   PR-AUC |   Sensitivity |   Specificity |   Precision |    F1 |
|:--------------------|-----------:|----------:|---------:|--------------:|--------------:|------------:|------:|
| XGBoost             |      0.843 |     0.922 |    0.939 |         0.884 |         0.787 |       0.854 | 0.869 |
| Lasso Logistic      |      0.848 |     0.921 |    0.938 |         0.887 |         0.793 |       0.858 | 0.872 |
| Logistic Regression |      0.848 |     0.921 |    0.938 |         0.888 |         0.791 |       0.857 | 0.872 |
| Ridge Logistic      |      0.848 |     0.921 |    0.938 |         0.888 |         0.791 |       0.857 | 0.872 |
| Linear SVM          |      0.846 |     0.921 |    0.938 |         0.889 |         0.786 |       0.854 | 0.871 |
| Random Forest       |      0.839 |     0.917 |    0.933 |         0.853 |         0.818 |       0.869 | 0.861 |
| KNN                 |      0.842 |     0.911 |    0.922 |         0.907 |         0.749 |       0.836 | 0.87  |

![Held-out metric comparison](reports/figures/model_metric_comparison.png)

![ROC curve comparison](reports/figures/roc_curve_comparison.png)

![Precision-Recall curve comparison](reports/figures/precision_recall_curve_comparison.png)

### Training-Set Cross-Validation

Cross-validation is performed only on the training set. The held-out test set is used once at the end to estimate final performance.

| Model               |   Best CV ROC-AUC |   CV Std. | Selected Parameters                                             |
|:--------------------|------------------:|----------:|:----------------------------------------------------------------|
| Lasso Logistic      |             0.921 |     0     | {'C': 0.05}                                                     |
| Logistic Regression |             0.921 |     0     | {'C': 0.1}                                                      |
| Ridge Logistic      |             0.921 |     0     | {'C': 0.1}                                                      |
| Linear SVM          |             0.921 |     0     | {'C': 0.1}                                                      |
| XGBoost             |             0.92  |     0     | {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 150}    |
| Random Forest       |             0.915 |     0.001 | {'max_depth': None, 'min_samples_leaf': 5, 'n_estimators': 150} |
| KNN                 |             0.911 |     0.001 | {'n_neighbors': 25}                                             |

## Results: Feature Interpretation

After confirming that several models perform similarly well, the next question is: which explanatory variables are consistently selected? The figure below compares Lasso, Random Forest, and XGBoost. Lasso is sorted by absolute coefficient size, and the sign annotation keeps the positive/negative direction.

![Cross-model feature importance](reports/figures/cross_model_feature_importance.png)

### Cross-Model Consensus

Variables that repeatedly appear across model families are treated as stronger dataset-level predictive signals. This is still predictive agreement, not causal evidence.

| Feature                          | Lasso   | Random Forest   | XGBoost   |   Models in Top 10 |
|:---------------------------------|:--------|:----------------|:----------|-------------------:|
| Academic Pressure                | 2       | 2               | 2         |                  3 |
| Age                              | 5       | 4               | 6         |                  3 |
| Dietary Habits                   | 4       | 6               | 3         |                  3 |
| Financial Stress                 | 3       | 3               | 4         |                  3 |
| Sleep Duration                   | 7       | 10              | 7         |                  3 |
| Study Satisfaction               | 8       | 9               | 9         |                  3 |
| Suicidal Thoughts                | 1       | 1               | 1         |                  3 |
| Work/Study Hours                 | 6       | 5               | 8         |                  3 |
| CGPA                             | 10      | 7               |           |                  2 |
| Degree                           |         | 8               | 5         |                  2 |
| Family History of Mental Illness | 9       |                 | 10        |                  2 |

### Logistic Regression Odds-Ratio View

To make the logistic regression interpretation readable, an auxiliary logistic model is fit with numeric variables kept on their original scale. For numeric variables, `exp(beta)` is interpreted as the multiplicative change in the odds of the risk label for a one-unit increase, holding other variables fixed. For categorical variables, `exp(beta)` compares the displayed level with the reference level created during one-hot encoding.

| Feature           | Comparison                            | Direction   |   Coefficient | Odds Ratio exp(beta)   | Approx. odds change   |
|:------------------|:--------------------------------------|:------------|--------------:|:-----------------------|:----------------------|
| Suicidal Thoughts | Yes vs. reference level               | Higher odds |         2.511 | 12.31x                 | +1131%                |
| Dietary Habits    | Unhealthy vs. reference level         | Higher odds |         1.089 | 2.97x                  | +197%                 |
| Academic Pressure | one-unit increase                     | Higher odds |         0.844 | 2.33x                  | +133%                 |
| Financial Stress  | one-unit increase                     | Higher odds |         0.558 | 1.75x                  | +75%                  |
| Dietary Habits    | Moderate vs. reference level          | Higher odds |         0.493 | 1.64x                  | +64%                  |
| Degree            | Llb vs. reference level               | Higher odds |         0.413 | 1.51x                  | +51%                  |
| Sleep Duration    | Less Than 5 Hours vs. reference level | Higher odds |         0.384 | 1.47x                  | +47%                  |
| Degree            | Mbbs vs. reference level              | Higher odds |         0.363 | 1.44x                  | +44%                  |

### Lasso Direction Summary

Lasso is sorted by absolute coefficient size to show variable strength, while the direction column keeps whether the selected signal is associated with higher or lower predicted risk.

| Feature            | Risk-related Level   | Direction   |   Abs. Coefficient |
|:-------------------|:---------------------|:------------|-------------------:|
| Suicidal Thoughts  | Yes                  | Positive    |              1.248 |
| Academic Pressure  | numeric increase     | Positive    |              1.137 |
| Financial Stress   | numeric increase     | Positive    |              0.781 |
| Dietary Habits     | Unhealthy            | Positive    |              0.559 |
| Age                | numeric increase     | Negative    |              0.529 |
| Work/Study Hours   | numeric increase     | Positive    |              0.429 |
| Sleep Duration     | Less Than 5 Hours    | Positive    |              0.339 |
| Study Satisfaction | numeric increase     | Negative    |              0.326 |
