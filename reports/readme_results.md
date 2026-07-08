## Results

The strongest held-out ROC-AUC is from **XGBoost** (ROC-AUC = 0.924). The highest sensitivity is from **KNN** (sensitivity = 0.907).

### Held-Out Test Performance

| Model               |   Accuracy |   ROC-AUC |   PR-AUC |   Sensitivity |   Specificity |   Precision |    F1 |
|:--------------------|-----------:|----------:|---------:|--------------:|--------------:|------------:|------:|
| XGBoost             |      0.848 |     0.924 |    0.941 |         0.889 |         0.791 |       0.857 | 0.873 |
| Lasso Logistic      |      0.849 |     0.923 |    0.94  |         0.886 |         0.797 |       0.86  | 0.873 |
| Linear SVM          |      0.849 |     0.923 |    0.94  |         0.887 |         0.795 |       0.859 | 0.873 |
| Ridge Logistic      |      0.848 |     0.922 |    0.94  |         0.885 |         0.796 |       0.86  | 0.872 |
| Logistic Regression |      0.848 |     0.922 |    0.94  |         0.885 |         0.796 |       0.86  | 0.872 |
| Random Forest       |      0.841 |     0.919 |    0.938 |         0.853 |         0.824 |       0.873 | 0.862 |
| KNN                 |      0.84  |     0.913 |    0.925 |         0.907 |         0.746 |       0.834 | 0.869 |

![Held-out metric comparison](reports/figures/model_metric_comparison.png)

![ROC curve comparison](reports/figures/roc_curve_comparison.png)

![Precision-Recall curve comparison](reports/figures/precision_recall_curve_comparison.png)

### Training-Set Cross-Validation

| Model               |   Best CV ROC-AUC |   CV Std. | Selected Parameters                                             |
|:--------------------|------------------:|----------:|:----------------------------------------------------------------|
| Lasso Logistic      |             0.921 |     0.002 | {'C': 0.1}                                                      |
| Logistic Regression |             0.921 |     0.003 | {'C': 0.1}                                                      |
| Ridge Logistic      |             0.921 |     0.003 | {'C': 0.1}                                                      |
| Linear SVM          |             0.92  |     0.003 | {'C': 0.1}                                                      |
| XGBoost             |             0.919 |     0.002 | {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 150}    |
| Random Forest       |             0.915 |     0.002 | {'max_depth': None, 'min_samples_leaf': 5, 'n_estimators': 150} |
| KNN                 |             0.909 |     0.002 | {'n_neighbors': 25}                                             |

## Cross-Model Feature Interpretation

The goal of this section is not to claim psychological causality. Instead, the analysis asks whether similar predictive signals appear across linear and tree-based models.

![Cross-model feature importance](reports/figures/cross_model_feature_importance.png)

### Lasso Direction Summary

Lasso is useful because it gives both variable magnitude and the direction of the selected signal. The ranking below uses the absolute coefficient size, while the direction column keeps the sign information.

| Feature            | Risk-related Level   | Direction   |   Abs. Coefficient |
|:-------------------|:---------------------|:------------|-------------------:|
| Suicidal Thoughts  | No                   | Negative    |              1.228 |
| Academic Pressure  | numeric increase     | Positive    |              1.151 |
| Financial Stress   | numeric increase     | Positive    |              0.79  |
| Dietary Habits     | Unhealthy            | Positive    |              0.581 |
| Age                | numeric increase     | Negative    |              0.535 |
| Work/Study Hours   | numeric increase     | Positive    |              0.426 |
| Sleep Duration     | Less Than 5 Hours    | Positive    |              0.377 |
| Study Satisfaction | numeric increase     | Negative    |              0.327 |

### Logistic Regression Interpretation

Traditional logistic regression remains valuable here because its coefficients give a direct, readable risk-direction summary. In this dataset, variables such as academic pressure and financial stress tend to carry positive coefficients, while some lifestyle or demographic levels carry negative coefficients. These are predictive associations in a synthetic dataset, not causal explanations.

Top positive logistic signals:

| Feature           | Level            |   Coefficient |
|:------------------|:-----------------|--------------:|
| Suicidal Thoughts | Yes              |         1.264 |
| Academic Pressure | numeric increase |         1.154 |
| Financial Stress  | numeric increase |         0.792 |
| Dietary Habits    | Unhealthy        |         0.562 |
| Work/Study Hours  | numeric increase |         0.43  |

Top negative logistic signals:

| Feature            | Level            |   Coefficient |
|:-------------------|:-----------------|--------------:|
| Age                | numeric increase |        -0.556 |
| Study Satisfaction | numeric increase |        -0.329 |
| Job Satisfaction   | numeric increase |        -0.038 |

### Cross-Model Consensus

A feature is more convincing as a dataset-level predictive signal when it appears repeatedly across model families.

| Feature                          | Lasso   | Random Forest   | XGBoost   |   Models in Top 10 |
|:---------------------------------|:--------|:----------------|:----------|-------------------:|
| Academic Pressure                | 2       | 2               | 2         |                  3 |
| Age                              | 5       | 4               | 7         |                  3 |
| Degree                           | 10      | 9               | 5         |                  3 |
| Dietary Habits                   | 4       | 6               | 3         |                  3 |
| Financial Stress                 | 3       | 3               | 4         |                  3 |
| Sleep Duration                   | 7       | 10              | 6         |                  3 |
| Study Satisfaction               | 8       | 8               | 9         |                  3 |
| Suicidal Thoughts                | 1       | 1               | 1         |                  3 |
| Work/Study Hours                 | 6       | 5               | 8         |                  3 |
| Family History of Mental Illness | 9       |                 | 10        |                  2 |
| CGPA                             |         | 7               |           |                  1 |
