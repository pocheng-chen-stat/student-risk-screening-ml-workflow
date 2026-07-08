## Model Performance

Held-out test performance after training-set cross-validation tuning:

| model          |   accuracy |   roc_auc |   pr_auc |   sensitivity_recall |   specificity |   precision |    f1 |
|:---------------|-----------:|----------:|---------:|---------------------:|--------------:|------------:|------:|
| lasso_logistic |      0.849 |     0.923 |    0.94  |                0.885 |         0.797 |       0.86  | 0.873 |
| svm_linear     |      0.849 |     0.923 |    0.94  |                0.887 |         0.795 |       0.859 | 0.873 |
| logistic       |      0.848 |     0.922 |    0.94  |                0.885 |         0.796 |       0.86  | 0.872 |
| ridge_logistic |      0.848 |     0.922 |    0.94  |                0.885 |         0.796 |       0.86  | 0.872 |
| xgboost        |      0.846 |     0.921 |    0.939 |                0.889 |         0.785 |       0.854 | 0.871 |
| random_forest  |      0.84  |     0.919 |    0.937 |                0.851 |         0.824 |       0.872 | 0.861 |
| knn            |      0.84  |     0.913 |    0.925 |                0.907 |         0.746 |       0.834 | 0.869 |

Training-set cross-validation stability:

| model          |   best_cv_roc_auc |   cv_std_roc_auc | best_params                                                                          |
|:---------------|------------------:|-----------------:|:-------------------------------------------------------------------------------------|
| lasso_logistic |             0.921 |            0.002 | {'model__C': 0.1}                                                                    |
| logistic       |             0.921 |            0.003 | {'model__C': 0.1}                                                                    |
| ridge_logistic |             0.921 |            0.003 | {'model__C': 0.1}                                                                    |
| svm_linear     |             0.92  |            0.003 | {'model__C': 0.1}                                                                    |
| xgboost        |             0.918 |            0.002 | {'model__learning_rate': 0.05, 'model__max_depth': 3, 'model__n_estimators': 100}    |
| random_forest  |             0.915 |            0.002 | {'model__max_depth': None, 'model__min_samples_leaf': 5, 'model__n_estimators': 120} |
| knn            |             0.909 |            0.002 | {'model__n_neighbors': 25}                                                           |

![ROC curve comparison](reports/figures/roc_curve_comparison.png)

![Precision-Recall curve comparison](reports/figures/precision_recall_curve_comparison.png)

## Important Predictive Features

Feature importance should be interpreted as predictive signal, not causal effect.

### Lasso logistic coefficients

| feature                                              |   coefficient |
|:-----------------------------------------------------|--------------:|
| categorical__have_you_ever_had_suicidal_thoughts_Yes |        1.4077 |
| numeric__academic_pressure                           |        1.1509 |
| categorical__have_you_ever_had_suicidal_thoughts_No  |       -1.0441 |
| numeric__financial_stress                            |        0.79   |
| categorical__dietary_habits_Unhealthy                |        0.5807 |
| numeric__age                                         |       -0.5353 |
| categorical__dietary_habits_Healthy                  |       -0.5011 |
| numeric__work_study_hours                            |        0.4262 |
| categorical__sleep_duration_'Less than 5 hours'      |        0.3764 |
| numeric__study_satisfaction                          |       -0.3273 |

### Random Forest feature importance

| feature                                              |   importance |
|:-----------------------------------------------------|-------------:|
| numeric__academic_pressure                           |       0.2106 |
| categorical__have_you_ever_had_suicidal_thoughts_No  |       0.1884 |
| categorical__have_you_ever_had_suicidal_thoughts_Yes |       0.1874 |
| numeric__financial_stress                            |       0.1138 |
| numeric__age                                         |       0.0595 |
| numeric__work_study_hours                            |       0.049  |
| numeric__cgpa                                        |       0.038  |
| numeric__study_satisfaction                          |       0.0306 |
| categorical__dietary_habits_Unhealthy                |       0.0246 |
| categorical__dietary_habits_Healthy                  |       0.0163 |

### XGBoost feature importance

| feature                                             |   importance |
|:----------------------------------------------------|-------------:|
| categorical__have_you_ever_had_suicidal_thoughts_No |       0.5484 |
| numeric__academic_pressure                          |       0.1251 |
| numeric__financial_stress                           |       0.0697 |
| categorical__dietary_habits_Healthy                 |       0.0417 |
| numeric__work_study_hours                           |       0.0388 |
| categorical__dietary_habits_Unhealthy               |       0.0375 |
| numeric__age                                        |       0.0374 |
| numeric__study_satisfaction                         |       0.0269 |
| categorical__sleep_duration_'Less than 5 hours'     |       0.0186 |
| categorical__family_history_of_mental_illness_No    |       0.018  |

### Permutation importance

| feature                             |   importance_mean |
|:------------------------------------|------------------:|
| have_you_ever_had_suicidal_thoughts |            0.1083 |
| academic_pressure                   |            0.0844 |
| financial_stress                    |            0.0323 |
| age                                 |            0.0152 |
| dietary_habits                      |            0.012  |
| work_study_hours                    |            0.0093 |
| study_satisfaction                  |            0.0052 |
| sleep_duration                      |            0.0017 |
| cgpa                                |            0.0008 |
| family_history_of_mental_illness    |            0.0005 |
