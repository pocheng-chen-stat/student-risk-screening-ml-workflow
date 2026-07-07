import pandas as pd

from src.cleaning import clean_dataset
from src.modeling import build_model_specs, make_train_test_split


def sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(12),
            "Gender": ["Male", "Female"] * 6,
            "Age": [20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25],
            "City": ["Delhi", "Mumbai"] * 6,
            "Profession": ["Student"] * 10 + ["Engineer", "Student"],
            "Academic Pressure": [3, 4, 5, 2, 1, 4, 5, 2, 3, 4, 2, 5],
            "CGPA": [8.0, 7.5, 0.0, 6.5, 9.0, 8.5, 7.2, 6.8, 9.1, 7.7, 8.1, 8.4],
            "Study Satisfaction": [3, 2, 1, 4, 5, 2, 1, 4, 3, 2, 5, 1],
            "Sleep Duration": ["5-6 hours", "Less than 5 hours"] * 6,
            "Dietary Habits": ["Healthy", "Unhealthy", "Moderate"] * 4,
            "Degree": ["BSc", "MSc", "BSc", "Others", "PhD", "BSc", "MSc", "BSc", "PhD", "BSc", "BSc", "MSc"],
            "Have you ever had suicidal thoughts ?": ["Yes", "No"] * 6,
            "Financial Stress": [3, 4, 5, 2, "?", 4, 5, 3, 2, 1, 4, 5],
            "Family History of Mental Illness": ["Yes", "No"] * 6,
            "Depression": [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        }
    )


def test_cleaning_applies_scope_and_anomaly_rules():
    cleaned, report = clean_dataset(sample_raw_data())
    assert "id" not in cleaned.columns
    assert "city" not in cleaned.columns
    assert "profession" not in cleaned.columns
    assert (cleaned["cgpa"] == 0).sum() == 0
    assert not cleaned["degree"].astype(str).str.lower().isin({"other", "others"}).any()
    assert cleaned["financial_stress"].isna().sum() == 0
    assert report.removed_non_student_rows == 1
    assert report.removed_cgpa_zero_rows == 1
    assert report.removed_degree_other_rows == 1
    assert report.removed_financial_stress_invalid_rows == 1


def test_train_test_split_preserves_classes():
    cleaned, _ = clean_dataset(sample_raw_data())
    split = make_train_test_split(cleaned, test_size=0.4)
    assert set(split.y_train.unique()) == {0, 1}
    assert set(split.y_test.unique()) == {0, 1}


def test_model_specs_separate_scaled_and_tree_preprocessing():
    cleaned, _ = clean_dataset(sample_raw_data())
    X = cleaned.drop(columns=["depression"])
    specs = build_model_specs(X)
    assert "logistic" in specs
    assert "random_forest" in specs
    logistic_numeric_pipeline = specs["logistic"][0].named_steps["preprocess"].transformers[0][1]
    rf_numeric_pipeline = specs["random_forest"][0].named_steps["preprocess"].transformers[0][1]
    assert "scaler" in logistic_numeric_pipeline.named_steps
    assert "scaler" not in rf_numeric_pipeline.named_steps
