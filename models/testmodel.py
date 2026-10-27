import joblib
import pandas as pd
from pathlib import Path

DATA_PATH = Path("Domoic_Acid.csv")
MODEL_PATH = Path("toxiguard_random_forest.joblib")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = "risk_label"

drop_columns = [
    TARGET,
    "risk_category",
    "domoic_acid_concentration",
    "DA_concentration",
    "VALUE",
    "reported_value",
    "concentration_exact_mg_kg",
    "measurement_qualifier",
    "record_origin",
    "is_exact_measurement",
    "is_censored_measurement",
    "toxin_name",
    "toxin_code",
    "biotoxin_group",
    "units",
    "units_description",
    "synthetic",
    "record_source",
    "source_type",
]

# Select a known high-risk record
sample = df[df[TARGET] == 1].iloc[[0]].copy()

# Remove the target and other columns excluded during training
X = sample.drop(columns=drop_columns, errors="ignore")

# Match the features expected by the trained model
expected_features = list(model.feature_names_in_)

# Handle date columns in the same way as your training script
for col in list(X.columns):
    if "date" in col.lower():
        parsed = pd.to_datetime(X[col], errors="coerce")

        if parsed.notna().any():
            X[col + "_year"] = parsed.dt.year
            X[col + "_month"] = parsed.dt.month
            X[col + "_day"] = parsed.dt.day
            X = X.drop(columns=[col])

X = X.reindex(columns=expected_features)

# Predict
prediction = int(model.predict(X)[0])

print("Actual label:", int(sample[TARGET].iloc[0]))
print("Predicted label:", prediction)

if prediction == 1:
    print("Predicted risk: HIGH RISK")
else:
    print("Predicted risk: LOW RISK")

print("\nNote: This test is valid only if the feature preparation")
print("matches the training pipeline exactly.")