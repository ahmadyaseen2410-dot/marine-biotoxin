
import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = Path("Domoic_Acid.csv")
MODEL_PATH = Path("toxiguard_random_forest.joblib")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

print("=" * 45)
print("       TOXIGUARD RISK PREDICTION")
print("=" * 45)

# --------------------------------------------------
# 3. PREPARE A SAMPLE INPUT
# --------------------------------------------------

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

X_reference = df.drop(
    columns=drop_columns,
    errors="ignore"
)

# Use the same constant-column filtering as training
X_reference = X_reference.loc[
    :, X_reference.nunique(dropna=False) > 1
]

# Start with an existing row as a template
sample = df.iloc[[0]].copy()

# Numeric measurements to request
numeric_inputs = [
    "pressure_dbar",
    "temperature_c",
    "salinity",
    "sigma_theta",
    "light_transmission_pct",
    "par",
    "chlorophyll_a",
    "dissolved_oxygen",
    "oxygen_saturation_pct",
    "sampling_depth_m",
]

print("\nEnter environmental measurements.")
print("Press Enter to use the dataset median as the default.\n")

for column in numeric_inputs:
    if column not in X_reference.columns:
        continue

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    if values.empty:
        continue

    default_value = float(values.median())

    while True:
        entered = input(
            f"{column} [{default_value:.3f}]: "
        ).strip()

        if entered == "":
            sample.loc[:, column] = default_value
            break

        try:
            sample.loc[:, column] = float(entered)
            break
        except ValueError:
            print("Please enter a valid number.")

# --------------------------------------------------
# 4. APPLY THE SAME FEATURE PREPARATION
# --------------------------------------------------

def prepare_features(frame):
    X = frame.drop(
        columns=drop_columns,
        errors="ignore"
    )

    X = X.reindex(columns=X_reference.columns)

    for col in list(X.columns):
        if "date" in col.lower():
            parsed = pd.to_datetime(
                X[col], errors="coerce"
            )

            if parsed.notna().any():
                X[col + "_year"] = parsed.dt.year
                X[col + "_month"] = parsed.dt.month
                X[col + "_day"] = parsed.dt.day
                X = X.drop(columns=[col])

    identifier_words = [
        "sample_id",
        "record_id",
        "row_id"
    ]

    X = X.drop(
        columns=[
            c for c in X.columns
            if c.lower() in identifier_words
        ],
        errors="ignore"
    )

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].map(
                lambda value:
                str(value) if pd.notna(value) else None
            )

    return X


X_new = prepare_features(sample)

# Match the exact feature order expected by the saved model
expected_features = list(model.feature_names_in_)
X_new = X_new.reindex(columns=expected_features)

# --------------------------------------------------
# 5. PREDICT RISK
# --------------------------------------------------

prediction = int(model.predict(X_new)[0])

print("\n" + "=" * 45)
print("             PREDICTION RESULT")
print("=" * 45)

if prediction == 1:
    print("Predicted Risk: HIGH RISK")
    print("Warning: The model classified this input as high-risk.")
else:
    print("Predicted Risk: LOW RISK")
    print("The model classified this input as low-risk.")

# Display class probabilities when available
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_new)[0]

    print("\nModel probability estimates:")

    for class_label, probability in zip(
        model.classes_, probabilities
    ):
        label = "High Risk" if class_label == 1 else "Low Risk"
        print(f"{label}: {probability * 100:.2f}%")

print("=" * 45)
