
from itertools import product
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "toxiguard_random_forest.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

sampling_methods = [
    "Niskin bottle",
    "Unknown",
    "hose-sampler (Lindahl 1986)",
]

mollusc_types = [
    "European flat oyster",
    "Mussel",
    "Pacific oyster",
]

species_values = [
    "Magallana gigas",
    "Mytilus galloprovincialis",
    "Ostrea edulis",
]

# Use typical environmental values from the training data.
# This avoids inventing arbitrary measurement values.
data_path = BASE_DIR / "Domoic_Acid.csv"
df = pd.read_csv(data_path)

numeric_fields = [
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

categorical_fields = [
    "sampling_method",
    "mollusc_type",
    "species",
]

required = numeric_fields + categorical_fields
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Dataset columns missing: {missing}")

# Start with the median environmental conditions.
base = {}
for col in numeric_fields:
    base[col] = pd.to_numeric(df[col], errors="coerce").median()

if any(pd.isna(value) for value in base.values()):
    raise ValueError(
        "Some numeric columns have no usable values. "
        "Check the dataset before continuing."
    )

rows = []
for method, mollusc, species in product(
    sampling_methods, mollusc_types, species_values
):
    row = {
        **base,
        "sampling_method": method,
        "mollusc_type": mollusc,
        "species": species,
    }
    rows.append(row)

X = pd.DataFrame(rows)

# Align inputs with the model's expected feature names.
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    X = X.reindex(columns=expected)

probabilities = model.predict_proba(X)
classes = list(model.classes_)

if 1 not in classes:
    raise ValueError(f"High-risk class 1 not found: {classes}")

high_risk_index = classes.index(1)
X["predicted_high_risk_pct"] = (
    probabilities[:, high_risk_index] * 100
).round(2)

print(
    X[
        categorical_fields + ["predicted_high_risk_pct"]
    ]
    .sort_values("predicted_high_risk_pct", ascending=False)
    .to_string(index=False)
)
