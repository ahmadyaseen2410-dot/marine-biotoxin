
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import json

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

DATA_PATH = "Domoic_Acid.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

TARGET = "risk_label"

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' was not found. "
        "Check your CSV column names."
    )


# --------------------------------------------------
# 2. PREPARE TARGET AND AUDIT DATA ORIGIN
# --------------------------------------------------

df = df[df[TARGET].notna()].copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

print("\nRecord origin counts:")
if "record_origin" in df.columns:
    print(df["record_origin"].value_counts(dropna=False))
else:
    print("No record_origin column available.")

print("\nRisk label counts:")
print(df[TARGET].value_counts().sort_index())

# --------------------------------------------------
# 3. SELECT ONLY PREDICTION-TIME FEATURES
# --------------------------------------------------

# Use only fields that could be available when making a prediction.
# Do not include toxin measurements or fields derived from the target.

allowed_features = [
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
    "sampling_method",
    "mollusc_type",
    "species",
]

available_features = [
    col for col in allowed_features if col in df.columns
]

missing_features = [
    col for col in allowed_features if col not in df.columns
]

if missing_features:
    print("\nFeatures not found in CSV:", missing_features)

if not available_features:
    raise ValueError("None of the allowed prediction features exist.")

X = df[available_features].copy()
y = df[TARGET].copy()

# Remove constant features
X = X.loc[:, X.nunique(dropna=False) > 1]

# Normalize categorical values
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = X[col].map(
        lambda value: str(value) if pd.notna(value) else None
    )

print("\n--- FEATURE AUDIT ---")
print("Target:", TARGET)
print("Features used:")
for col in X.columns:
    print("-", col)

print("\nExcluded fields include:")
print("- Risk labels and categories")
print("- Toxin concentrations and laboratory results")
print("- Record origin and source identifiers")
print("- Station, coordinates, and date/time fields")

if X.shape[1] == 0:
    raise ValueError("No usable input features remain.")


# --------------------------------------------------
# 6. CHECK CLASS BALANCE
# --------------------------------------------------

counts = y.value_counts()

if len(counts) != 2 or counts.min() < 2:
    raise ValueError(
        "The dataset must contain at least two records "
        "from each risk class."
    )

# --------------------------------------------------
# 7. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 8. PREPROCESS FEATURES
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 9. RANDOM FOREST CLASSIFIER
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)


# --------------------------------------------------
# 10. EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Find the probability column belonging to class 1 (HIGH RISK)
high_risk_index = list(model.classes_).index(1)
high_risk_probabilities = probabilities[:, high_risk_index]

# Calculate evaluation metrics
metrics = {
    "Accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "Precision": round(float(precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    )), 4),
    "Recall": round(float(recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )), 4),
    "F1-score": round(float(f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    )), 4),
    "ROC-AUC": round(float(roc_auc_score(
        y_test, high_risk_probabilities
    )), 4),
    "Balanced Accuracy": round(float(
        balanced_accuracy_score(y_test, predictions)
    ), 4),
    "test_samples": int(len(y_test))
}

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Low Risk", "High Risk"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nCLASSIFIER EVALUATION METRICS")
for name, value in metrics.items():
    print(f"{name}: {value}")

# Save metrics for the dashboard integration
with open("toxiguard_classification_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved to toxiguard_classification_metrics.json")


# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

MODEL_PATH = os.path.join(
    "models",
    "toxiguard_random_forest.joblib"
)

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
