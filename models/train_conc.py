
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# ==================================================
# 1. FILE PATHS
# ==================================================

DATA_PATH = Path("Domoic_Acid.csv")
MODEL_PATH = Path("toxiguard_concentration_random_forest.joblib")
METRICS_PATH = Path("toxiguard_regression_metrics.json")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

TARGET = "concentration_exact_mg_kg"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

if "risk_label" not in df.columns:
    raise ValueError("Column 'risk_label' not found.")

print("=" * 55)
print(" TOXIGUARD - HIGH-RISK CONCENTRATION MODEL")
print("=" * 55)

# ==================================================
# 3. SELECT HIGH-RISK SAMPLES
# ==================================================

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

high_risk = df[
    (df["risk_label"] == 1)
    & (df[TARGET].notna())
    & (df[TARGET] >= 0)
].copy()

print("\nTotal dataset records:", len(df))
print("High-risk records with known concentration:", len(high_risk))

if len(high_risk) < 10:
    raise ValueError(
        "Not enough high-risk records with known concentration "
        "to train and evaluate this model reliably."
    )

# ==================================================
# 4. PREPARE FEATURES
# ==================================================

# Exclude the target, risk labels, and fields that directly
# contain or describe the measured toxin result.
excluded_columns = [
    TARGET,
    "risk_label",
    "risk_category",
    "domoic_acid_concentration",
    "DA_concentration",
    "VALUE",
    "reported_value",
    "measurement_qualifier",
    "is_exact_measurement",
    "is_censored_measurement",
    "toxin_name",
    "toxin_code",
    "biotoxin_group",
    "units",
    "units_description",
    "record_origin",
    "date_raw",
    "time_raw",
    "date",
    "time",
    "datum",
]

X = high_risk.drop(
    columns=excluded_columns,
    errors="ignore"
).copy()

y = high_risk[TARGET].astype(float)

# Remove columns with no usable values or no variation.
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=False) > 1]

# Convert boolean columns to numeric.
for column in X.select_dtypes(include=["bool"]).columns:
    X[column] = X[column].astype(int)

# Convert numeric-looking text columns to numbers when appropriate.
for column in X.select_dtypes(include=["object", "str"]).columns:
    converted = pd.to_numeric(X[column], errors="coerce")

    if converted.notna().mean() >= 0.8:
        X[column] = converted

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

if not numeric_features and not categorical_features:
    raise ValueError("No usable input features were found.")

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ==================================================
# 5. PREPROCESSING
# ==================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# ==================================================
# 6. TRAIN AND TEST
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# ==================================================
# 7. EVALUATE MODEL
# ==================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

print("\n" + "=" * 55)
print(" MODEL EVALUATION")
print("=" * 55)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"MAE:  {mae:.3f} mg/kg")
print(f"RMSE: {rmse:.3f} mg/kg")
print(f"R²:   {r2:.3f}")

print("\nExample predictions:")
comparison = pd.DataFrame({
    "Actual concentration (mg/kg)": y_test.to_numpy(),
    "Predicted concentration (mg/kg)": predictions
})
print(comparison.head(10).round(3).to_string(index=False))

# ==================================================
# 8. SAVE EVALUATION METRICS
# ==================================================

metrics = {
    "MAE": round(float(mae), 3),
    "RMSE": round(float(rmse), 3),
    "R2": round(float(r2), 3) if np.isfinite(r2) else None,
    "test_samples": int(len(y_test)),
    "training_samples": int(len(y_train)),
    "target_unit": "mg/kg"
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nRegression metrics saved to:", METRICS_PATH.resolve())
# ==================================================
# 8. SAVE MODEL
# ==================================================

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH.resolve())
print("Training complete.")
