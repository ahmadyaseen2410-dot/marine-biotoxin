
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
    balanced_accuracy_score
)

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
# 2. STANDARDIZE RECORD ORIGIN LABELS
# --------------------------------------------------

if "record_origin" in df.columns:
    df["record_origin"] = "OBSERVED"

    # Save the updated dataset
    df.to_csv(DATA_PATH, index=False)

    print("\nRecord origin counts:")
    print(df["record_origin"].value_counts(dropna=False))

# --------------------------------------------------
# 3. PREPARE TARGET
# --------------------------------------------------

df = df[df[TARGET].notna()].copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

print("\nRisk label counts:")
print(df[TARGET].value_counts().sort_index())

# --------------------------------------------------
# 4. REMOVE TARGET LEAKAGE
# --------------------------------------------------

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

X = df.drop(columns=drop_columns, errors="ignore")
y = df[TARGET]

# Remove constant columns
X = X.loc[:, X.nunique(dropna=False) > 1]

print("\n--- FEATURE AUDIT ---")
print("Target column:", TARGET)
print("Features used by model:")

for col in X.columns:
    print("-", col)

print("\nTarget distribution:")
print(y.value_counts())

if X.shape[1] == 0:
    raise ValueError("No usable input features remain.")

# --------------------------------------------------
# 5. PROCESS DATE COLUMNS
# --------------------------------------------------

for col in list(X.columns):
    if "date" in col.lower():
        parsed = pd.to_datetime(X[col], errors="coerce")

        if parsed.notna().any():
            X[col + "_year"] = parsed.dt.year
            X[col + "_month"] = parsed.dt.month
            X[col + "_day"] = parsed.dt.day
            X = X.drop(columns=[col])

# Remove record identifier columns
identifier_words = ["sample_id", "record_id", "row_id"]

X = X.drop(
    columns=[
        c for c in X.columns
        if c.lower() in identifier_words
    ],
    errors="ignore"
)

# Normalize categorical values
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].map(
            lambda value: str(value) if pd.notna(value) else None
        )

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

print(
    "\nBalanced Accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)

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
