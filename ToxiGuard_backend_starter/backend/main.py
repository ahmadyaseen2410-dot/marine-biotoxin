from pathlib import Path
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Place the two .joblib files and Domoic_Acid.csv in the backend folder.
BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = BASE_DIR / "toxiguard_random_forest.joblib"
REGRESSOR_PATH = BASE_DIR / "toxiguard_concentration_random_forest.joblib"
DATA_PATH = BASE_DIR / "Domoic_Acid.csv"

app = FastAPI(title="ToxiGuard API", version="1.0.0")

# Local development only. Restrict this to your deployed frontend origin later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

NUMERIC_INPUTS = [
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
CATEGORICAL_INPUTS = ["sampling_method", "mollusc_type", "species"]

DROP_CLASSIFIER_COLUMNS = [
    "risk_label", "risk_category", "domoic_acid_concentration",
    "DA_concentration", "VALUE", "reported_value",
    "concentration_exact_mg_kg", "measurement_qualifier",
    "record_origin", "is_exact_measurement", "is_censored_measurement",
    "toxin_name", "toxin_code", "biotoxin_group", "units",
    "units_description", "synthetic", "record_source", "source_type",
]

class PredictionRequest(BaseModel):
    pressure_dbar: float
    temperature_c: float
    salinity: float
    sigma_theta: float
    light_transmission_pct: float
    par: float
    chlorophyll_a: float
    dissolved_oxygen: float
    oxygen_saturation_pct: float
    sampling_depth_m: float
    sampling_method: str = "Unknown"
    mollusc_type: str = "Unknown"
    species: str = "Unknown"

def load_assets():
    missing = [p.name for p in (CLASSIFIER_PATH, REGRESSOR_PATH, DATA_PATH) if not p.exists()]
    if missing:
        return None, None, None, missing
    try:
        classifier = joblib.load(CLASSIFIER_PATH)
        regressor = joblib.load(REGRESSOR_PATH)
        dataset = pd.read_csv(DATA_PATH)
        return classifier, regressor, dataset, []
    except Exception as exc:
        raise RuntimeError(f"Could not load ToxiGuard assets: {exc}") from exc

classifier, regressor, dataset, missing_assets = load_assets()

@app.get("/")
def root():
    return {
        "message": "ToxiGuard API",
        "status": "ready" if not missing_assets else "missing_assets",
        "missing_assets": missing_assets,
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if not missing_assets else "not_ready",
        "missing_assets": missing_assets,
    }


def make_classifier_frame(payload: PredictionRequest) -> pd.DataFrame:
    if dataset is None:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not loaded: {missing_assets}"
        )

    # Start with a dataset row as the template.
    frame = dataset.iloc[[0]].copy()

    # Replace user-provided numeric measurements.
    for col in NUMERIC_INPUTS:
        if col in frame.columns:
            frame.loc[:, col] = getattr(payload, col)

    # Preserve template categories when the input is "Unknown".
    for col in CATEGORICAL_INPUTS:
        if col in frame.columns:
            value = getattr(payload, col, None)

            if (
                value
                and str(value).strip().lower() != "unknown"
            ):
                frame.loc[:, col] = value

    # Remove targets and leakage-related columns.
    X = frame.drop(
        columns=DROP_CLASSIFIER_COLUMNS,
        errors="ignore"
    )

    # Apply the same constant-column filtering used in your CLI script.
    X_reference = dataset.drop(
        columns=DROP_CLASSIFIER_COLUMNS,
        errors="ignore"
    )
    X_reference = X_reference.loc[
        :, X_reference.nunique(dropna=False) > 1
    ]

    X = X.reindex(columns=X_reference.columns)

    # Expand date columns, if present and parseable.
    for col in list(X.columns):
        if "date" in col.lower():
            parsed = pd.to_datetime(X[col], errors="coerce")

            if parsed.notna().any():
                X[f"{col}_year"] = parsed.dt.year
                X[f"{col}_month"] = parsed.dt.month
                X[f"{col}_day"] = parsed.dt.day
                X = X.drop(columns=[col])

    # Remove identifier columns.
    identifier_words = {"sample_id", "record_id", "row_id"}
    X = X.drop(
        columns=[
            col for col in X.columns
            if col.lower() in identifier_words
        ],
        errors="ignore"
    )

    # Convert object values consistently with the CLI script.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].map(
                lambda value: str(value)
                if pd.notna(value) else None
            )

    # Align with the saved model's exact input columns.
    expected = getattr(classifier, "feature_names_in_", None)

    if expected is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Classifier does not expose feature_names_in_. "
                "Check the original training script."
            )
        )

    missing = [col for col in expected if col not in X.columns]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing expected classifier features: {missing}"
        )

    return X.reindex(columns=list(expected))


    identifier_words = ["sample_id", "record_id", "row_id"]
    X = X.drop(columns=[c for c in X.columns if c.lower() in identifier_words], errors="ignore")

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else None)

    expected = getattr(classifier, "feature_names_in_", None)
    if expected is None:
        raise HTTPException(
            status_code=500,
            detail="Classifier has no feature_names_in_. Please share the classifier training script so its feature preparation can be reproduced exactly."
        )
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Classifier feature mismatch. Missing expected columns: {missing}. Recreate preprocessing from the original training script."
        )
    return X.reindex(columns=list(expected))

def make_regressor_frame(payload: PredictionRequest) -> pd.DataFrame:
    # The regressor is a sklearn Pipeline trained on selected raw columns.
    expected = getattr(regressor, "feature_names_in_", None)
    if expected is None:
        raise HTTPException(status_code=500, detail="Could not read regressor input feature names.")
    values = {}
    for col in expected:
        if hasattr(payload, col):
            values[col] = getattr(payload, col)
        elif dataset is not None and col in dataset.columns:
            # Preserve a known baseline for any non-UI field the pipeline expects.
            values[col] = dataset[col].dropna().iloc[0] if dataset[col].notna().any() else np.nan
        else:
            values[col] = np.nan
    return pd.DataFrame([values], columns=list(expected))

@app.post("/predict")
def predict(payload: PredictionRequest):
    if missing_assets:
        raise HTTPException(status_code=503, detail=f"Missing backend files: {missing_assets}")
    if classifier is None or regressor is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    X_cls = make_classifier_frame(payload)
    try:
        risk_prediction = int(classifier.predict(X_cls)[0])
        probabilities = {}
        if hasattr(classifier, "predict_proba"):
            raw = classifier.predict_proba(X_cls)[0]
            for label, probability in zip(classifier.classes_, raw):
                probabilities["high_risk" if int(label) == 1 else "low_risk"] = round(float(probability), 4)

        result = {
            "risk_label": "HIGH RISK" if risk_prediction == 1 else "LOW RISK",
            "risk_prediction": risk_prediction,
            "probabilities": probabilities,
            "concentration_mg_kg": None,
            "concentration_note": "Concentration estimated only when classifier predicts HIGH RISK.",
            "disclaimer": "Research prototype only; not a validated food-safety or public-health decision system.",
        }

        if risk_prediction == 1:
            X_reg = make_regressor_frame(payload)
            concentration = float(regressor.predict(X_reg)[0])
            result["concentration_mg_kg"] = round(max(0.0, concentration), 3)

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
