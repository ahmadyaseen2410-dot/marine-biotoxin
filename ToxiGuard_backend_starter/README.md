# ToxiGuard FastAPI backend starter

## 1. Put these files in `backend/`
Copy these files from your existing model project into the `backend` folder:
- `toxiguard_random_forest.joblib`
- `toxiguard_concentration_random_forest.joblib`
- `Domoic_Acid.csv`

Keep the Python scripts and frontend files outside or alongside this folder as you prefer.

## 2. Install dependencies
Open PowerShell in the `backend` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the install without activating:
```powershell
python -m pip install -r requirements.txt
```

## 3. Start the API
```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## 4. Test `/predict`
Use this sample request in `/docs`:

```json
{
  "pressure_dbar": 20,
  "temperature_c": 12,
  "salinity": 32,
  "sigma_theta": 24,
  "light_transmission_pct": 80,
  "par": 10,
  "chlorophyll_a": 1,
  "dissolved_oxygen": 7,
  "oxygen_saturation_pct": 95,
  "sampling_depth_m": 20,
  "sampling_method": "Unknown",
  "mollusc_type": "Unknown",
  "species": "Unknown"
}
```

## Important validation notes
- This is a starter integration, not proof that the models are correctly aligned. It mirrors the CLI classifier script's baseline-row approach, then checks expected classifier columns.
- The classifier training script was not provided, so feature preprocessing may still need adjustment if `/predict` reports a feature mismatch.
- The concentration model's training script dynamically selected columns from the training CSV. The API uses the saved pipeline's `feature_names_in_` and fills any fields not exposed by the UI from the first available dataset value. That baseline behavior must be reviewed against the original training setup before relying on predictions.
- Dashboard metrics and demo figures in the frontend are not automatically real model metrics.
- Do not present the output as an official safety determination. Validate with independent samples/stations and domain experts before real-world use.
