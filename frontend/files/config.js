// ---- Editable settings. No model logic lives here, only configuration. ----
export const CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000",
  PREDICT_PATH: "/predict",
  USE_DEMO_DATA: true,                   // set false once real dataset/metrics endpoints exist
};

// Numeric inputs. `key` is the JSON field name sent to the backend.
export const NUMERIC_FIELDS = [
  { key: "pressure_dbar", label: "Pressure", unit: "dbar" },
  { key: "temperature_c", label: "Temperature", unit: "°C" },
  { key: "salinity", label: "Salinity", unit: "PSU" },
  { key: "sigma_theta", label: "Sigma-theta", unit: "kg/m³" },
  { key: "light_transmission_pct", label: "Light transmission", unit: "%" },
  { key: "par", label: "PAR", unit: "µmol/m²/s" },
  { key: "chlorophyll_a", label: "Chlorophyll-a", unit: "µg/L" },
  { key: "dissolved_oxygen", label: "Dissolved oxygen", unit: "mL/L" },
  { key: "oxygen_saturation_pct", label: "Oxygen saturation", unit: "%" },
  { key: "sampling_depth_m", label: "Sampling depth", unit: "m" },
];
// PLACEHOLDER options. Replace with the exact categories used when you trained the models.

export const CATEGORICAL_FIELDS = [
  {
    key: "sampling_method",
    label: "Sampling method",
    options: [
      "Niskin bottle",
      "Unknown",
      "hose-sampler (Lindahl 1986)",
    ],
  },
  {
    key: "mollusc_type",
    label: "Mollusc type",
    options: [
      "European flat oyster",
      "Mussel",
      "Pacific oyster",
    ],
  },
  {
    key: "species",
    label: "Species",
    options: [
      "Magallana gigas",
      "Mytilus galloprovincialis",
      "Ostrea edulis",
    ],
  },
];

// Model evaluation values. Leave null until you have real numbers.
export const METRICS = {
  regression:     { MAE: null, RMSE: null, "R²": null },
  classification: { Accuracy: null, Precision: null, Recall: null, "F1-score": null, "ROC-AUC": null },
};

// ---- DEMO placeholders (clearly labelled in the UI). Delete when real data is wired in. ----
export const DEMO_DATASET = {
  total: 362, high: 83, low: 279,
  meanByClass: { labels: ["Temperature", "Salinity", "Chlorophyll-a", "Oxygen sat."], low: [12, 33, 3, 95], high: [15, 32, 8, 88] },
  chlorophyllBins: { labels: ["0–2", "2–4", "4–6", "6–8", "8–10", "10+"], counts: [320, 280, 190, 110, 60, 40] },
};
export const DEMO_RESULT = { risk_label: "HIGH", probabilities: { low: 0.0, high: 1.0 }, estimated_concentration_mg_kg: null };
