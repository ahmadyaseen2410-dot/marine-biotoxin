
import { CONFIG } from "./config.js";

export class BackendError extends Error {}

export async function predictRisk(payload) {
  let response;

  try {
    response = await fetch(
      CONFIG.API_BASE_URL + CONFIG.PREDICT_PATH,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      }
    );
  } catch {
    throw new BackendError(
      "Cannot connect to FastAPI. Check that the backend is running."
    );
  }

  const data = await response.json();

  if (!response.ok) {
    throw new BackendError(
      typeof data.detail === "string"
        ? data.detail
        : `Backend error: HTTP ${response.status}`
    );
  }

  const high =
    String(data.risk_label).toUpperCase().includes("HIGH");

  return {
    risk_label: high ? "HIGH" : "LOW",

    probabilities: {
      low:
        data.probabilities.low ??
        data.probabilities.low_risk,

      high:
        data.probabilities.high ??
        data.probabilities.high_risk
    },

    estimated_concentration_mg_kg:
      data.concentration_mg_kg ??
      data.estimated_concentration_mg_kg ??
      null
  };
}
