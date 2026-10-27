# ToxiGuard frontend (static, no build step)

Run locally:  `python -m http.server 5500`  (ES modules need http, not file://)

## Files
- `js/config.js`  fields, dropdown options, metrics, demo data (edit here)
- `js/api.js`     the only code that calls the backend
- `js/app.js`     UI rendering
- `css/styles.css`

## FastAPI hookup
Serve it from your app:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory="toxiguard", html=True), name="ui")   # mount AFTER your routes
and set `API_BASE_URL: ""` in config.js. If the frontend runs on another port, add CORSMiddleware.

`POST /predict` request: pressure, temperature, salinity, sigma_theta, light_transmission, par,
chlorophyll_a, dissolved_oxygen, oxygen_saturation, sampling_depth (numbers) + sampling_method,
mollusc_type, species (strings).
Response: `{"risk_label":"HIGH"|"LOW","probabilities":{"low":0.2,"high":0.8},"estimated_concentration_mg_kg":1.5|null}`
