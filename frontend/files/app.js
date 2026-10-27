// ---- UI only. No model logic: it calls api.js and renders what comes back. ----
import { CONFIG, NUMERIC_FIELDS, CATEGORICAL_FIELDS, METRICS, DEMO_DATASET, DEMO_RESULT } from "./config.js";
import { predictRisk, BackendError } from "./api.js";

const $ = (s) => document.querySelector(s);
const fmt = (v, d = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const TEAL = "#13a89e", NAVY = "#13315c", RED = "#c0392b", GREEN = "#1f9d6b";

// ----- Navigation -----
function showPage() {
  const page = (location.hash || "#dashboard").slice(1);
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + page));
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  $("#sidebar").classList.remove("open");
  $("#demoBanner").hidden = !CONFIG.USE_DEMO_DATA || page === "about" || page === "predict" || page === "performance";
}
addEventListener("hashchange", showPage);
$("#menuBtn").onclick = () => $("#sidebar").classList.toggle("open");

// ----- Cards -----
const card = (label, value, cls = "", tag = "") =>
  `<div class="card ${cls}"><div class="label">${label}</div><div class="value">${value}</div>${tag ? `<div class="tag">${tag}</div>` : ""}</div>`;

function renderDashboard() {
  const d = DEMO_DATASET, tag = CONFIG.USE_DEMO_DATA ? "demo data" : "";
  $("#statCards").innerHTML =
    card("Total records", d.total.toLocaleString(), "", tag) +
    card("High-risk records", d.high.toLocaleString(), "high", tag) +
    card("Low-risk records", d.low.toLocaleString(), "low", tag) +
    card("Model status", `<span class="dot"></span>Not connected`, "", "Classifier + regressor pending");
  doughnut("dashChart");
}

// ----- Charts -----
const charts = {};
function make(id, cfg) { charts[id]?.destroy(); charts[id] = new Chart($("#" + id), { ...cfg, options: { responsive: true, maintainAspectRatio: false, ...cfg.options } }); }
function doughnut(id) {
  make(id, { type: "doughnut", data: { labels: ["Low risk", "High risk"], datasets: [{ data: [DEMO_DATASET.low, DEMO_DATASET.high], backgroundColor: [GREEN, RED] }] }, options: { plugins: { legend: { position: "bottom" } } } });
}
function renderDataset() {
  doughnut("riskChart");
  const m = DEMO_DATASET.meanByClass;
  make("meanChart", { type: "bar", data: { labels: m.labels, datasets: [{ label: "Low risk", data: m.low, backgroundColor: GREEN }, { label: "High risk", data: m.high, backgroundColor: RED }] } });
  const c = DEMO_DATASET.chlorophyllBins;
  make("chlChart", { type: "bar", data: { labels: c.labels, datasets: [{ label: "Records (µg/L bins)", data: c.counts, backgroundColor: TEAL }] } });
}

// ----- Performance -----
function renderPerformance() {
  const build = (obj) => Object.entries(obj).map(([k, v]) => card(k, fmt(v), "", v === null ? "not set" : "")).join("");
  $("#regMetrics").innerHTML = build(METRICS.regression);
  $("#clsMetrics").innerHTML = build(METRICS.classification);
}

// ----- Prediction form -----
function buildForm() {
  $("#endpointLabel").textContent = "POST " + CONFIG.API_BASE_URL + CONFIG.PREDICT_PATH;
  $("#numericFields").innerHTML = NUMERIC_FIELDS.map((f) =>
    `<div><label for="${f.key}">${f.label} <span>(${f.unit})</span></label><input id="${f.key}" name="${f.key}" type="number" step="any" required></div>`).join("");
  $("#categoricalFields").innerHTML = CATEGORICAL_FIELDS.map((f) =>
    `<div><label for="${f.key}">${f.label}</label><select id="${f.key}" name="${f.key}" required><option value="">Select…</option>${f.options.map((o) => `<option>${o}</option>`).join("")}</select></div>`).join("");
}

function collectPayload() {
  const p = {};
  NUMERIC_FIELDS.forEach((f) => (p[f.key] = parseFloat($("#" + f.key).value)));
  CATEGORICAL_FIELDS.forEach((f) => (p[f.key] = $("#" + f.key).value));
  return p;
}

function showResult(r, demo = false) {
  const high = String(r.risk_label).toUpperCase() === "HIGH";
  const pct = (v) => (v * 100).toFixed(1) + "%";
  const conc = r.estimated_concentration_mg_kg;
  const concHtml = high
    ? `<div class="conc">Estimated domoic acid concentration<br><b>${conc == null ? "—" : fmt(conc, 2) + " mg/kg"}</b></div>`
    : `<p class="note">A concentration estimate is shown only for high-risk predictions.</p>`;
  const el = $("#result");
  el.hidden = false;
  el.innerHTML = `
    ${demo ? `<div class="demo-banner">DEMO LAYOUT — sample values, not a model output.</div>` : ""}
    <h2>Result</h2>
    <span class="badge ${high ? "high" : "low"}">${high ? "HIGH RISK" : "LOW RISK"}</span>
    <div class="prob"><div class="row"><span>Low-risk probability</span><b>${pct(r.probabilities.low)}</b></div><div class="bar"><i style="width:${pct(r.probabilities.low)};background:${GREEN}"></i></div></div>
    <div class="prob"><div class="row"><span>High-risk probability</span><b>${pct(r.probabilities.high)}</b></div><div class="bar"><i style="width:${pct(r.probabilities.high)};background:${RED}"></i></div></div>
    ${concHtml}
    <p class="note">Model estimate only. It is not a laboratory measurement. Confirm with approved lab testing.</p>`;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("#predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!e.target.reportValidity()) return;
  const btn = $("#predictBtn");
  btn.disabled = true; btn.textContent = "Predicting…";
  try {
    showResult(await predictRisk(collectPayload()));
  } catch (err) {
    const el = $("#result"); el.hidden = false;
    el.innerHTML = `<div class="error">${err instanceof BackendError ? err.message : "Something went wrong."}</div>`;
  } finally { btn.disabled = false; btn.textContent = "Predict Risk"; }
});
$("#previewBtn").onclick = () => showResult(DEMO_RESULT, true);

// ----- Start -----
buildForm(); renderDashboard(); renderDataset(); renderPerformance(); showPage();
