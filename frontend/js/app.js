/**
 * Smart Grid Renewable Energy Forecasting - Frontend Controller
 * Vanilla ES6+, Chart.js integration, responsive dark mode, real-time AJAX.
 */

// Global Chart Instances
let chartOverlay = null;
let chartRenewables = null;
let chartImportance = null;

// Application State
const state = {
  theme: localStorage.getItem("sg_theme") || "light",
  horizon: "24h",
  modelType: "xgboost",
  city: "London",
  currentPredictions: null,
};

function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const apiKey = sessionStorage.getItem("sg_api_key");
  if (apiKey) headers.set("X-API-Key", apiKey);
  return fetch(url, { ...options, headers });
}

// DOM Content Loaded Handler
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  const keyInput = document.getElementById("api-key-input");
  if (keyInput) keyInput.value = sessionStorage.getItem("sg_api_key") || "";
  document.getElementById("save-api-key")?.addEventListener("click", () => {
    sessionStorage.setItem("sg_api_key", keyInput?.value.trim() || "");
    showToast("API key saved for this browser session.", "success");
  });
  document.getElementById("signout-link")?.addEventListener("click", () => {
    localStorage.removeItem("sg_demo_session");
    sessionStorage.removeItem("sg_demo_session");
  });
  initClock();
  initEventListeners();
  initCharts();
  document.getElementById("hero-refresh")?.addEventListener("click", () => {
    loadForecastAndRecommendations();
    document.getElementById("forecast-overlay")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // Initial Data Bootstrap
  fetchWeather(state.city);
  fetchMetrics();
  loadForecastAndRecommendations();
});

/* -------------------------------------------------------------
 * 1. THEME & LIVE CLOCK
 * -----------------------------------------------------------*/
function initTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
  updateThemeButtonUI();

  const toggleBtn = document.getElementById("theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      state.theme = state.theme === "light" ? "dark" : "light";
      localStorage.setItem("sg_theme", state.theme);
      document.documentElement.setAttribute("data-theme", state.theme);
      updateThemeButtonUI();
      // Update charts for theme contrast
      updateChartsTheme();
    });
  }
}

function updateThemeButtonUI() {
  const icon = document.getElementById("theme-icon");
  const text = document.getElementById("theme-text");
  if (icon && text) {
    if (state.theme === "dark") {
      icon.textContent = "☀️";
      text.textContent = "Light Mode";
    } else {
      icon.textContent = "🌙";
      text.textContent = "Dark Mode";
    }
  }
}

function initClock() {
  const clockEl = document.getElementById("live-clock");
  if (!clockEl) return;
  const update = () => {
    const now = new Date();
    clockEl.textContent = now.toUTCString().replace("GMT", "UTC");
  };
  update();
  setInterval(update, 1000);
}

/* -------------------------------------------------------------
 * 2. TOAST NOTIFICATIONS
 * -----------------------------------------------------------*/
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const iconMap = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    warning: "⚠️",
  };

  toast.innerHTML = `
    <span>${iconMap[type] || "ℹ️"}</span>
    <div style="flex: 1;">${message}</div>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

/* -------------------------------------------------------------
 * 3. EVENT LISTENERS
 * -----------------------------------------------------------*/
function initEventListeners() {
  // Dropzone file upload
  const dropzone = document.getElementById("csv-dropzone");
  const fileInput = document.getElementById("csv-file-input");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) {
        handleFileUpload(fileInput.files[0]);
      }
    });
  }

  // Preprocess Button
  const btnPreprocess = document.getElementById("btn-preprocess");
  if (btnPreprocess) {
    btnPreprocess.addEventListener("click", handlePreprocess);
  }

  // Reload Sample Button
  const btnReloadSample = document.getElementById("btn-load-sample");
  if (btnReloadSample) {
    btnReloadSample.addEventListener("click", handlePreprocess);
  }

  // Weather Fetch
  const btnFetchWeather = document.getElementById("btn-fetch-weather");
  const cityInput = document.getElementById("city-input");
  if (btnFetchWeather && cityInput) {
    btnFetchWeather.addEventListener("click", () => {
      const city = cityInput.value.trim();
      if (city) {
        state.city = city;
        fetchWeather(city);
      }
    });
    cityInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        btnFetchWeather.click();
      }
    });
  }

  // Weather Presets
  document.querySelectorAll(".city-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const city = btn.getAttribute("data-city");
      if (city) {
        document.getElementById("city-input").value = city;
        state.city = city;
        fetchWeather(city);
      }
    });
  });

  // Train Model
  const btnTrain = document.getElementById("btn-train-model");
  if (btnTrain) {
    btnTrain.addEventListener("click", handleTrainModel);
  }

  // Forecast Horizon and Refresh
  const horizonSelect = document.getElementById("forecast-horizon-select");
  if (horizonSelect) {
    horizonSelect.addEventListener("change", (e) => {
      state.horizon = e.target.value;
      document.getElementById("status-horizon").textContent =
        e.target.value === "7d" ? "Next 7 days" : "Next 24 hours";
      loadForecastAndRecommendations();
    });
  }

  const btnRefreshForecast = document.getElementById("btn-refresh-forecast");
  if (btnRefreshForecast) {
    btnRefreshForecast.addEventListener("click", loadForecastAndRecommendations);
  }

  // Model Type Selector
  const modelTypeSelect = document.getElementById("select-model-type");
  if (modelTypeSelect) {
    modelTypeSelect.addEventListener("change", (e) => {
      state.modelType = e.target.value;
      document.getElementById("status-model").textContent = `${e.target.value.toUpperCase()} selected`;
    });
  }

  // Reports Export
  const btnExportPdf = document.getElementById("btn-export-pdf");
  if (btnExportPdf) {
    btnExportPdf.addEventListener("click", () => downloadReport("pdf"));
  }

  const btnExportCsv = document.getElementById("btn-export-csv");
  if (btnExportCsv) {
    btnExportCsv.addEventListener("click", () => downloadReport("csv"));
  }
}

async function downloadReport(format) {
  try {
    const url = `/api/report?format=${format}&horizon=${state.horizon}&model_type=${state.modelType}`;
    const response = await apiFetch(url);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Report generation failed.");
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `smart_grid_forecast.${format}`;
    link.click();
    URL.revokeObjectURL(link.href);
  } catch (err) {
    showToast(err.message, "error");
  }
}

/* -------------------------------------------------------------
 * 4. DATA UPLOAD & PREPROCESSING
 * -----------------------------------------------------------*/
async function handleFileUpload(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showToast("Please upload a valid CSV file.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const statusBadge = document.getElementById("upload-status-badge");
  statusBadge.className = "badge badge-warning";
  statusBadge.textContent = "Uploading & Validating...";

  try {
    const res = await apiFetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail?.message || data.detail || "Upload validation failed.");
    }

    statusBadge.className = "badge badge-success";
    statusBadge.textContent = `Uploaded: ${data.filename}`;
    showToast(`Dataset validated successfully (${data.row_count} rows).`, "success");

    renderPreviewTable(data.columns, data.preview);
  } catch (err) {
    statusBadge.className = "badge badge-danger";
    statusBadge.textContent = "Upload Failed";
    showToast(err.message, "error");
  }
}

function renderPreviewTable(columns, previewData) {
  const container = document.getElementById("upload-preview-container");
  const table = document.getElementById("preview-table");
  if (!container || !table || !previewData || !previewData.length) return;

  container.style.display = "block";
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  thead.innerHTML = `<tr>${columns.slice(0, 7).map((c) => `<th>${c}</th>`).join("")}</tr>`;
  tbody.innerHTML = previewData
    .map(
      (row) => `<tr>${columns.slice(0, 7).map((c) => `<td>${row[c] !== undefined ? row[c] : "-"}</td>`).join("")}</tr>`
    )
    .join("");
}

async function handlePreprocess() {
  const btn = document.getElementById("btn-preprocess");
  const originalHtml = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> <span>Cleaning...</span>`;
  btn.disabled = true;

  try {
    const res = await apiFetch("/api/preprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resample_freq: "1h", clip_outliers: true }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Preprocessing failed.");

    showToast(`Data cleaned: ${data.processed_rows} hourly records ready for ML.`, "success");
    renderPreviewTable(data.columns, data.sample_preview);
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btn.innerHTML = originalHtml;
    btn.disabled = false;
  }
}

/* -------------------------------------------------------------
 * 5. WEATHER INTEGRATION
 * -----------------------------------------------------------*/
async function fetchWeather(city) {
  const badge = document.getElementById("weather-source-badge");
  const noteEl = document.getElementById("weather-note");

  try {
    const res = await apiFetch(`/api/weather?city=${encodeURIComponent(city)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to fetch weather.");

    if (data.is_live_key) {
      badge.className = "badge badge-success";
      badge.textContent = "OpenWeather Live";
      noteEl.textContent = `Live atmospheric data for ${data.city}.`;
      document.getElementById("status-weather").textContent = "Live OpenWeather feed";
    } else {
      badge.className = "badge badge-warning";
      badge.textContent = "Simulated Fallback";
      noteEl.textContent = "Note: Operating on high-fidelity calibrated simulation. Add OPENWEATHER_API_KEY to .env for live API calls.";
      document.getElementById("status-weather").textContent = "Simulation fallback active";
    }

    const curr = data.current;
    document.getElementById("weather-temp").textContent = `${curr.temperature}°C`;
    document.getElementById("weather-hum").textContent = `${curr.humidity}%`;
    document.getElementById("weather-wind").textContent = `${curr.wind_speed} m/s`;
    document.getElementById("weather-irr").textContent = `${curr.solar_irradiance} W/m²`;
  } catch (err) {
    badge.className = "badge badge-danger";
    badge.textContent = "Weather Offline";
    showToast(err.message, "error");
    document.getElementById("status-weather").textContent = "Connection unavailable";
  }
}

/* -------------------------------------------------------------
 * 6. MODEL TRAINING
 * -----------------------------------------------------------*/
async function handleTrainModel() {
  const btn = document.getElementById("btn-train-model");
  const btnText = document.getElementById("train-btn-text");
  const modelType = document.getElementById("select-model-type").value;
  const target = document.getElementById("select-target").value;

  btnText.innerHTML = `<span class="spinner"></span> <span>Training ${modelType.toUpperCase()}...</span>`;
  btn.disabled = true;

  try {
    const res = await apiFetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_type: modelType,
        target: target,
        test_size: 0.2,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Training failed.");

    showToast(`Trained ${data.results.length} model(s) successfully!`, "success");
    renderMetricsTable(data.results);
    updateFeatureImportanceChart(data.results[0]?.feature_importance || {});

    // Update KPI accuracy
    const avgMae = (data.results.reduce((acc, r) => acc + r.mae, 0) / data.results.length).toFixed(2);
    const avgRmse = (data.results.reduce((acc, r) => acc + r.rmse, 0) / data.results.length).toFixed(2);
    const avgR2 = (data.results.reduce((acc, r) => acc + r.r2, 0) / data.results.length).toFixed(3);

    document.getElementById("kpi-mae").textContent = `${avgMae} MW`;
    document.getElementById("kpi-rmse").textContent = `RMSE: ${avgRmse} MW | R²: ${avgR2}`;

    // Reload forecasts with new model weights
    state.modelType = modelType;
    document.getElementById("status-model").textContent = `${modelType.toUpperCase()} active`;
    loadForecastAndRecommendations();
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btnText.textContent = "🚀 Train Forecasting Models";
    btn.disabled = false;
  }
}

async function fetchMetrics() {
  try {
    const res = await apiFetch("/api/metrics");
    const data = await res.json();
    if (res.ok && data.models && data.models.length > 0) {
      renderMetricsTable(data.models);
      const topModel = data.models[0];
      document.getElementById("kpi-mae").textContent = `${topModel.mae} MW`;
      document.getElementById("kpi-rmse").textContent = `RMSE: ${topModel.rmse} MW | R²: ${topModel.r2}`;
    }
  } catch (e) {
    console.warn("Could not fetch metrics:", e);
  }
}

function renderMetricsTable(modelsList) {
  const tbody = document.getElementById("metrics-table-body");
  if (!tbody || !modelsList.length) return;

  const targetNames = {
    solar_power_mw: "Solar Generation",
    wind_power_mw: "Wind Generation",
    electricity_demand_mw: "Grid Electricity Demand",
  };

  tbody.innerHTML = modelsList
    .map((m) => {
      const r2Badge =
        m.r2 > 0.85
          ? '<span class="badge badge-success">Excellent</span>'
          : m.r2 > 0.70
          ? '<span class="badge badge-info">Good</span>'
          : '<span class="badge badge-warning">Moderate</span>';

      return `
      <tr>
        <td><strong>${targetNames[m.target] || m.target}</strong></td>
        <td><span class="badge badge-info">${(m.model_type || "xgboost").toUpperCase()}</span></td>
        <td>${Number(m.mae).toFixed(2)}</td>
        <td>${Number(m.rmse).toFixed(2)}</td>
        <td><strong>${Number(m.r2).toFixed(3)}</strong></td>
        <td>${m.train_samples ? `${m.train_samples} / ${m.test_samples}` : "80% / 20%"}</td>
        <td>${r2Badge}</td>
      </tr>
    `;
    })
    .join("");
}

/* -------------------------------------------------------------
 * 7. FORECAST & RECOMMENDATIONS PIPELINE
 * -----------------------------------------------------------*/
async function loadForecastAndRecommendations() {
  try {
    const [predRes, recomRes] = await Promise.all([
      apiFetch(`/api/predict?horizon=${state.horizon}&target=all&model_type=${state.modelType}&city=${encodeURIComponent(state.city)}`),
      apiFetch(`/api/recommendations?horizon=${state.horizon}&model_type=${state.modelType}&city=${encodeURIComponent(state.city)}`),
    ]);

    const predData = await predRes.json();
    const recomData = await recomRes.json();

    if (!predRes.ok) throw new Error(predData.detail || "Prediction fetch failed.");

    state.currentPredictions = predData;
    updateKpis(predData.summary);
    updateForecastCharts(predData);

    if (recomRes.ok) {
      renderRecommendations(recomData);
    }
  } catch (err) {
    showToast(err.message, "error");
  }
}

function updateKpis(summary) {
  if (!summary) return;

  document.getElementById("kpi-solar").textContent = `${summary.total_solar_mwh.toLocaleString()} MWh`;
  document.getElementById("kpi-wind").textContent = `${summary.total_wind_mwh.toLocaleString()} MWh`;
  document.getElementById("kpi-demand").textContent = `${summary.total_demand_mwh.toLocaleString()} MWh`;

  const balanceEl = document.getElementById("kpi-balance");
  const balanceCard = balanceEl.closest(".kpi-card");
  const balance = summary.net_balance_mwh;

  if (balance >= 0) {
    balanceEl.textContent = `+${balance.toLocaleString()} MWh`;
    balanceEl.style.color = "var(--accent-surplus)";
    balanceCard.className = "kpi-card balance";
  } else {
    balanceEl.textContent = `${balance.toLocaleString()} MWh`;
    balanceEl.style.color = "var(--accent-deficit)";
    balanceCard.className = "kpi-card demand";
  }

  document.getElementById("kpi-penetration").textContent = `Clean Energy Penetration: ${summary.avg_renewable_penetration_pct}%`;
}

function renderRecommendations(recomData) {
  const container = document.getElementById("recommendations-container");
  const badge = document.getElementById("recom-horizon-badge");
  if (!container) return;

  if (badge) badge.textContent = `${recomData.horizon} Plan`;

  if (!recomData.recommendations || !recomData.recommendations.length) {
    container.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 20px;">No operational dispatch alarms for this window.</div>`;
    return;
  }

  container.innerHTML = recomData.recommendations
    .map(
      (r) => `
    <div class="recom-item ${r.severity}">
      <div class="recom-header">
        <span class="recom-title">${r.title}</span>
        <span class="recom-time">${r.timestamp}</span>
      </div>
      <p class="recom-desc">${r.description}</p>
      <div class="recom-action">⚡ Recommendation: ${r.action}</div>
    </div>
  `
    )
    .join("");
}

/* -------------------------------------------------------------
 * 8. CHART.JS VISUALIZATIONS
 * -----------------------------------------------------------*/
function initCharts() {
  const isDark = state.theme === "dark";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
  const textColor = isDark ? "#94a3b8" : "#64748b";

  // Chart 1: Overlay (Generation vs Demand)
  const ctxOverlay = document.getElementById("chart-overlay")?.getContext("2d");
  if (ctxOverlay) {
    chartOverlay = new Chart(ctxOverlay, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Total Renewable Generation (MW)",
            data: [],
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.15)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            pointRadius: 2,
          },
          {
            label: "Electricity Demand (MW)",
            data: [],
            borderColor: "#ef4444",
            backgroundColor: "rgba(239, 68, 68, 0.1)",
            borderWidth: 2.5,
            borderDash: [5, 5],
            fill: false,
            tension: 0.35,
            pointRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top", labels: { color: textColor } },
          tooltip: {
            callbacks: {
              footer: (items) => {
                const gen = items[0]?.parsed?.y || 0;
                const dem = items[1]?.parsed?.y || 0;
                const bal = gen - dem;
                return `Net Balance: ${bal >= 0 ? "+" : ""}${bal.toFixed(1)} MW (${bal >= 0 ? "Surplus" : "Deficit"})`;
              },
            },
          },
        },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: textColor, maxTicksLimit: 12 } },
          y: { grid: { color: gridColor }, ticks: { color: textColor }, title: { display: true, text: "Power (MW)", color: textColor } },
        },
      },
    });
  }

  // Chart 2: Renewables Breakdown
  const ctxRenewables = document.getElementById("chart-renewables")?.getContext("2d");
  if (ctxRenewables) {
    chartRenewables = new Chart(ctxRenewables, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Solar Power (MW)",
            data: [],
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.2)",
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 1,
          },
          {
            label: "Wind Power (MW)",
            data: [],
            borderColor: "#0284c7",
            backgroundColor: "rgba(2, 132, 199, 0.2)",
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { color: textColor } } },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: textColor, maxTicksLimit: 8 } },
          y: { grid: { color: gridColor }, ticks: { color: textColor } },
        },
      },
    });
  }

  // Chart 3: Feature Importance
  const ctxImportance = document.getElementById("chart-importance")?.getContext("2d");
  if (ctxImportance) {
    chartImportance = new Chart(ctxImportance, {
      type: "bar",
      data: {
        labels: ["Hour", "Solar Irradiance", "Wind Speed", "Lag-1", "Lag-24", "Temperature"],
        datasets: [
          {
            label: "Relative Importance (%)",
            data: [28, 24, 20, 14, 9, 5],
            backgroundColor: [
              "#10b981",
              "#f59e0b",
              "#0284c7",
              "#8b5cf6",
              "#6366f1",
              "#ec4899",
            ],
            borderRadius: 4,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: textColor } },
          y: { grid: { display: false }, ticks: { color: textColor } },
        },
      },
    });
  }
}

function updateForecastCharts(predData) {
  if (!predData || !predData.predictions) return;

  const preds = predData.predictions;
  const labels = preds.map((p) => {
    const parts = p.timestamp.split(" ");
    return parts.length > 1 ? parts[1].substring(0, 5) : p.timestamp;
  });

  const totalGen = preds.map((p) => p.predicted_generation);
  const demand = preds.map((p) => p.predicted_demand);
  const solar = preds.map((p) => p.predicted_solar);
  const wind = preds.map((p) => p.predicted_wind);

  if (chartOverlay) {
    chartOverlay.data.labels = labels;
    chartOverlay.data.datasets[0].data = totalGen;
    chartOverlay.data.datasets[1].data = demand;
    chartOverlay.update();
  }

  if (chartRenewables) {
    chartRenewables.data.labels = labels;
    chartRenewables.data.datasets[0].data = solar;
    chartRenewables.data.datasets[1].data = wind;
    chartRenewables.update();
  }
}

function updateFeatureImportanceChart(importanceDict) {
  if (!chartImportance || !importanceDict || Object.keys(importanceDict).length === 0) return;

  const labels = Object.keys(importanceDict).slice(0, 8);
  const values = labels.map((k) => importanceDict[k]);

  chartImportance.data.labels = labels;
  chartImportance.data.datasets[0].data = values;
  chartImportance.update();
}

function updateChartsTheme() {
  const isDark = state.theme === "dark";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
  const textColor = isDark ? "#94a3b8" : "#64748b";

  [chartOverlay, chartRenewables, chartImportance].forEach((chart) => {
    if (!chart) return;
    if (chart.options.scales.x) {
      chart.options.scales.x.grid.color = gridColor;
      chart.options.scales.x.ticks.color = textColor;
    }
    if (chart.options.scales.y) {
      chart.options.scales.y.grid.color = gridColor;
      chart.options.scales.y.ticks.color = textColor;
    }
    if (chart.options.plugins?.legend?.labels) {
      chart.options.plugins.legend.labels.color = textColor;
    }
    chart.update();
  });
}
