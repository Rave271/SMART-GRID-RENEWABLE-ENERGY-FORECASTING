# Smart Grid Renewable Energy Forecasting

## Final Project Report

**Institution:** Chitkara University, Punjab  
**Student authors:** Harmanjot Singh; Kirpal Singh; Anuj Yadav; Raghav Verma  
**Faculty supervisor:** Dr. Preeti Saini

## Abstract

Smart Grid Renewable Energy Forecasting is a Python-based decision-support platform for forecasting solar generation, wind generation, and electricity demand from hourly smart-grid data. It combines validation, preprocessing, feature engineering, supervised regression, optional weather context, a browser dashboard, rule-based recommendations, and CSV/PDF export. XGBoost is the primary model and scikit-learn MLP is the alternative. Evaluation uses generated/sample data and is not a claim of real-world accuracy.

## 1. Introduction

Renewable generation is variable while electricity demand changes with time, weather, and user activity. This project provides an end-to-end prototype that converts historical energy and weather observations into hourly forecasts and visual decision support.

## 2. Problem Statement and Objectives

The project addresses short-term forecasting of renewable generation and electricity demand using structured historical data. Its objectives are to validate uploaded datasets, create regular hourly data, engineer temporal/weather/lag features, train and persist forecasting models, produce 24-hour or seven-day forecasts, calculate surplus/deficit summaries, and expose the workflow through a REST API and dashboard.

The scope is educational and demonstrational. Automatic grid control, battery state-of-charge modelling, market bidding, protection control, and production deployment guarantees are not implemented.

## 3. Architecture and Technology Stack

The system uses a modular FastAPI backend, service modules for preprocessing, features, modelling, weather, recommendations, and reports, a local SQLite database, and a static HTML/CSS/JavaScript frontend using Chart.js.

Technologies include Python, FastAPI, Uvicorn, Pydantic, Pandas, NumPy, scikit-learn, XGBoost, Joblib, SQLite, Requests, Matplotlib, ReportLab, HTML, CSS, JavaScript, Chart.js, Pytest, and HTTPX-compatible test dependencies.

## 4. Software Modules

- `app/main.py`: application lifecycle, database initialization, static dashboard/sign-in routes, CORS, and router registration.
- Upload routes: bounded CSV upload, validation, safe storage, and dataset metadata.
- Training routes: dataset selection, preprocessing, model training, metric persistence, and metadata.
- Prediction routes: validation, weather alignment, forecasting, and balance summaries.
- Weather routes: current and forecast weather context.
- Recommendation routes: rule-based recommendations and metrics.
- Report routes: CSV and PDF forecast exports.
- `app/services/`: preprocessing, feature engineering, model training/inference, weather, recommendations, and report logic.
- `frontend/`: presentation dashboard, charts, controls, sign-in demo, and report actions.

## 5. Data Preprocessing and Feature Engineering

Uploaded data is mapped to canonical names, timestamps are parsed and sorted, duplicate timestamps are averaged, data is resampled hourly, missing values are interpolated, boundary gaps are filled, night-time solar values are constrained, and physical ranges are clipped. Required fields include timestamp, temperature, humidity, wind speed, solar irradiance, solar power, wind power, and electricity demand.

Features include hour, weekday, month, day of year, weekend and season indicators, cyclical sine/cosine encodings, wind-speed cubed, solar temperature derating, degree-hour variables, target lags at 1, 2, 24, and 48 hours, and six-hour/24-hour rolling statistics. Training and prediction use the same feature-preparation path.

## 6. Forecasting Approaches

XGBoost is the primary tree-based regressor and exposes feature importance values. The alternative is scikit-learn `MLPRegressor` with two hidden layers, ReLU activation, Adam optimization, adaptive learning rate, early stopping, and a deterministic random state. The project does not implement LSTM or recurrent neural networks.

## 7. Training, Evaluation, and Prediction

Training uses a chronological split, with the earliest observations for training and the latest observations for testing. `StandardScaler` is fitted only on training features. Models are evaluated using MAE, RMSE, and R². Persisted artifacts include the estimator, scaler, feature names, metrics, model type, target, timestamp, and dataset-version metadata.

Prediction requires an existing trained artifact and does not silently train missing models. Future values are generated autoregressively so forecast values can update lag and rolling features. Solar predictions are zero during configured night hours and predictions are non-negative.

## 8. Weather Integration

The weather client can request current conditions and a five-day, three-hour OpenWeather forecast using an environment-provided key. Weather values are aligned to hourly timestamps by resampling/interpolation. When no usable key is available, the application can use its documented simulated fallback.

## 9. Renewable Forecasting and Grid Balance

The prediction response includes solar generation, wind generation, electricity demand, total renewable generation, renewable penetration, and surplus/deficit calculations. These summaries support visual interpretation of expected grid balance but are not an automated dispatch system.

## 10. Rule-Based Recommendations

Recommendations are deterministic heuristics based on forecast renewable generation, demand, surplus, deficit, and curtailment thresholds. They are presented as recommendations only. Battery state of charge, degradation, charge/discharge constraints, economic dispatch, and automatic control are not modelled.

## 11. Database and API Design

SQLite stores dataset metadata, trained-model metadata and metrics, and prediction-history summaries. FastAPI routes use typed request/response schemas. Optional API-key protection uses the `X-API-Key` header and constant-time comparison. If authentication is enabled without a configured key, startup fails. Upload, training, prediction, recommendation, and report operations have configurable in-memory rate limits. CORS is configured through `ALLOWED_CORS_ORIGINS`.

## 12. Frontend and Dashboard

The dashboard provides presentation-oriented status information, model and weather controls, upload/training actions, forecast charts, balance summaries, recommendation output, and CSV/PDF report actions. The sign-in page is a presentation demo; it is not a production account or identity system. API keys are entered for the browser session rather than embedded in frontend source.

## 13. Testing and Results

The project test suite covers model training, autoregressive prediction, target-specific output, persistence/reload, feature consistency, authentication, uploads, rate limits, reports, and weather integration. The verified suite completed with 19 passing tests. Sample-data metrics were approximately: solar XGBoost R² 0.9993, solar MLP 0.9991, wind XGBoost 0.9983, wind MLP 0.9978, demand XGBoost 0.9830, and demand MLP 0.9770. These are generated/sample-data results and must not be interpreted as real-world accuracy.

## 14. Limitations and Conclusion

The application is an educational prototype. In-memory rate limiting resets on restart, is per worker, and is unsuitable for distributed deployment. The weather fallback is simulated when live data is unavailable. Forecast quality has not been established on a representative operational grid dataset. Battery state, dispatch optimization, automated control, production deployment, and real-world accuracy claims are outside the current implementation.

The completed prototype demonstrates a coherent pipeline from data upload and preparation through model training, weather alignment, forecasting, balance analysis, recommendations, and report export.

## References

1. FastAPI documentation: https://fastapi.tiangolo.com/
2. XGBoost documentation: https://xgboost.readthedocs.io/
3. scikit-learn documentation: https://scikit-learn.org/
4. OpenWeather API documentation: https://openweathermap.org/api
5. ReportLab documentation: https://docs.reportlab.com/
