# Smart Grid Renewable Energy Forecasting

This project forecasts solar generation, wind generation, and electricity
demand from hourly smart-grid data. It provides a FastAPI backend and browser
dashboard for data preparation, model training, forecasting, weather context,
rule-based grid recommendations, and CSV/PDF report export.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the dashboard or
`http://127.0.0.1:8000/signin` for the presentation demo sign-in screen.

## Configuration

Set `OPENWEATHER_API_KEY` in `.env` to enable live weather data. The
application uses simulated weather when the key is empty or unavailable.

Set `AUTH_ENABLED=true` and provide a strong `API_KEY` to protect API routes.
The dashboard accepts that key through its session-only API-key field. Never
commit `.env` or expose the API key in frontend source.

`ALLOWED_CORS_ORIGINS` controls permitted browser origins. Rate limits are
configurable through the `RATE_LIMIT_*` variables. The limiter is in-memory:
counters reset on restart, are per worker, and are not suitable for
distributed deployments.

## Models

The supported model types are:

- `xgboost` - gradient-boosted tree regression
- `mlp` - scikit-learn multi-layer perceptron regression

Train models with `POST /api/train` before requesting predictions. Prediction
requests do not train missing models automatically. Training and prediction
share the same feature preparation path, and saved artifacts include feature
schema and validation metadata.

## Data and reports

Upload CSV files through the dashboard or `POST /api/upload`. Input datasets
must contain timestamp, weather, solar power, wind power, and electricity
demand fields (recognized aliases are supported). Generated model artifacts,
processed data, and reports are stored locally and are ignored by Git where
appropriate.

The recommendation engine is rule-based decision support. It does not model
battery state of charge or issue automatic grid-control commands.

## Testing

```bash
pytest -q
```

The final model artifacts should be retrained after installing the pinned
environment used for submission to avoid serialization-version warnings.
