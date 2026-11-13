"""Machine Learning training, validation, and multi-step forecasting engine."""

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

from app.config import MODELS_DIR
from app.services.features import prepare_features

TARGETS = ["solar_power_mw", "wind_power_mw", "electricity_demand_mw"]
MODEL_TYPES = ["xgboost", "mlp"]


class GridForecaster:
    """Manages training, metrics, persistence, and autoregressive forecasting."""

    def __init__(self, target: str = "solar_power_mw", model_type: str = "xgboost"):
        self.target = target
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def _init_estimator(self):
        """Initialize the model estimator based on requested model_type."""
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=2,
            )
        elif self.model_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=64,
                learning_rate="adaptive",
                learning_rate_init=0.005,
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=8,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Must be one of {MODEL_TYPES}.")

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train model with strict chronological time-based split (NO random shuffle)
        to prevent future data leakage. Scaler is fitted strictly on training data.
        """
        # Feature Engineering
        featured_df, feature_cols = prepare_features(df, target_col=self.target, include_lags=True)
        self.feature_names = feature_cols

        X = featured_df[feature_cols].copy()
        y = featured_df[self.target].copy()

        # Strict chronological time-based train/test split
        n_samples = len(featured_df)
        split_idx = int(n_samples * (1.0 - test_size))

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Fit Scaler strictly on X_train to prevent leakage
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train estimator
        estimator = self._init_estimator()
        estimator.fit(X_train_scaled, y_train)
        self.model = estimator

        # Evaluate predictions on unseen chronological test partition
        y_pred = self.model.predict(X_test_scaled)

        # Enforce physical post-processing bounds
        y_pred = np.maximum(0.0, y_pred)
        if self.target == "solar_power_mw":
            # Zero out at night hours
            hours = featured_df["hour"].iloc[split_idx:].values
            night_mask = (hours >= 20) | (hours < 6)
            y_pred[night_mask] = 0.0

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        # Compute feature importances
        importance_dict = {}
        if hasattr(self.model, "feature_importances_"):
            raw_importances = self.model.feature_importances_
            # Normalize to percentage
            total = np.sum(raw_importances) or 1.0
            norm_imp = (raw_importances / total) * 100.0
            # Sort top 15 features
            sorted_indices = np.argsort(norm_imp)[::-1][:15]
            for idx in sorted_indices:
                importance_dict[self.feature_names[idx]] = round(float(norm_imp[idx]), 2)
        else:
            # Equal / weight proxy for neural network
            importance_dict = {feat: round(100.0 / len(self.feature_names), 2) for feat in self.feature_names[:10]}

        self.metrics = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importance": importance_dict,
        }
        self.metadata = {
            "model_type": self.model_type,
            "target": self.target,
            "feature_names": list(self.feature_names),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": "unspecified",
            "validation_metrics": dict(self.metrics),
        }

        # Auto-save model artifact
        self.save()

        return {
            "target": self.target,
            "model_type": self.model_type,
            **self.metrics,
            "model_path": str(self.get_model_path()),
        }

    def get_model_path(self) -> Path:
        """Standardized model file path."""
        return MODELS_DIR / f"{self.target}_{self.model_type}.joblib"

    def save(self):
        """Serialize model, scaler, and metadata."""
        path = self.get_model_path()
        payload = {
            "target": self.target,
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
        joblib.dump(payload, path)

    def load(self) -> bool:
        """Load model from disk."""
        path = self.get_model_path()
        if not path.exists():
            return False
        payload = joblib.load(path)
        self.target = payload["target"]
        self.model_type = payload["model_type"]
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self.metrics = payload.get("metrics", {})
        self.metadata = payload.get("metadata", {
            "model_type": self.model_type,
            "target": self.target,
            "feature_names": list(self.feature_names),
            "dataset_version": "legacy",
        })
        return True

    def predict_future(
        self,
        historical_df: pd.DataFrame,
        horizon_hours: int = 24,
        future_weather: Optional[pd.DataFrame] = None,
    ) -> List[float]:
        """Multi-step future forecasting with iterative autoregressive lag updating."""
        if self.model is None or self.scaler is None:
            if not self.load():
                raise RuntimeError(f"Model for {self.target} ({self.model_type}) is not trained yet.")

        # Ensure datetime on historical df
        df = historical_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp").reset_index(drop=True)

        last_timestamp = df["timestamp"].iloc[-1]
        future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")

        # Prepare future weather dataframe
        if future_weather is not None and len(future_weather) >= horizon_hours:
            fw = future_weather.head(horizon_hours).copy()
        else:
            # Extrapolate synthetic diurnal weather pattern from the final days of historical df
            fw = pd.DataFrame({"timestamp": future_dates})
            for col in ["temperature", "humidity", "wind_speed", "solar_irradiance"]:
                if col in df.columns:
                    # Match by hour of day from recent week
                    recent = df.tail(168)
                    hour_map = recent.groupby(recent["timestamp"].dt.hour)[col].mean()
                    fw[col] = [hour_map.get(h, recent[col].mean()) for h in future_dates.hour]
                else:
                    fw[col] = 0.0

        fw["timestamp"] = future_dates

        # Combine historical and future placeholders for smooth lag calculation
        full_df = pd.concat([df.tail(200), fw], ignore_index=True)

        # Pre-engineer static temporal, cyclical, and weather features for the combined window ONCE
        featured_df, _ = prepare_features(full_df.copy(), target_col=self.target, include_lags=True)
        missing_features = [name for name in self.feature_names if name not in featured_df.columns]
        if missing_features:
            raise RuntimeError(f"Feature schema mismatch. Missing features: {missing_features}")

        # Autoregressive multi-step rolling inference
        predictions = []
        start_idx = len(df.tail(200))

        lag_1_col = f"{self.target}_lag_1"
        lag_2_col = f"{self.target}_lag_2"
        lag_24_col = f"{self.target}_lag_24"
        lag_48_col = f"{self.target}_lag_48"
        roll_6_mean = f"{self.target}_rolling_mean_6h"
        roll_6_std = f"{self.target}_rolling_std_6h"
        roll_24_mean = f"{self.target}_rolling_mean_24h"
        roll_24_std = f"{self.target}_rolling_std_24h"

        for step in range(horizon_hours):
            current_idx = start_idx + step

            # Efficiently update dynamic lag and rolling values from autoregressive predictions
            if lag_1_col in featured_df.columns and current_idx - 1 >= 0:
                featured_df.loc[current_idx, lag_1_col] = full_df.loc[current_idx - 1, self.target]
            if lag_2_col in featured_df.columns and current_idx - 2 >= 0:
                featured_df.loc[current_idx, lag_2_col] = full_df.loc[current_idx - 2, self.target]
            if lag_24_col in featured_df.columns and current_idx - 24 >= 0:
                featured_df.loc[current_idx, lag_24_col] = full_df.loc[current_idx - 24, self.target]
            if lag_48_col in featured_df.columns and current_idx - 48 >= 0:
                featured_df.loc[current_idx, lag_48_col] = full_df.loc[current_idx - 48, self.target]

            if roll_6_mean in featured_df.columns and current_idx - 6 >= 0:
                window_6 = full_df.loc[current_idx - 6 : current_idx - 1, self.target]
                featured_df.loc[current_idx, roll_6_mean] = window_6.mean()
                featured_df.loc[current_idx, roll_6_std] = float(window_6.std()) if len(window_6) > 1 else 0.0

            if roll_24_mean in featured_df.columns and current_idx - 24 >= 0:
                window_24 = full_df.loc[current_idx - 24 : current_idx - 1, self.target]
                featured_df.loc[current_idx, roll_24_mean] = window_24.mean()
                featured_df.loc[current_idx, roll_24_std] = float(window_24.std()) if len(window_24) > 1 else 0.0

            # Extract feature row for the current step
            current_row = featured_df.loc[current_idx : current_idx, self.feature_names]
            current_row_scaled = self.scaler.transform(current_row)

            # Predict step
            pred_val = float(self.model.predict(current_row_scaled)[0])
            pred_val = max(0.0, pred_val)

            # Domain adjustment: solar is strictly 0 at night
            h = future_dates[step].hour
            if self.target == "solar_power_mw" and (h >= 20 or h < 6):
                pred_val = 0.0

            predictions.append(round(pred_val, 2))

            # Update the full_df with this prediction to feed subsequent lag steps
            full_df.loc[current_idx, self.target] = pred_val

        return predictions


def train_all_models(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    test_size: float = 0.2,
) -> List[Dict[str, Any]]:
    """Train forecasters for solar, wind, and demand."""
    results = []
    for target in TARGETS:
        forecaster = GridForecaster(target=target, model_type=model_type)
        res = forecaster.train(df, test_size=test_size)
        results.append(res)
    return results
