"""
Production Inference for Two-Stage Load Forecast Error Models
==============================================================
Makes predictions using trained models for all quarters (Q0-Q3) and horizons (H+1 to H+5).

Usage:
    from inference import LoadForecastPredictor

    predictor = LoadForecastPredictor('models/')
    predictions = predictor.predict(
        quarter=2,  # 0=xx:00, 1=xx:15, 2=xx:30, 3=xx:45
        error_history=[...],  # Last 48+ hourly errors
        forecast_history=[...],  # Last 48+ hourly forecasts
        partial_3min_load=[...],  # 3-min load data for H+1 (0-15 values)
        hour=14,  # Current hour
        dow=2,  # Day of week (0=Monday)
    )
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


class LoadForecastPredictor:
    """Production predictor for two-stage load forecast error models."""

    def __init__(self, models_dir: str):
        """
        Initialize predictor with trained models.

        Args:
            models_dir: Path to directory containing model files
        """
        self.models_dir = Path(models_dir)

        # Load models
        self.s1_models = {}
        self.s2_models = {}
        for quarter in range(4):
            for horizon in range(1, 6):
                self.s1_models[(quarter, horizon)] = joblib.load(
                    self.models_dir / f's1_q{quarter}_h{horizon}.joblib'
                )
                self.s2_models[(quarter, horizon)] = joblib.load(
                    self.models_dir / f's2_q{quarter}_h{horizon}.joblib'
                )

        # Load feature configs
        with open(self.models_dir / 'feature_configs.json') as f:
            self.feature_configs = json.load(f)

        # Load seasonal bias
        with open(self.models_dir / 'seasonal_bias.json') as f:
            self.seasonal_bias = json.load(f)

        print(f"Loaded {len(self.s1_models)} S1 models and {len(self.s2_models)} S2 models")

    def _build_base_features(self,
                             error_history: List[float],
                             forecast_history: List[float],
                             hour: int,
                             dow: int) -> Dict[str, float]:
        """Build base Stage 1 features available at all quarters."""
        features = {}

        # Error lags
        for i in range(min(9, len(error_history))):
            features[f'error_lag{i+1}'] = error_history[i]

        # Rolling statistics
        if len(error_history) >= 3:
            features['error_roll_mean_3h'] = np.mean(error_history[:3])
            features['error_roll_std_3h'] = np.std(error_history[:3])
        if len(error_history) >= 6:
            features['error_roll_mean_6h'] = np.mean(error_history[:6])
            features['error_roll_std_6h'] = np.std(error_history[:6])
        if len(error_history) >= 12:
            features['error_roll_mean_12h'] = np.mean(error_history[:12])
            features['error_roll_std_12h'] = np.std(error_history[:12])
        if len(error_history) >= 24:
            features['error_roll_mean_24h'] = np.mean(error_history[:24])
            features['error_roll_std_24h'] = np.std(error_history[:24])

        # Error trends
        if len(error_history) >= 3:
            features['error_trend_3h'] = error_history[0] - error_history[2]
        if len(error_history) >= 6:
            features['error_trend_6h'] = error_history[0] - error_history[5]
        if len(error_history) >= 4:
            features['error_momentum'] = (
                0.5 * (error_history[0] - error_history[1]) +
                0.3 * (error_history[1] - error_history[2]) +
                0.2 * (error_history[2] - error_history[3])
            )

        # Time features
        features['hour'] = hour
        features['dow'] = dow
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['is_weekend'] = 1 if dow >= 5 else 0

        # Placeholder for features that need external data
        features['load_volatility_lag1'] = 0
        features['load_trend_lag1'] = 0
        features['seasonal_error'] = 0

        for lag in range(1, 4):
            features[f'reg_mean_lag{lag}'] = 0
        features['reg_std_lag1'] = 0

        return features

    def _compute_extrap_features(self,
                                 partial_3min_load: List[float],
                                 minutes: int,
                                 h1_hour: int,
                                 h1_forecast: float) -> tuple:
        """Compute extrapolation features from partial 3-min data."""
        if len(partial_3min_load) == 0:
            return 0, 0, 0

        extrap_load = np.mean(partial_3min_load)

        # Apply seasonal bias
        bias = self.seasonal_bias.get(str(minutes), {}).get(str(h1_hour), 0)
        corrected_load = extrap_load + float(bias)
        extrap_error = corrected_load - h1_forecast

        # Trend
        if len(partial_3min_load) >= 2:
            trend = np.polyfit(range(len(partial_3min_load)), partial_3min_load, 1)[0]
        else:
            trend = 0

        # Volatility
        vol = np.std(partial_3min_load) if len(partial_3min_load) > 1 else 0

        return extrap_error, trend, vol

    def _add_quarter_features(self,
                              features: Dict[str, float],
                              quarter: int,
                              partial_3min_load: List[float],
                              h1_hour: int,
                              h1_forecast: float) -> Dict[str, float]:
        """Add quarter-specific extrapolation features."""
        features = features.copy()

        # Q1 features (15 min)
        if quarter >= 1:
            n_readings = min(5, len(partial_3min_load))  # 5 readings for 15 min
            if n_readings > 0:
                extrap, trend, vol = self._compute_extrap_features(
                    partial_3min_load[:n_readings], 15, h1_hour, h1_forecast
                )
                features['extrap_h1_error_q1'] = extrap
                features['trend_q1'] = trend
                features['vol_q1'] = vol

        # Q2 features (30 min)
        if quarter >= 2:
            n_readings = min(10, len(partial_3min_load))  # 10 readings for 30 min
            if n_readings > 0:
                extrap, trend, vol = self._compute_extrap_features(
                    partial_3min_load[:n_readings], 30, h1_hour, h1_forecast
                )
                features['extrap_h1_error_q2'] = extrap
                features['trend_q2'] = trend
                features['vol_q2'] = vol
                features['delta_est_q1_q2'] = features.get('extrap_h1_error_q2', 0) - features.get('extrap_h1_error_q1', 0)
                features['trend_change_q1_q2'] = features.get('trend_q2', 0) - features.get('trend_q1', 0)

        # Q3 features (45 min)
        if quarter == 3:
            n_readings = min(15, len(partial_3min_load))  # 15 readings for 45 min
            if n_readings > 0:
                extrap, trend, vol = self._compute_extrap_features(
                    partial_3min_load[:n_readings], 45, h1_hour, h1_forecast
                )
                features['extrap_h1_error_q3'] = extrap
                features['trend_q3'] = trend
                features['vol_q3'] = vol
                features['delta_est_q2_q3'] = features.get('extrap_h1_error_q3', 0) - features.get('extrap_h1_error_q2', 0)
                features['momentum_q'] = features.get('delta_est_q2_q3', 0) - features.get('delta_est_q1_q2', 0)

        return features

    def _build_residual_features(self,
                                 residual_history: List[float],
                                 horizon: int) -> Dict[str, float]:
        """Build Stage 2 residual features."""
        features = {}

        # Residual lags (shifted by horizon)
        for lag in range(1, 7):
            idx = horizon + lag - 2  # -2 because history is 0-indexed and we want lag1 at position horizon-1
            if idx < len(residual_history):
                features[f'residual_lag{lag}'] = residual_history[idx]
            else:
                features[f'residual_lag{lag}'] = 0

        # Rolling stats
        for window in [3, 6, 12]:
            start_idx = horizon - 1
            end_idx = start_idx + window
            if end_idx <= len(residual_history):
                segment = residual_history[start_idx:end_idx]
                features[f'residual_roll_mean_{window}h'] = np.mean(segment)
                features[f'residual_roll_std_{window}h'] = np.std(segment)

        # Trends
        if 'residual_lag1' in features and 'residual_lag3' in features:
            features['residual_trend_3h'] = features['residual_lag1'] - features['residual_lag3']
        if 'residual_lag1' in features and 'residual_lag6' in features:
            features['residual_trend_6h'] = features['residual_lag1'] - features['residual_lag6']

        return features

    def predict(self,
                quarter: int,
                error_history: List[float],
                forecast_history: List[float],
                partial_3min_load: List[float],
                hour: int,
                dow: int,
                h1_forecast: Optional[float] = None,
                residual_history: Optional[List[float]] = None,
                load_volatility: Optional[float] = None,
                load_trend: Optional[float] = None,
                seasonal_error: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for all horizons at a specific quarter.

        Args:
            quarter: Hour quarter (0=xx:00, 1=xx:15, 2=xx:30, 3=xx:45)
            error_history: List of past hourly errors [t-1, t-2, ...] (at least 48)
            forecast_history: List of past hourly forecasts [t-1, t-2, ...] (at least 48)
            partial_3min_load: 3-min load readings for H+1 hour (0-15 values depending on quarter)
            hour: Current hour (0-23)
            dow: Day of week (0=Monday, 6=Sunday)
            h1_forecast: DAMAS forecast for H+1 (optional, computed from forecast_history if not provided)
            residual_history: Previous Stage 1 residuals (optional, needed for full Stage 2)
            load_volatility: Previous hour's load volatility (optional)
            load_trend: Previous hour's load trend (optional)
            seasonal_error: Seasonal error for this hour/dow (optional)

        Returns:
            Dict with predictions for each horizon:
            {
                'h1': {'s1_pred': float, 's2_pred': float, 'final_pred': float},
                'h2': {...},
                ...
            }
        """
        if quarter not in [0, 1, 2, 3]:
            raise ValueError(f"Quarter must be 0-3, got {quarter}")

        # Compute H+1 forecast if not provided
        if h1_forecast is None:
            # Assume forecast_history contains forecasts for hours [t, t-1, t-2, ...]
            # and we need the forecast for t+1
            h1_forecast = forecast_history[0] if forecast_history else 0

        # H+1 hour for bias lookup
        h1_hour = (hour + 1) % 24

        # Build base features
        features = self._build_base_features(error_history, forecast_history, hour, dow)

        # Add optional features
        if load_volatility is not None:
            features['load_volatility_lag1'] = load_volatility
        if load_trend is not None:
            features['load_trend_lag1'] = load_trend
        if seasonal_error is not None:
            features['seasonal_error'] = seasonal_error

        # Add quarter-specific features
        features = self._add_quarter_features(features, quarter, partial_3min_load, h1_hour, h1_forecast)

        predictions = {}

        for horizon in range(1, 6):
            # Get feature config
            config_key = f'q{quarter}_h{horizon}'
            s1_feature_names = self.feature_configs[config_key]['s1_features']
            s2_feature_names = self.feature_configs[config_key]['s2_features']

            # Stage 1 prediction
            s1_model = self.s1_models[(quarter, horizon)]
            X_s1 = pd.DataFrame([{fn: features.get(fn, 0) for fn in s1_feature_names}])
            s1_pred = s1_model.predict(X_s1)[0]

            # Stage 2 prediction (if residual history available)
            s2_pred = 0
            if residual_history is not None and len(residual_history) > 0:
                residual_features = self._build_residual_features(residual_history, horizon)
                s2_model = self.s2_models[(quarter, horizon)]
                X_s2 = pd.DataFrame([{fn: residual_features.get(fn, 0) for fn in s2_feature_names}])
                s2_pred = s2_model.predict(X_s2)[0]

            final_pred = s1_pred + s2_pred

            predictions[f'h{horizon}'] = {
                's1_pred': float(s1_pred),
                's2_pred': float(s2_pred),
                'final_pred': float(final_pred),
            }

        return predictions

    def predict_load(self,
                     quarter: int,
                     error_history: List[float],
                     forecast_history: List[float],
                     partial_3min_load: List[float],
                     hour: int,
                     dow: int,
                     damas_forecasts: List[float],
                     **kwargs) -> Dict[str, float]:
        """
        Predict corrected load values (not errors).

        Args:
            quarter, error_history, etc.: Same as predict()
            damas_forecasts: DAMAS forecasts for H+1 to H+5 [f_h1, f_h2, f_h3, f_h4, f_h5]

        Returns:
            Dict with corrected load predictions:
            {'h1': float, 'h2': float, 'h3': float, 'h4': float, 'h5': float}
        """
        error_predictions = self.predict(
            quarter, error_history, forecast_history, partial_3min_load,
            hour, dow, h1_forecast=damas_forecasts[0], **kwargs
        )

        load_predictions = {}
        for h in range(1, 6):
            error_pred = error_predictions[f'h{h}']['final_pred']
            load_pred = damas_forecasts[h-1] + error_pred
            load_predictions[f'h{h}'] = load_pred

        return load_predictions


def example_usage():
    """Example of how to use the predictor in production."""
    # Initialize predictor
    predictor = LoadForecastPredictor('models/')

    # Example data (would come from your data pipeline)
    error_history = [10.5, 15.2, -5.3, 8.1, 12.4, -3.2] + [0] * 42  # Last 48 errors
    forecast_history = [3500, 3480, 3510, 3490, 3520, 3505] + [3500] * 42  # Last 48 forecasts
    partial_3min_load = [3512, 3518, 3525, 3530, 3528, 3535, 3540, 3545, 3548, 3550]  # Q2 data

    # Make prediction at Q2 (xx:30)
    predictions = predictor.predict(
        quarter=2,
        error_history=error_history,
        forecast_history=forecast_history,
        partial_3min_load=partial_3min_load,
        hour=14,
        dow=2,  # Wednesday
    )

    print("Error Predictions:")
    for h, pred in predictions.items():
        print(f"  {h}: S1={pred['s1_pred']:.1f}, S2={pred['s2_pred']:.1f}, Final={pred['final_pred']:.1f} MW")

    # Get corrected load predictions
    damas_forecasts = [3520, 3540, 3560, 3580, 3590]  # H+1 to H+5
    load_predictions = predictor.predict_load(
        quarter=2,
        error_history=error_history,
        forecast_history=forecast_history,
        partial_3min_load=partial_3min_load,
        hour=14,
        dow=2,
        damas_forecasts=damas_forecasts,
    )

    print("\nCorrected Load Predictions:")
    for h, load in load_predictions.items():
        print(f"  {h}: {load:.1f} MW")


if __name__ == '__main__':
    example_usage()
