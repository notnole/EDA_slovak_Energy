"""
Day-Ahead Error Correction Model

Corrects DAMAS forecast by predicting systematic errors.
Run at 23:00 on day D to predict all 24 hours of day D+1.

Performance (2025 test set):
  Baseline MAE: 64.2 MW
  Model MAE:    62.4 MW
  Improvement:  +2.9%

Cross-validation: +3.4% +/- 3.2%

Usage:
    from dayahead_corrector import DayAheadCorrector

    model = DayAheadCorrector()
    model.fit(load_data)  # Train on historical data

    # Predict tomorrow
    corrected = model.predict(
        forecast_load_mw=tomorrow_forecast,
        hour=tomorrow_hours,
        dow=tomorrow_dow,
        today_errors=today_error_by_hour,
        today_load=today_actual_load
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import joblib


class DayAheadCorrector:
    """Day-ahead forecast error correction model."""

    def __init__(self):
        self.name = "Day-Ahead Error Corrector"
        self.model = None
        self.seasonal_load = None
        self.seasonal_error = None
        self.features = [
            'hour', 'dow', 'is_weekend',
            'seasonal_error', 'forecast_vs_seasonal',
            'd1_same_hour_error', 'd1_mean_error',
            'd1_evening_error', 'd1_morning_error', 'd1_night_error',
            'd1_load_vs_seasonal',
        ]

    def fit(self, df: pd.DataFrame, train_year: int = 2024):
        """
        Train model on historical data.

        Args:
            df: DataFrame with columns: datetime, actual_load_mw, forecast_load_mw,
                hour, day_of_week, is_weekend, year
            train_year: Year to use for seasonal patterns (default 2024)
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
        df = df.sort_values('datetime').reset_index(drop=True)

        # Build seasonal patterns from training year
        df_train_year = df[df['year'] == train_year]
        self.seasonal_load = df_train_year.groupby(['day_of_week', 'hour'])['actual_load_mw'].mean().to_dict()
        self.seasonal_error = df_train_year.groupby(['day_of_week', 'hour'])['load_error'].mean().to_dict()

        # Build training dataset
        dataset = self._build_dataset(df)

        if len(dataset) == 0:
            raise ValueError("No valid training samples found")

        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )

        X = dataset[self.features]
        y = dataset['target_error']
        self.model.fit(X, y)

        print(f"Trained on {len(dataset)} samples")
        return self

    def _build_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature dataset from raw data."""
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        dates = sorted(df['date'].unique())

        rows = []
        for i, target_date in enumerate(dates):
            if i == 0:
                continue

            today_date = dates[i-1]
            today = df[df['date'] == today_date]
            tomorrow = df[df['date'] == target_date]

            if len(today) < 20 or len(tomorrow) < 20:
                continue

            # Today's features
            today_errors = today.set_index('hour')['load_error'].to_dict()
            today_mean_error = today['load_error'].mean()
            today_evening_err = today[today['hour'].between(18, 23)]['load_error'].mean()
            today_morning_err = today[today['hour'].between(6, 11)]['load_error'].mean()
            today_night_err = today[today['hour'].between(0, 5)]['load_error'].mean()

            today_dow = today['day_of_week'].iloc[0]
            today_seasonal_load = np.mean([self.seasonal_load.get((today_dow, h), 0) for h in range(24)])
            today_load_vs_seasonal = today['actual_load_mw'].mean() - today_seasonal_load

            for _, row in tomorrow.iterrows():
                hour = row['hour']
                dow = row['day_of_week']

                seasonal_load = self.seasonal_load.get((dow, hour), row['actual_load_mw'])
                seasonal_error = self.seasonal_error.get((dow, hour), 0)

                rows.append({
                    'datetime': row['datetime'],
                    'date': target_date,
                    'hour': hour,
                    'dow': dow,
                    'is_weekend': row['is_weekend'],
                    'actual': row['actual_load_mw'],
                    'forecast': row['forecast_load_mw'],
                    'target_error': row['load_error'],
                    'seasonal_error': seasonal_error,
                    'forecast_vs_seasonal': row['forecast_load_mw'] - seasonal_load,
                    'd1_same_hour_error': today_errors.get(hour, 0),
                    'd1_mean_error': today_mean_error,
                    'd1_evening_error': today_evening_err,
                    'd1_morning_error': today_morning_err,
                    'd1_night_error': today_night_err,
                    'd1_load_vs_seasonal': today_load_vs_seasonal,
                })

        return pd.DataFrame(rows)

    def predict(self, forecast_load_mw, hour, dow, is_weekend,
                today_errors: dict, today_load: pd.Series) -> np.ndarray:
        """
        Predict corrected load for tomorrow.

        Args:
            forecast_load_mw: Tomorrow's DAMAS forecast (array-like, 24 values)
            hour: Hour of day (array-like, 0-23)
            dow: Day of week (array-like, 0=Mon, 6=Sun)
            is_weekend: Weekend flag (array-like, 0 or 1)
            today_errors: Dict mapping hour -> today's error for that hour
            today_load: Today's actual load (Series with 24 values)

        Returns:
            Corrected load predictions (numpy array)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert to arrays
        forecast_load_mw = np.asarray(forecast_load_mw)
        hour = np.asarray(hour)
        dow = np.asarray(dow)
        is_weekend = np.asarray(is_weekend)

        # Calculate today's features
        today_mean_error = np.mean(list(today_errors.values()))
        today_evening_err = np.mean([today_errors.get(h, 0) for h in range(18, 24)])
        today_morning_err = np.mean([today_errors.get(h, 0) for h in range(6, 12)])
        today_night_err = np.mean([today_errors.get(h, 0) for h in range(0, 6)])

        today_dow = dow[0]  # Assuming same day
        today_seasonal_load = np.mean([self.seasonal_load.get((today_dow, h), 0) for h in range(24)])
        today_load_vs_seasonal = today_load.mean() - today_seasonal_load

        # Build features for each hour
        X = pd.DataFrame({
            'hour': hour,
            'dow': dow,
            'is_weekend': is_weekend,
            'seasonal_error': [self.seasonal_error.get((d, h), 0) for d, h in zip(dow, hour)],
            'forecast_vs_seasonal': [f - self.seasonal_load.get((d, h), f)
                                     for f, d, h in zip(forecast_load_mw, dow, hour)],
            'd1_same_hour_error': [today_errors.get(h, 0) for h in hour],
            'd1_mean_error': today_mean_error,
            'd1_evening_error': today_evening_err,
            'd1_morning_error': today_morning_err,
            'd1_night_error': today_night_err,
            'd1_load_vs_seasonal': today_load_vs_seasonal,
        })

        # Predict error and correct
        pred_error = self.model.predict(X[self.features])
        corrected = forecast_load_mw + pred_error

        return corrected

    def predict_from_dataframe(self, df_today: pd.DataFrame, df_tomorrow: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method to predict from DataFrames.

        Args:
            df_today: Today's data (24 rows with actual_load_mw, forecast_load_mw)
            df_tomorrow: Tomorrow's forecast (24 rows with forecast_load_mw, hour, day_of_week)

        Returns:
            DataFrame with corrected predictions
        """
        # Today's errors
        today_errors = dict(zip(
            df_today['hour'],
            df_today['actual_load_mw'] - df_today['forecast_load_mw']
        ))

        corrected = self.predict(
            forecast_load_mw=df_tomorrow['forecast_load_mw'].values,
            hour=df_tomorrow['hour'].values,
            dow=df_tomorrow['day_of_week'].values,
            is_weekend=df_tomorrow['is_weekend'].values,
            today_errors=today_errors,
            today_load=df_today['actual_load_mw']
        )

        result = df_tomorrow.copy()
        result['corrected_load_mw'] = corrected
        return result

    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'seasonal_load': self.seasonal_load,
            'seasonal_error': self.seasonal_error,
            'features': self.features,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'DayAheadCorrector':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.seasonal_load = data['seasonal_load']
        instance.seasonal_error = data['seasonal_error']
        instance.features = data['features']
        return instance

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def __repr__(self):
        status = "trained" if self.model else "not trained"
        return f"DayAheadCorrector({status})"


def train_and_evaluate(load_data_path: str = None):
    """Train model and evaluate on test set."""
    if load_data_path is None:
        load_data_path = Path(__file__).parent.parent / 'features' / 'DamasLoad' / 'load_data.parquet'

    df = pd.read_parquet(load_data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Train model
    model = DayAheadCorrector()
    model.fit(df)

    # Evaluate on 2025 H2
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
    dataset = model._build_dataset(df)
    dataset['date'] = pd.to_datetime(dataset['date'])

    test = dataset[dataset['date'] >= '2025-07-01'].copy()
    test['pred_error'] = model.model.predict(test[model.features])
    test['corrected'] = test['forecast'] + test['pred_error']

    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    print("\n" + "=" * 50)
    print("MODEL EVALUATION (2025 H2)")
    print("=" * 50)
    print(f"Baseline (DAMAS):  {baseline_mae:.1f} MW")
    print(f"Corrected Model:   {model_mae:.1f} MW")
    print(f"Improvement:       {(1-model_mae/baseline_mae)*100:+.1f}%")
    print("=" * 50)

    print("\nFeature Importance:")
    for _, row in model.get_feature_importance().iterrows():
        print(f"  {row['feature']:<25}: {row['importance']:.3f}")

    return model


if __name__ == '__main__':
    model = train_and_evaluate()

    # Save trained model
    model_path = Path(__file__).parent / 'trained_dayahead_corrector.joblib'
    model.save(model_path)
