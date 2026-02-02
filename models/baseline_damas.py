"""
Baseline Model: DAMAS Day-Ahead Forecast

This is the baseline predictor - simply uses the DAMAS forecast as-is.

Performance (2025 test set):
  MAE: 64.2 MW
  MAPE: 2.34%

Usage:
    from baseline_damas import DamasBaseline

    model = DamasBaseline()
    predictions = model.predict(forecast_load_mw)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DamasBaseline:
    """Baseline model that returns DAMAS forecast unchanged."""

    def __init__(self):
        self.name = "DAMAS Baseline"
        self.mae = 64.2  # MW on 2025 test set
        self.mape = 2.34  # %

    def predict(self, forecast_load_mw):
        """
        Return DAMAS forecast as prediction.

        Args:
            forecast_load_mw: DAMAS day-ahead forecast (scalar, array, or Series)

        Returns:
            Same as input - baseline just returns the forecast
        """
        return forecast_load_mw

    def __repr__(self):
        return f"DamasBaseline(MAE={self.mae:.1f} MW, MAPE={self.mape:.2f}%)"


def evaluate_baseline(load_data_path: str = None):
    """
    Evaluate baseline performance on load data.

    Args:
        load_data_path: Path to load_data.parquet (optional)
    """
    if load_data_path is None:
        load_data_path = Path(__file__).parent.parent / 'features' / 'DamasLoad' / 'load_data.parquet'

    df = pd.read_parquet(load_data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter to 2025
    df_2025 = df[df['year'] == 2025].copy()

    # Calculate errors
    df_2025['error'] = df_2025['actual_load_mw'] - df_2025['forecast_load_mw']

    mae = df_2025['error'].abs().mean()
    mape = (df_2025['error'].abs() / df_2025['actual_load_mw'] * 100).mean()
    bias = df_2025['error'].mean()
    rmse = np.sqrt((df_2025['error'] ** 2).mean())

    print("=" * 50)
    print("DAMAS BASELINE PERFORMANCE (2025)")
    print("=" * 50)
    print(f"  MAE:  {mae:.1f} MW")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.1f} MW")
    print(f"  Bias: {bias:.1f} MW")
    print("=" * 50)

    # By hour
    hourly = df_2025.groupby('hour').apply(lambda x: x['error'].abs().mean())
    print(f"\nWorst hour:  {hourly.idxmax()} ({hourly.max():.1f} MW)")
    print(f"Best hour:   {hourly.idxmin()} ({hourly.min():.1f} MW)")

    return {'mae': mae, 'mape': mape, 'rmse': rmse, 'bias': bias}


if __name__ == '__main__':
    evaluate_baseline()
