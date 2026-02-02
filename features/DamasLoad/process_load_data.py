"""
Process and combine DAMAS load data for load prediction.

Target: Actual load (Skutočnosť) - what we want to predict
Baseline: Day-ahead forecast (Denná predikcia) - what we want to beat
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_clean_excel(filepath: str, value_col_name: str) -> pd.DataFrame:
    """Load Excel file and clean the data structure."""
    df = pd.read_excel(filepath)

    # Rename columns
    df.columns = ['date_raw', 'hour', value_col_name]

    # Forward fill dates (they're only on first hour of each day)
    df['date_raw'] = df['date_raw'].ffill()

    # Parse date
    df['date'] = pd.to_datetime(df['date_raw'], format='%d.%m.%Y')

    # Create datetime index
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour'] - 1, unit='h')

    return df[['datetime', 'date', 'hour', value_col_name]]


def main():
    raw_path = Path(__file__).parent.parent.parent / 'RawData' / 'Damas'
    output_path = Path(__file__).parent

    print("=" * 60)
    print("Processing DAMAS Load Data")
    print("=" * 60)

    # Load actual data (Skutočnosť = Reality)
    print("\nLoading actual load data...")
    actual_2024 = load_and_clean_excel(
        raw_path / 'Zaťaženie ES SR - Skutočnosť (20260129 144136).xlsx',
        'actual_load_mw'
    )
    actual_2025 = load_and_clean_excel(
        raw_path / 'Zaťaženie ES SR - Skutočnosť (20260129 144044).xlsx',
        'actual_load_mw'
    )
    actual_2026 = load_and_clean_excel(
        raw_path / 'Zaťaženie ES SR - Skutočnosť (20260131 205018).xlsx',
        'actual_load_mw'
    )

    # Load forecast data (Denná predikcia = Daily prediction)
    print("Loading forecast data...")
    forecast_2024 = load_and_clean_excel(
        raw_path / 'Zaťaženie ES SR - Denná predikcia (20260129 144157).xlsx',
        'forecast_load_mw'
    )
    forecast_2025 = load_and_clean_excel(
        raw_path / 'Zaťaženie ES SR - Denná predikcia (20260129 144637).xlsx',
        'forecast_load_mw'
    )
    forecast_2026 = load_and_clean_excel(
        raw_path / 'Zaťaženie ES SR - Denná predikcia (20260131 205031).xlsx',
        'forecast_load_mw'
    )

    # Combine years
    actual = pd.concat([actual_2024, actual_2025, actual_2026], ignore_index=True)
    forecast = pd.concat([forecast_2024, forecast_2025, forecast_2026], ignore_index=True)

    print(f"\nActual data: {actual['datetime'].min()} to {actual['datetime'].max()}")
    print(f"Forecast data: {forecast['datetime'].min()} to {forecast['datetime'].max()}")

    # Merge on datetime
    combined = pd.merge(
        actual[['datetime', 'actual_load_mw']],
        forecast[['datetime', 'forecast_load_mw']],
        on='datetime',
        how='outer'
    ).sort_values('datetime').reset_index(drop=True)

    # Add time features
    combined['date'] = combined['datetime'].dt.date
    combined['year'] = combined['datetime'].dt.year
    combined['month'] = combined['datetime'].dt.month
    combined['day'] = combined['datetime'].dt.day
    combined['hour'] = combined['datetime'].dt.hour + 1  # 1-24 format
    combined['day_of_week'] = combined['datetime'].dt.dayofweek  # 0=Monday
    combined['day_of_year'] = combined['datetime'].dt.dayofyear
    combined['is_weekend'] = combined['day_of_week'].isin([5, 6]).astype(int)

    # Calculate forecast error (to beat this)
    combined['forecast_error_mw'] = combined['actual_load_mw'] - combined['forecast_load_mw']
    combined['forecast_error_pct'] = (combined['forecast_error_mw'] / combined['actual_load_mw'] * 100).round(2)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Combined Dataset Summary")
    print("=" * 60)
    print(f"Total records: {len(combined):,}")
    print(f"Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
    print(f"Missing actual: {combined['actual_load_mw'].isna().sum()}")
    print(f"Missing forecast: {combined['forecast_load_mw'].isna().sum()}")

    print("\n--- Actual Load Statistics ---")
    print(combined['actual_load_mw'].describe().round(1))

    print("\n--- Forecast Load Statistics ---")
    print(combined['forecast_load_mw'].describe().round(1))

    print("\n--- Forecast Error Statistics (what to beat) ---")
    valid_errors = combined['forecast_error_mw'].dropna()
    print(f"MAE: {np.abs(valid_errors).mean():.1f} MW")
    print(f"RMSE: {np.sqrt((valid_errors**2).mean()):.1f} MW")
    print(f"MAPE: {np.abs(combined['forecast_error_pct'].dropna()).mean():.2f}%")
    print(f"Bias: {valid_errors.mean():.1f} MW")

    # Save to parquet (efficient storage)
    output_file = output_path / 'load_data.parquet'
    combined.to_parquet(output_file, index=False)
    print(f"\nSaved combined data to: {output_file}")

    # Also save CSV for easy inspection
    csv_file = output_path / 'load_data.csv'
    combined.to_csv(csv_file, index=False)
    print(f"Saved CSV copy to: {csv_file}")

    # Show sample
    print("\n--- Sample Data ---")
    print(combined.head(24).to_string(index=False))

    return combined


if __name__ == '__main__':
    df = main()
