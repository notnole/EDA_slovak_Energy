"""
Process Day-Ahead Electricity Prices from DAMAS.

Data source: Slovak electricity market (OKTE)
- 2024-2025: Hourly data
- 2026: 15-minute data (aggregated to hourly)

Output: Cleaned hourly price data with features for load forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
RAW_PATH = BASE_PATH / 'RawData' / 'Damas'
OUTPUT_PATH = Path(__file__).parent

# Raw files
FILES = {
    2024: RAW_PATH / 'Celkove_vysledky_DT_01-01-2024_31-12-2024.csv',
    2025: RAW_PATH / 'Celkove_vysledky_DT_01-01-2025_31-12-2025.csv',
    2026: RAW_PATH / 'Celkove_vysledky_DT_01-01-2026_28-01-2026.csv',
}


def parse_price(value):
    """Parse price value, handling European decimal format."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # Handle European format: "0,10" -> 0.10
    return float(str(value).replace(',', '.'))


def parse_flow(value):
    """Parse flow value."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(',', '.'))


def load_file(filepath: Path, year: int) -> pd.DataFrame:
    """Load and parse a single CSV file."""
    print(f"\nLoading {year} data from {filepath.name}...")

    # Read with semicolon separator
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')

    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Rename columns to English
    column_map = {
        'Obchodný deň': 'date',
        'Číslo periódy': 'period_num',
        'Perióda': 'period',
        'Perióda (min)': 'period_min',
        'Cena SK (EUR/MWh)': 'price_eur_mwh',
        'Dopyt úspešný (MW)': 'demand_mw',
        'Ponuka úspešná (MW)': 'supply_mw',
        'Tok CZ ➝ SK (MW)': 'flow_cz_to_sk',
        'Tok SK ➝ CZ (MW)': 'flow_sk_to_cz',
        'Tok PL ➝ SK (MW)': 'flow_pl_to_sk',
        'Tok SK ➝ PL (MW)': 'flow_sk_to_pl',
        'Tok HU ➝ SK (MW)': 'flow_hu_to_sk',
        'Tok SK ➝ HU (MW)': 'flow_sk_to_hu',
        'Stav zverejnenia': 'status',
    }
    df = df.rename(columns=column_map)

    # Parse date
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

    # Parse numeric columns
    df['price_eur_mwh'] = df['price_eur_mwh'].apply(parse_price)
    df['demand_mw'] = df['demand_mw'].apply(parse_flow)
    df['supply_mw'] = df['supply_mw'].apply(parse_flow)

    # Parse flow columns
    for col in ['flow_cz_to_sk', 'flow_sk_to_cz', 'flow_pl_to_sk', 'flow_sk_to_pl',
                'flow_hu_to_sk', 'flow_sk_to_hu']:
        if col in df.columns:
            df[col] = df[col].apply(parse_flow)

    # Calculate net flows
    df['net_flow_cz'] = df['flow_cz_to_sk'] - df['flow_sk_to_cz']
    df['net_flow_pl'] = df['flow_pl_to_sk'] - df['flow_sk_to_pl']
    df['net_flow_hu'] = df['flow_hu_to_sk'] - df['flow_sk_to_hu']
    df['net_import'] = df['net_flow_cz'] + df['net_flow_pl'] + df['net_flow_hu']

    df['year'] = year

    return df


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sub-hourly data to hourly."""
    period_min = df['period_min'].iloc[0]

    if period_min == 60:
        # Already hourly
        df['hour'] = df['period_num'] - 1  # 0-23
        return df

    print(f"  Aggregating {period_min}-minute data to hourly...")

    # For 15-minute data: periods 1-4 = hour 0, 5-8 = hour 1, etc.
    df['hour'] = (df['period_num'] - 1) // (60 // period_min)

    # Aggregate by date and hour
    agg_funcs = {
        'price_eur_mwh': 'mean',  # Average price
        'demand_mw': 'mean',
        'supply_mw': 'mean',
        'flow_cz_to_sk': 'mean',
        'flow_sk_to_cz': 'mean',
        'flow_pl_to_sk': 'mean',
        'flow_sk_to_pl': 'mean',
        'flow_hu_to_sk': 'mean',
        'flow_sk_to_hu': 'mean',
        'net_flow_cz': 'mean',
        'net_flow_pl': 'mean',
        'net_flow_hu': 'mean',
        'net_import': 'mean',
        'year': 'first',
    }

    # Also calculate price volatility within hour
    df_hourly = df.groupby(['date', 'hour']).agg(agg_funcs).reset_index()

    # Add intra-hour price volatility
    price_std = df.groupby(['date', 'hour'])['price_eur_mwh'].std().reset_index()
    price_std.columns = ['date', 'hour', 'price_std_intra']
    df_hourly = df_hourly.merge(price_std, on=['date', 'hour'], how='left')

    print(f"  Aggregated to {len(df_hourly)} hourly records")

    return df_hourly


def create_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create datetime column from date and hour."""
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['datetime'].dt.quarter
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features for forecasting."""
    df = df.sort_values('datetime').reset_index(drop=True)

    # Yesterday's prices (lag 24h)
    df['price_lag24'] = df['price_eur_mwh'].shift(24)
    df['price_change_24h'] = df['price_eur_mwh'] - df['price_lag24']
    df['price_change_24h_pct'] = (df['price_change_24h'] / df['price_lag24'].abs().clip(lower=1)) * 100

    # Week ago price (lag 168h)
    df['price_lag168'] = df['price_eur_mwh'].shift(168)
    df['price_change_7d'] = df['price_eur_mwh'] - df['price_lag168']

    # Same hour yesterday
    df['price_same_hour_yesterday'] = df['price_lag24']

    # Daily statistics (rolling)
    df['price_daily_mean'] = df['price_eur_mwh'].rolling(24, min_periods=1).mean()
    df['price_daily_std'] = df['price_eur_mwh'].rolling(24, min_periods=1).std()
    df['price_daily_min'] = df['price_eur_mwh'].rolling(24, min_periods=1).min()
    df['price_daily_max'] = df['price_eur_mwh'].rolling(24, min_periods=1).max()
    df['price_daily_range'] = df['price_daily_max'] - df['price_daily_min']

    # Price relative to daily mean
    df['price_vs_daily_mean'] = df['price_eur_mwh'] - df['price_daily_mean'].shift(1)

    # Yesterday's daily stats
    df['yesterday_mean_price'] = df['price_daily_mean'].shift(24)
    df['yesterday_std_price'] = df['price_daily_std'].shift(24)
    df['yesterday_range_price'] = df['price_daily_range'].shift(24)

    # Price momentum (change in last 6 hours)
    df['price_momentum_6h'] = df['price_eur_mwh'] - df['price_eur_mwh'].shift(6)

    # Net import features
    df['net_import_lag24'] = df['net_import'].shift(24)
    df['net_import_change_24h'] = df['net_import'] - df['net_import_lag24']

    return df


def main():
    """Process all files and create combined dataset."""
    print("=" * 70)
    print("Processing Day-Ahead Prices")
    print("=" * 70)

    # Load and process each year
    dfs = []
    for year, filepath in FILES.items():
        if filepath.exists():
            df = load_file(filepath, year)
            df = aggregate_to_hourly(df)
            dfs.append(df)
        else:
            print(f"  WARNING: File not found for {year}")

    # Combine all years
    print("\nCombining all years...")
    df = pd.concat(dfs, ignore_index=True)

    # Create datetime
    df = create_datetime(df)

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    # Add features
    print("Adding time features...")
    df = add_time_features(df)

    print("Adding price features...")
    df = add_price_features(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nPrice statistics:")
    print(f"  Mean:   {df['price_eur_mwh'].mean():.2f} EUR/MWh")
    print(f"  Std:    {df['price_eur_mwh'].std():.2f} EUR/MWh")
    print(f"  Min:    {df['price_eur_mwh'].min():.2f} EUR/MWh")
    print(f"  Max:    {df['price_eur_mwh'].max():.2f} EUR/MWh")

    # Check for negative prices
    neg_count = (df['price_eur_mwh'] < 0).sum()
    print(f"\nNegative prices: {neg_count} ({neg_count/len(df)*100:.2f}%)")

    # Save
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Select columns for output
    output_cols = [
        'datetime', 'date', 'hour', 'year', 'month', 'day_of_week', 'is_weekend', 'quarter',
        'price_eur_mwh', 'demand_mw', 'supply_mw',
        'net_flow_cz', 'net_flow_pl', 'net_flow_hu', 'net_import',
        'price_lag24', 'price_change_24h', 'price_change_24h_pct',
        'price_lag168', 'price_change_7d',
        'price_daily_mean', 'price_daily_std', 'price_daily_range',
        'yesterday_mean_price', 'yesterday_std_price', 'yesterday_range_price',
        'price_momentum_6h',
        'net_import_lag24', 'net_import_change_24h',
    ]

    df_out = df[output_cols]

    # Save parquet and CSV
    df_out.to_parquet(OUTPUT_PATH / 'da_prices.parquet', index=False)
    df_out.to_csv(OUTPUT_PATH / 'da_prices.csv', index=False)

    print(f"\nSaved to:")
    print(f"  {OUTPUT_PATH / 'da_prices.parquet'}")
    print(f"  {OUTPUT_PATH / 'da_prices.csv'}")

    return df_out


if __name__ == '__main__':
    df = main()
