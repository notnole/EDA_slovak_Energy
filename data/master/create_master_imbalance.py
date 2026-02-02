"""
Script to merge all OKTE imbalance CSV files into a single master file.
Handles both English (SystemImbalance) and Slovak (OdchylkaSustavy) naming conventions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
IMBALANCE_DIR = BASE_DIR / "OKTE_Imbalnce"
OUTPUT_DIR = BASE_DIR / "data" / "master"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column mapping from Slovak to English
COLUMN_MAPPING = {
    'Dátum': 'Date',
    'Zúčtovacia perióda': 'Settlement Term',
    'Veľkosť odchýlky sústavy (MWh)': 'System Imbalance (MWh)',
    'Zúčtovacia cena PNRE (EUR/MWh)': 'SREC (EUR/MWh)',
    'Cena za PRE (EUR/MWh)': 'SREC price (EUR/MWh)',
    'Cena odchýlky sústavy (EUR/MWh)': 'Imbalance Settlement Price (EUR/MWh)',
    'Zúčtovacia cena odchýlky (EUR/MWh)': 'Settlement Price (EUR/MWh)',
    'Platba SZ za odchýlku (EUR)': 'Payment of subject for imbalance (EUR)',
    'Platba OKTE za odchýlku (EUR)': 'Payment of OKTE for imbalance (EUR)',
    'Platba za odchýlku sústavy (EUR)': 'Payment for System Imbalance (EUR)',
    'Celkové náklady na RE (EUR)': 'Total Cost of RE (EUR)',
    'Platba za PRE (EUR)': 'Payment for SREC (EUR)',
    'Saldo výnosov a nákladov za odchýlky a RE (EUR)': 'Balance of Revenues and Expenses for Imbalance and RE (EUR)',
    'Zúčtovanie v zvláštnom režime': 'Settlement in special mode'
}

SPECIAL_MODE_MAPPING = {
    'Nie': 'No',
    'Áno': 'Yes'
}


def parse_number(value):
    """Parse number handling European format (comma as decimal separator)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    # Handle European format: 1.234,56 -> 1234.56
    if ',' in value and '.' in value:
        value = value.replace('.', '').replace(',', '.')
    elif ',' in value:
        value = value.replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_date_english(date_str):
    """Parse date in M/D/YYYY format."""
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except:
        try:
            return pd.to_datetime(date_str, dayfirst=False)
        except:
            return pd.NaT


def parse_date_slovak(date_str):
    """Parse date in D.M.YYYY format."""
    try:
        return pd.to_datetime(date_str, format='%d.%m.%Y')
    except:
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except:
            return pd.NaT


def load_english_file(filepath):
    """Load SystemImbalance file (English format)."""
    print(f"  Loading English file: {filepath.name}")
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')

    # Parse date
    df['Date'] = df['Date'].apply(parse_date_english)

    # Parse numeric columns
    numeric_cols = [col for col in df.columns if col not in ['Date', 'Settlement Term', 'Settlement in special mode']]
    for col in numeric_cols:
        df[col] = df[col].apply(parse_number)

    df['source_file'] = filepath.name
    return df


def load_slovak_file(filepath):
    """Load OdchylkaSustavy file (Slovak format)."""
    print(f"  Loading Slovak file: {filepath.name}")
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')

    # Rename columns to English
    df = df.rename(columns=COLUMN_MAPPING)

    # Parse date
    df['Date'] = df['Date'].apply(parse_date_slovak)

    # Parse numeric columns
    numeric_cols = [col for col in df.columns if col not in ['Date', 'Settlement Term', 'Settlement in special mode']]
    for col in numeric_cols:
        df[col] = df[col].apply(parse_number)

    # Convert special mode values
    if 'Settlement in special mode' in df.columns:
        df['Settlement in special mode'] = df['Settlement in special mode'].map(SPECIAL_MODE_MAPPING).fillna(df['Settlement in special mode'])

    df['source_file'] = filepath.name
    return df


def create_datetime_column(df):
    """Create a proper datetime column from Date and Settlement Term."""
    # Settlement Term 1 = 00:00-00:15, Term 2 = 00:15-00:30, etc.
    # Each term is 15 minutes, starting from midnight
    df['datetime'] = df['Date'] + pd.to_timedelta((df['Settlement Term'] - 1) * 15, unit='m')
    return df


def main():
    print("=" * 60)
    print("Creating Master Imbalance Data File")
    print("=" * 60)

    all_dfs = []

    # Get all CSV files
    csv_files = list(IMBALANCE_DIR.glob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files in {IMBALANCE_DIR}")

    # Process each file
    for filepath in sorted(csv_files):
        try:
            if filepath.name.startswith("SystemImbalance"):
                df = load_english_file(filepath)
            elif filepath.name.startswith("OdchylkaSustavy"):
                df = load_slovak_file(filepath)
            else:
                print(f"  Skipping unknown file format: {filepath.name}")
                continue

            all_dfs.append(df)
            print(f"    -> {len(df)} rows loaded")
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")

    # Combine all dataframes
    print("\nMerging all data...")
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Create datetime column
    master_df = create_datetime_column(master_df)

    # Sort by datetime
    master_df = master_df.sort_values('datetime').reset_index(drop=True)

    # Remove duplicates (keep first occurrence)
    before_dedup = len(master_df)
    master_df = master_df.drop_duplicates(subset=['datetime'], keep='first')
    after_dedup = len(master_df)
    print(f"Removed {before_dedup - after_dedup} duplicate rows")

    # Remove outliers (|imbalance| > 300 MWh are likely data errors)
    OUTLIER_THRESHOLD = 300
    before_outlier = len(master_df)
    outliers = master_df[master_df['System Imbalance (MWh)'].abs() > OUTLIER_THRESHOLD]
    if len(outliers) > 0:
        print(f"\nOutliers detected (|imbalance| > {OUTLIER_THRESHOLD} MWh):")
        for _, row in outliers.iterrows():
            print(f"  {row['datetime']}: {row['System Imbalance (MWh)']:.2f} MWh")
    master_df = master_df[master_df['System Imbalance (MWh)'].abs() <= OUTLIER_THRESHOLD]
    after_outlier = len(master_df)
    print(f"Removed {before_outlier - after_outlier} outlier rows")

    # Final columns order
    cols_order = ['datetime', 'Date', 'Settlement Term', 'System Imbalance (MWh)',
                  'SREC (EUR/MWh)', 'SREC price (EUR/MWh)', 'Imbalance Settlement Price (EUR/MWh)',
                  'Payment of subject for imbalance (EUR)', 'Payment of OKTE for imbalance (EUR)',
                  'Payment for System Imbalance (EUR)', 'Total Cost of RE (EUR)',
                  'Payment for SREC (EUR)', 'Balance of Revenues and Expenses for Imbalance and RE (EUR)',
                  'Settlement in special mode', 'source_file']

    # Only keep columns that exist
    cols_order = [c for c in cols_order if c in master_df.columns]
    master_df = master_df[cols_order]

    # Save to CSV
    output_path = OUTPUT_DIR / "master_imbalance_data.csv"
    master_df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(master_df):,}")
    print(f"Date range: {master_df['datetime'].min()} to {master_df['datetime'].max()}")
    print(f"Days covered: {(master_df['datetime'].max() - master_df['datetime'].min()).days}")
    print(f"\nSystem Imbalance (MWh) Statistics:")
    print(master_df['System Imbalance (MWh)'].describe())
    print(f"\nOutput saved to: {output_path}")

    return master_df


if __name__ == "__main__":
    df = main()
