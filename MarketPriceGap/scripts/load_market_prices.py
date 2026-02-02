"""
Market Price Data Loading Module
Loads and merges DA, IDM, and Imbalance market prices for price gap analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA = BASE_DIR / "RawData"
OKTE_DATA = BASE_DIR / "OKTE_Imbalnce"
OUTPUT_DIR = BASE_DIR / "MarketPriceGap" / "data" / "processed"


def load_da_market() -> pd.DataFrame:
    """Load Day-Ahead market prices (hourly)."""
    print("[*] Loading DA market data...")

    da_files = sorted(RAW_DATA.glob("DA_market/Celkove_vysledky_DT_*.csv"))
    dfs = []

    for f in da_files:
        print(f"    Loading {f.name}")
        df = pd.read_csv(f, sep=';', encoding='utf-8')
        dfs.append(df)

    da = pd.concat(dfs, ignore_index=True)

    # Rename columns
    da = da.rename(columns={
        'Obchodny den': 'date',
        'Obchodný deň': 'date',
        'Cislo periody': 'period',
        'Číslo periódy': 'period',
        'Cena SK (EUR/MWh)': 'da_price',
        'Dopyt úspešný (MW)': 'da_demand',
        'Ponuka úspešná (MW)': 'da_supply',
        'Tok CZ ➝ SK (MW)': 'flow_cz_sk',
        'Tok SK ➝ CZ (MW)': 'flow_sk_cz',
        'Tok PL ➝ SK (MW)': 'flow_pl_sk',
        'Tok SK ➝ PL (MW)': 'flow_sk_pl',
        'Tok HU ➝ SK (MW)': 'flow_hu_sk',
        'Tok SK ➝ HU (MW)': 'flow_sk_hu'
    })

    # Parse date
    da['date'] = pd.to_datetime(da['date'], format='%d.%m.%Y', dayfirst=True)

    # Helper for European number format
    def parse_european_number(x):
        if pd.isna(x) or x == '':
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        x = str(x).strip()
        if x == '':
            return np.nan
        dots = x.count('.')
        commas = x.count(',')
        if dots > 1:
            x = x.replace('.', '')
        elif dots == 1 and commas == 1:
            x = x.replace('.', '').replace(',', '.')
        elif commas == 1:
            x = x.replace(',', '.')
        return float(x)

    # Convert all numeric columns
    numeric_cols = ['da_price', 'da_demand', 'da_supply', 'flow_cz_sk', 'flow_sk_cz',
                    'flow_pl_sk', 'flow_sk_pl', 'flow_hu_sk', 'flow_sk_hu']
    for col in numeric_cols:
        if col in da.columns:
            da[col] = da[col].apply(parse_european_number)

    # Create hourly timestamp
    da['hour'] = da['period'] - 1
    da['timestamp_hour'] = da['date'] + pd.to_timedelta(da['hour'], unit='h')

    # Calculate net cross-border flow
    da['net_import'] = 0.0
    for col in ['flow_cz_sk', 'flow_pl_sk', 'flow_hu_sk']:
        if col in da.columns:
            da['net_import'] += da[col].fillna(0)
    for col in ['flow_sk_cz', 'flow_sk_pl', 'flow_sk_hu']:
        if col in da.columns:
            da['net_import'] -= da[col].fillna(0)

    print(f"[+] DA: {len(da)} hourly records, {da['date'].min()} to {da['date'].max()}")
    return da


def load_idm_market(resolution: str = '60') -> pd.DataFrame:
    """Load Intraday Market prices.

    Args:
        resolution: '15' for 15-min or '60' for hourly
    """
    print(f"[*] Loading IDM market data ({resolution} min resolution)...")

    idm_folders = sorted(RAW_DATA.glob("IDM_MarketData/IDM_total_results_*"))
    dfs = []

    for folder in idm_folders:
        csv_file = folder / f"{resolution} min.csv"
        if csv_file.exists():
            print(f"    Loading {folder.name}/{resolution} min.csv")
            df = pd.read_csv(csv_file, sep=';', encoding='utf-8')
            dfs.append(df)

    if not dfs:
        print("[-] No IDM files found")
        return pd.DataFrame()

    idm = pd.concat(dfs, ignore_index=True)

    # Rename columns
    idm = idm.rename(columns={
        'Delivery day': 'date',
        'Period number': 'period',
        'Weighted average price of all trades (EUR/MWh)': 'idm_vwap',
        'Total Traded Quantity (MW)': 'idm_volume_mw',
        'Total Traded Quantity (MWh)': 'idm_volume_mwh',
        'Buy trades (MW)': 'idm_buy_mw',
        'Sell trades (MW)': 'idm_sell_mw',
        'Buy trades (MWh)': 'idm_buy_mwh',
        'Sell trades (MWh)': 'idm_sell_mwh',
        'Traded Quantity Difference (MW)': 'idm_net_mw',
        'Traded Quantity Difference (MWh)': 'idm_net_mwh'
    })

    # Parse date
    idm['date'] = pd.to_datetime(idm['date'], format='%d.%m.%Y', dayfirst=True)

    # Convert numeric columns (handle European format: 1.234,56 -> 1234.56)
    def parse_european_number(x):
        if pd.isna(x) or x == '':
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        x = str(x).strip()
        if x == '':
            return np.nan
        # Count dots and commas to detect format
        dots = x.count('.')
        commas = x.count(',')
        if dots > 1:  # Multiple dots = thousand separator (e.g., 1.234.567)
            x = x.replace('.', '')
        elif dots == 1 and commas == 1:  # Both = European (e.g., 1.234,56)
            x = x.replace('.', '').replace(',', '.')
        elif commas == 1:  # Single comma = European decimal
            x = x.replace(',', '.')
        return float(x)

    for col in ['idm_vwap', 'idm_volume_mw', 'idm_volume_mwh', 'idm_buy_mw', 'idm_sell_mw',
                'idm_buy_mwh', 'idm_sell_mwh', 'idm_net_mw', 'idm_net_mwh']:
        if col in idm.columns:
            idm[col] = idm[col].apply(parse_european_number)

    # Create timestamp
    if resolution == '60':
        idm['hour'] = idm['period'] - 1
        idm['timestamp_hour'] = idm['date'] + pd.to_timedelta(idm['hour'], unit='h')
    else:  # 15-min
        idm['minutes'] = (idm['period'] - 1) * 15
        idm['timestamp_15min'] = idm['date'] + pd.to_timedelta(idm['minutes'], unit='m')
        idm['hour'] = idm['minutes'] // 60
        idm['timestamp_hour'] = idm['date'] + pd.to_timedelta(idm['hour'], unit='h')

    print(f"[+] IDM ({resolution}min): {len(idm)} records, {idm['date'].min()} to {idm['date'].max()}")
    return idm


def load_imbalance_prices() -> pd.DataFrame:
    """Load OKTE system imbalance prices (15-min)."""
    print("[*] Loading Imbalance price data...")

    # Load both naming conventions
    imb_files = list(OKTE_DATA.glob("OdchylkaSustavy_*.csv")) + list(OKTE_DATA.glob("SystemImbalance_*.csv"))
    dfs = []

    for f in sorted(imb_files):
        print(f"    Loading {f.name}")
        df = pd.read_csv(f, sep=';', encoding='utf-8')
        dfs.append(df)

    imb = pd.concat(dfs, ignore_index=True)

    # Rename columns (Slovak names)
    col_map = {
        'Datum': 'date',
        'Dátum': 'date',
        'Zuctovacia perioda': 'period',
        'Zúčtovacia perióda': 'period',
        'Velkost odchylky sustavy (MWh)': 'imbalance_mwh',
        'Veľkosť odchýlky sústavy (MWh)': 'imbalance_mwh',
        'Cena odchylky sustavy (EUR/MWh)': 'imb_price',
        'Cena odchýlky sústavy (EUR/MWh)': 'imb_price',
        'Zuctovacia cena odchylky (EUR/MWh)': 'imb_settlement_price',
        'Zúčtovacia cena odchýlky (EUR/MWh)': 'imb_settlement_price',
        'Zuctovacia cena PNRE (EUR/MWh)': 'imb_pnre_price',
        'Zúčtovacia cena PNRE (EUR/MWh)': 'imb_pnre_price'
    }
    imb = imb.rename(columns=col_map)

    # Parse date
    imb['date'] = pd.to_datetime(imb['date'], format='%d.%m.%Y', dayfirst=True, errors='coerce')
    if imb['date'].isna().any():
        imb['date'] = pd.to_datetime(imb['date'], format='%m/%d/%Y', errors='coerce')

    # Convert numeric columns (handle European format)
    def parse_european_number(x):
        if pd.isna(x) or x == '':
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        x = str(x).strip()
        if x == '':
            return np.nan
        dots = x.count('.')
        commas = x.count(',')
        if dots > 1:
            x = x.replace('.', '')
        elif dots == 1 and commas == 1:
            x = x.replace('.', '').replace(',', '.')
        elif commas == 1:
            x = x.replace(',', '.')
        return float(x)

    for col in ['imbalance_mwh', 'imb_price', 'imb_settlement_price', 'imb_pnre_price']:
        if col in imb.columns:
            imb[col] = imb[col].apply(parse_european_number)

    # Create 15-min timestamp
    imb['minutes'] = (imb['period'] - 1) * 15
    imb['timestamp_15min'] = imb['date'] + pd.to_timedelta(imb['minutes'], unit='m')

    # Also create hourly reference
    imb['hour'] = imb['minutes'] // 60
    imb['timestamp_hour'] = imb['date'] + pd.to_timedelta(imb['hour'], unit='h')

    print(f"[+] Imbalance: {len(imb)} 15-min records, {imb['date'].min()} to {imb['date'].max()}")
    return imb


def merge_hourly_prices(da: pd.DataFrame, idm: pd.DataFrame, imb: pd.DataFrame) -> pd.DataFrame:
    """Merge all market prices at hourly resolution."""
    print("[*] Merging prices at hourly resolution...")

    # Start with DA
    merged = da[['timestamp_hour', 'da_price', 'da_demand', 'da_supply', 'net_import']].copy()
    merged = merged.drop_duplicates(subset='timestamp_hour')

    # Add IDM hourly
    if not idm.empty:
        agg_dict = {'idm_vwap': 'mean'}
        if 'idm_volume_mwh' in idm.columns:
            agg_dict['idm_volume_mwh'] = 'sum'
        if 'idm_volume_mw' in idm.columns:
            agg_dict['idm_volume_mw'] = 'sum'
        idm_hourly = idm.groupby('timestamp_hour').agg(agg_dict).reset_index()
        merged = merged.merge(idm_hourly, on='timestamp_hour', how='outer')

    # Add imbalance (aggregate to hourly)
    if not imb.empty:
        imb_hourly = imb.groupby('timestamp_hour').agg({
            'imb_price': 'mean',
            'imb_settlement_price': 'mean',
            'imbalance_mwh': 'sum'
        }).reset_index()
        merged = merged.merge(imb_hourly, on='timestamp_hour', how='outer')

    # Sort and fill
    merged = merged.sort_values('timestamp_hour').reset_index(drop=True)

    # Calculate price spreads
    merged['spread_da_idm'] = merged['da_price'] - merged['idm_vwap']
    merged['spread_idm_imb'] = merged['idm_vwap'] - merged['imb_price']
    merged['spread_da_imb'] = merged['da_price'] - merged['imb_price']

    # Time features
    merged['date'] = merged['timestamp_hour'].dt.date
    merged['hour'] = merged['timestamp_hour'].dt.hour
    merged['dayofweek'] = merged['timestamp_hour'].dt.dayofweek
    merged['month'] = merged['timestamp_hour'].dt.month
    merged['is_weekend'] = merged['dayofweek'] >= 5

    print(f"[+] Merged: {len(merged)} hourly records")
    return merged


def load_all_prices(save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all market price data and create merged dataset.

    Returns:
        Tuple of (da, idm_hourly, imbalance, merged_hourly)
    """
    da = load_da_market()
    idm = load_idm_market(resolution='60')
    imb = load_imbalance_prices()

    merged = merge_hourly_prices(da, idm, imb)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_csv(OUTPUT_DIR / "hourly_market_prices.csv", index=False)
        print(f"[+] Saved merged data to {OUTPUT_DIR / 'hourly_market_prices.csv'}")

    return da, idm, imb, merged


if __name__ == "__main__":
    da, idm, imb, merged = load_all_prices()

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nDA Market:     {len(da):,} hourly records")
    print(f"IDM Market:    {len(idm):,} hourly records")
    print(f"Imbalance:     {len(imb):,} 15-min records")
    print(f"Merged Hourly: {len(merged):,} records")

    print("\n--- Price Statistics (EUR/MWh) ---")
    for col, name in [('da_price', 'DA Price'), ('idm_vwap', 'IDM VWAP'), ('imb_price', 'Imbalance')]:
        if col in merged.columns:
            print(f"{name:15s}: mean={merged[col].mean():7.2f}, std={merged[col].std():7.2f}, "
                  f"min={merged[col].min():7.2f}, max={merged[col].max():7.2f}")

    print("\n--- Price Spreads (EUR/MWh) ---")
    for col, name in [('spread_da_idm', 'DA - IDM'), ('spread_idm_imb', 'IDM - Imb'), ('spread_da_imb', 'DA - Imb')]:
        if col in merged.columns:
            s = merged[col].dropna()
            print(f"{name:15s}: mean={s.mean():7.2f}, std={s.std():7.2f}, "
                  f"min={s.min():7.2f}, max={s.max():7.2f}")
