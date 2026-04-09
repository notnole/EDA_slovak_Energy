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


def load_da_market() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Day-Ahead market prices.

    Returns:
        Tuple of (da_hourly, da_qh) where da_qh has native 15-min for 2026
        and hourly rows broadcast to 4 QH periods for 2024-2025.
    """
    print("[*] Loading DA market data...")

    da_files = sorted(RAW_DATA.glob("DA_market/Celkove_vysledky_DT_*.csv"))
    da_files += sorted(RAW_DATA.glob("DA_market/Total_results_DAM_*.csv"))

    # Column rename map (Slovak + English headers)
    da_col_map = {
        # Slovak headers
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
        'Tok SK ➝ HU (MW)': 'flow_sk_hu',
        # English headers (2026 OKTE format)
        'Delivery day': 'date',
        'Period number': 'period',
        'Price SK (EUR/MWh)': 'da_price',
        'Demand successful (MW)': 'da_demand',
        'Supply successful (MW)': 'da_supply',
        'Flow CZ ➝ SK (MW)': 'flow_cz_sk',
        'Flow SK ➝ CZ (MW)': 'flow_sk_cz',
        'Flow PL ➝ SK (MW)': 'flow_pl_sk',
        'Flow SK ➝ PL (MW)': 'flow_sk_pl',
        'Flow HU ➝ SK (MW)': 'flow_hu_sk',
        'Flow SK ➝ HU (MW)': 'flow_sk_hu',
    }

    # Rename per-file before concat to avoid duplicate column names
    dfs = []
    for f in da_files:
        print(f"    Loading {f.name}")
        df = pd.read_csv(f, sep=';', encoding='utf-8')
        df = df.rename(columns=da_col_map)
        dfs.append(df)

    da = pd.concat(dfs, ignore_index=True)

    # Parse date (D.M.YYYY format - same for both Slovak and English DA files)
    da['date'] = pd.to_datetime(da['date'], format='%d.%m.%Y', dayfirst=True)

    # Deduplicate overlapping files (keep last = prefer newer English file)
    n_before = len(da)
    da = da.drop_duplicates(subset=['date', 'period'], keep='last')
    n_dropped = n_before - len(da)
    if n_dropped > 0:
        print(f"[*] Deduplicated {n_dropped} overlapping DA rows")

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
            # Detect format by which separator comes last
            if x.rfind('.') > x.rfind(','):
                # English: 1,072.3 -> dot is decimal
                x = x.replace(',', '')
            else:
                # European: 1.072,3 -> comma is decimal
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

    # Detect resolution per row: 2024-2025 = 24 periods/day (60-min),
    # 2026+ = 96 periods/day (15-min).
    resolution_col = None
    for c in da.columns:
        if 'min' in str(c).lower() and 'peri' in str(c).lower():
            resolution_col = c
            break

    if resolution_col is not None:
        is_15min_mask = da[resolution_col] == 15
    else:
        max_per_day = da.groupby('date')['period'].transform('max')
        is_15min_mask = max_per_day > 24

    # --- Build QH DataFrame (native 15-min where available, broadcast hourly elsewhere) ---
    da_qh_parts = []

    # 15-min rows: already at QH resolution
    if is_15min_mask.any():
        qh_native = da[is_15min_mask].copy()
        qh_native['minute'] = ((qh_native['period'] - 1) % 4) * 15
        qh_native['hour'] = (qh_native['period'] - 1) // 4
        qh_native['timestamp_qh'] = qh_native['date'] + pd.to_timedelta(qh_native['hour'], unit='h') + pd.to_timedelta(qh_native['minute'], unit='m')
        qh_native['timestamp_hour'] = qh_native['date'] + pd.to_timedelta(qh_native['hour'], unit='h')
        da_qh_parts.append(qh_native)
        print(f"[*] DA 15-min native: {len(qh_native)} QH rows")

    # Hourly rows: broadcast each hour to 4 QH periods
    if (~is_15min_mask).any():
        hourly_rows = da[~is_15min_mask].copy()
        hourly_rows['hour'] = hourly_rows['period'] - 1
        hourly_rows['timestamp_hour'] = hourly_rows['date'] + pd.to_timedelta(hourly_rows['hour'], unit='h')
        # Expand to 4 QH periods per hour
        expanded = []
        for qh_offset in [0, 15, 30, 45]:
            part = hourly_rows.copy()
            part['minute'] = qh_offset
            part['timestamp_qh'] = part['timestamp_hour'] + pd.to_timedelta(qh_offset, unit='m')
            expanded.append(part)
        hourly_broadcast = pd.concat(expanded, ignore_index=True)
        da_qh_parts.append(hourly_broadcast)
        print(f"[*] DA hourly broadcast: {len(hourly_rows)} hourly -> {len(hourly_broadcast)} QH rows")

    da_qh = pd.concat(da_qh_parts, ignore_index=True)

    # --- Build hourly DataFrame (aggregate 15-min where needed) ---
    da_hourly_parts = []

    if (~is_15min_mask).any():
        hourly_native = da[~is_15min_mask].copy()
        hourly_native['hour'] = hourly_native['period'] - 1
        hourly_native['timestamp_hour'] = hourly_native['date'] + pd.to_timedelta(hourly_native['hour'], unit='h')
        da_hourly_parts.append(hourly_native)

    if is_15min_mask.any():
        qh_to_agg = da[is_15min_mask].copy()
        qh_to_agg['hour'] = (qh_to_agg['period'] - 1) // 4
        qh_to_agg['timestamp_hour'] = qh_to_agg['date'] + pd.to_timedelta(qh_to_agg['hour'], unit='h')
        agg_dict = {'da_price': 'mean'}
        for col in ['da_demand', 'da_supply', 'flow_cz_sk', 'flow_sk_cz',
                     'flow_pl_sk', 'flow_sk_pl', 'flow_hu_sk', 'flow_sk_hu']:
            if col in qh_to_agg.columns:
                agg_dict[col] = 'mean'
        qh_agged = qh_to_agg.groupby(['date', 'hour', 'timestamp_hour']).agg(agg_dict).reset_index()
        da_hourly_parts.append(qh_agged)
        print(f"[*] Aggregated {len(qh_to_agg)} 15-min DA records to {len(qh_agged)} hourly")

    common_cols = None
    for part in da_hourly_parts:
        if common_cols is None:
            common_cols = set(part.columns)
        else:
            common_cols &= set(part.columns)
    common_cols = sorted(common_cols) if common_cols else []
    da_hourly = pd.concat([p[common_cols] for p in da_hourly_parts], ignore_index=True)

    # Calculate net cross-border flow on both DataFrames
    for df in [da_hourly, da_qh]:
        df['net_import'] = 0.0
        for col in ['flow_cz_sk', 'flow_pl_sk', 'flow_hu_sk']:
            if col in df.columns:
                df['net_import'] += df[col].fillna(0)
        for col in ['flow_sk_cz', 'flow_sk_pl', 'flow_sk_hu']:
            if col in df.columns:
                df['net_import'] -= df[col].fillna(0)

    da_hourly = da_hourly.sort_values(['date', 'hour']).reset_index(drop=True)
    da_qh = da_qh.sort_values('timestamp_qh').reset_index(drop=True)

    print(f"[+] DA: {len(da_hourly)} hourly + {len(da_qh)} QH records, "
          f"{da_hourly['date'].min()} to {da_hourly['date'].max()}")
    return da_hourly, da_qh


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

    # Rename columns per-file before concat (avoids duplicate column names)
    col_map = {
        # Slovak headers
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
        'Zúčtovacia cena PNRE (EUR/MWh)': 'imb_pnre_price',
        # English headers (2026 OKTE format)
        'Date': 'date',
        'Settlement Term': 'period',
        'System Imbalance (MWh)': 'imbalance_mwh',
        'System imbalance price (EUR/MWh)': 'imb_price',
        'Imbalance Settlement Price (EUR/MWh)': 'imb_settlement_price',
        'SREC (EUR/MWh)': 'imb_pnre_price',
    }

    for f in sorted(imb_files):
        print(f"    Loading {f.name}")
        df = pd.read_csv(f, sep=';', encoding='utf-8')
        df = df.rename(columns=col_map)
        dfs.append(df)

    imb = pd.concat(dfs, ignore_index=True)

    # Parse date (mixed formats: Slovak D.M.YYYY and English M/D/YYYY)
    date_raw = imb['date'].copy()
    imb['date'] = pd.to_datetime(date_raw, format='%d.%m.%Y', dayfirst=True, errors='coerce')
    mask_na = imb['date'].isna()
    if mask_na.any():
        imb.loc[mask_na, 'date'] = pd.to_datetime(date_raw[mask_na], format='%m/%d/%Y', errors='coerce')

    # Deduplicate overlapping files (keep last = prefer newer file)
    n_before = len(imb)
    imb = imb.drop_duplicates(subset=['date', 'period'], keep='last')
    n_dropped = n_before - len(imb)
    if n_dropped > 0:
        print(f"[*] Deduplicated {n_dropped} overlapping imbalance rows")

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
            # Detect format by which separator comes last
            if x.rfind('.') > x.rfind(','):
                # English: 1,072.3 -> dot is decimal
                x = x.replace(',', '')
            else:
                # European: 1.072,3 -> comma is decimal
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
    # NOTE: Use imb_settlement_price (not imb_price) for spreads.
    # imb_settlement_price = what a market participant actually pays/receives.
    # imb_price = marginal system imbalance price (not the trading-relevant price).
    # In deficit hours, settlement_price >> imb_price; in surplus, they're close.
    merged['spread_da_idm'] = merged['da_price'] - merged['idm_vwap']
    merged['spread_idm_imb'] = merged['idm_vwap'] - merged['imb_settlement_price']
    merged['spread_da_imb'] = merged['da_price'] - merged['imb_settlement_price']

    # Time features
    merged['date'] = merged['timestamp_hour'].dt.date
    merged['hour'] = merged['timestamp_hour'].dt.hour
    merged['dayofweek'] = merged['timestamp_hour'].dt.dayofweek
    merged['month'] = merged['timestamp_hour'].dt.month
    merged['is_weekend'] = merged['dayofweek'] >= 5

    print(f"[+] Merged: {len(merged)} hourly records")
    return merged


def merge_qh_prices(da_qh: pd.DataFrame, idm_qh: pd.DataFrame, imb: pd.DataFrame) -> pd.DataFrame:
    """Merge all market prices at quarter-hourly (15-min) resolution.

    All three sources are at native 15-min resolution (DA 2024-2025 hourly
    rows are pre-broadcast to QH by load_da_market).
    """
    print("[*] Merging prices at quarter-hourly resolution...")

    # Start from DA QH as spine (has the broadest date range)
    da_cols = ['timestamp_qh', 'timestamp_hour', 'da_price', 'da_demand', 'da_supply', 'net_import']
    merged = da_qh[[c for c in da_cols if c in da_qh.columns]].copy()
    merged = merged.drop_duplicates(subset='timestamp_qh')

    # Add any QH timestamps from imbalance/IDM not in DA
    qh_extra = set()
    if not imb.empty:
        qh_extra.update(imb['timestamp_15min'].dropna())
    if not idm_qh.empty and 'timestamp_15min' in idm_qh.columns:
        qh_extra.update(idm_qh['timestamp_15min'].dropna())
    qh_extra -= set(merged['timestamp_qh'])
    if qh_extra:
        extra_df = pd.DataFrame({'timestamp_qh': sorted(qh_extra)})
        extra_df['timestamp_hour'] = extra_df['timestamp_qh'].dt.floor('h')
        merged = pd.concat([merged, extra_df], ignore_index=True)

    # Join IDM at 15-min
    if not idm_qh.empty and 'timestamp_15min' in idm_qh.columns:
        idm_cols = ['timestamp_15min', 'idm_vwap']
        for c in ['idm_volume_mwh', 'idm_volume_mw']:
            if c in idm_qh.columns:
                idm_cols.append(c)
        idm_dedup = idm_qh[idm_cols].drop_duplicates(subset='timestamp_15min')
        merged = merged.merge(idm_dedup, left_on='timestamp_qh', right_on='timestamp_15min', how='left')
        merged = merged.drop(columns=['timestamp_15min'], errors='ignore')

    # Join imbalance at 15-min
    if not imb.empty:
        imb_cols = ['timestamp_15min', 'imb_price', 'imb_settlement_price', 'imbalance_mwh']
        imb_dedup = imb[[c for c in imb_cols if c in imb.columns]].drop_duplicates(subset='timestamp_15min')
        merged = merged.merge(imb_dedup, left_on='timestamp_qh', right_on='timestamp_15min', how='left')
        merged = merged.drop(columns=['timestamp_15min'], errors='ignore')

    merged = merged.sort_values('timestamp_qh').reset_index(drop=True)

    # Price spreads (settlement price for trading-relevant spreads)
    merged['spread_da_idm'] = merged['da_price'] - merged['idm_vwap']
    merged['spread_idm_imb'] = merged['idm_vwap'] - merged['imb_settlement_price']
    merged['spread_da_imb'] = merged['da_price'] - merged['imb_settlement_price']

    # Time features
    merged['date'] = merged['timestamp_qh'].dt.date
    merged['hour'] = merged['timestamp_qh'].dt.hour
    merged['minute'] = merged['timestamp_qh'].dt.minute
    merged['period'] = merged['hour'] * 4 + merged['minute'] // 15 + 1
    merged['dayofweek'] = merged['timestamp_qh'].dt.dayofweek
    merged['month'] = merged['timestamp_qh'].dt.month
    merged['is_weekend'] = merged['dayofweek'] >= 5

    print(f"[+] Merged: {len(merged)} quarter-hourly records")
    return merged


def load_all_prices(save: bool = True):
    """Load all market price data and create merged datasets.

    Returns:
        Tuple of (da_hourly, da_qh, idm_hourly, idm_qh, imbalance, merged_hourly, merged_qh)
    """
    da_hourly, da_qh = load_da_market()
    idm_hourly = load_idm_market(resolution='60')
    idm_qh = load_idm_market(resolution='15')
    imb = load_imbalance_prices()

    merged_hourly = merge_hourly_prices(da_hourly, idm_hourly, imb)
    merged_qh = merge_qh_prices(da_qh, idm_qh, imb)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        merged_hourly.to_csv(OUTPUT_DIR / "hourly_market_prices.csv", index=False)
        print(f"[+] Saved {OUTPUT_DIR / 'hourly_market_prices.csv'}")
        merged_qh.to_csv(OUTPUT_DIR / "qh_market_prices.csv", index=False)
        print(f"[+] Saved {OUTPUT_DIR / 'qh_market_prices.csv'}")

    return da_hourly, da_qh, idm_hourly, idm_qh, imb, merged_hourly, merged_qh


if __name__ == "__main__":
    da_hourly, da_qh, idm_hourly, idm_qh, imb, merged_hourly, merged_qh = load_all_prices()

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nDA Market:        {len(da_hourly):,} hourly + {len(da_qh):,} QH records")
    print(f"IDM Market (60m): {len(idm_hourly):,} hourly records")
    print(f"IDM Market (15m): {len(idm_qh):,} QH records")
    print(f"Imbalance:        {len(imb):,} QH records")
    print(f"Merged Hourly:    {len(merged_hourly):,} records")
    print(f"Merged QH:        {len(merged_qh):,} records")

    print("\n--- QH Price Statistics (EUR/MWh) ---")
    for col, name in [('da_price', 'DA Price'), ('idm_vwap', 'IDM VWAP'),
                       ('imb_settlement_price', 'Imb Settlement')]:
        if col in merged_qh.columns:
            s = merged_qh[col].dropna()
            print(f"{name:15s}: mean={s.mean():7.2f}, std={s.std():7.2f}, "
                  f"min={s.min():7.2f}, max={s.max():7.2f}, n={len(s):,}")

    print("\n--- QH Price Spreads (EUR/MWh) ---")
    for col, name in [('spread_da_idm', 'DA - IDM'), ('spread_idm_imb', 'IDM - Imb'), ('spread_da_imb', 'DA - Imb')]:
        if col in merged_qh.columns:
            s = merged_qh[col].dropna()
            print(f"{name:15s}: mean={s.mean():7.2f}, std={s.std():7.2f}, "
                  f"min={s.min():7.2f}, max={s.max():7.2f}, n={len(s):,}")
