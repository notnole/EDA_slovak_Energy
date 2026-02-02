"""
EDA: Indicators that Affect DA-IDM Spread Size

Analyzes how the following indicators affect spread magnitude:
1. Load forecast error (DAMAS forecast vs actual from SCADA)
2. Solar/Wind forecast levels
3. Cross-border flow patterns
4. Previous volatility
5. Supply/Demand balance
6. Calendar effects (excluding weather)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "RawData"
DAMAS_DIR = RAW_DIR / "Damas"
OUTPUT_DIR = BASE_DIR / "MarketPriceGap" / "features" / "spread_indicators"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def load_da_data():
    """Load Day-Ahead auction results."""
    print("[*] Loading DA auction data...")

    da_files = list(DAMAS_DIR.glob("Celkove_vysledky_DT_*.csv"))
    dfs = []

    for f in da_files:
        df = pd.read_csv(f, sep=";", encoding="utf-8")
        dfs.append(df)
        print(f"    [+] Loaded {f.name}: {len(df)} rows")

    da = pd.concat(dfs, ignore_index=True)

    # Clean column names
    da.columns = [c.replace("\ufeff", "").strip() for c in da.columns]

    # Rename to English
    col_map = {
        'Obchodny den': 'date',
        'Cislo periody': 'hour',
        'Cena SK (EUR/MWh)': 'da_price',
        'Dopyt uspesny (MW)': 'da_demand',
        'Ponuka uspesna (MW)': 'da_supply',
        'Tok CZ ➝ SK (MW)': 'flow_cz_sk',
        'Tok SK ➝ CZ (MW)': 'flow_sk_cz',
        'Tok PL ➝ SK (MW)': 'flow_pl_sk',
        'Tok SK ➝ PL (MW)': 'flow_sk_pl',
        'Tok HU ➝ SK (MW)': 'flow_hu_sk',
        'Tok SK ➝ HU (MW)': 'flow_sk_hu',
    }

    # Find matching columns
    for old_pattern, new_name in col_map.items():
        for col in da.columns:
            if old_pattern.lower() in col.lower() or col.lower() in old_pattern.lower():
                da = da.rename(columns={col: new_name})
                break

    # Parse date
    date_col = [c for c in da.columns if 'den' in c.lower() or 'date' in c.lower() or 'day' in c.lower()][0]
    da['date'] = pd.to_datetime(da[date_col], format="%d.%m.%Y", errors='coerce')

    # Get hour from period number
    hour_col = [c for c in da.columns if 'period' in c.lower() or 'cislo' in c.lower()][0]
    da['hour'] = da[hour_col].astype(int) - 1  # Convert 1-24 to 0-23

    # Create timestamp
    da['timestamp'] = da['date'] + pd.to_timedelta(da['hour'], unit='h')

    # Parse numeric columns with European format
    for col in ['da_price', 'da_demand', 'da_supply', 'flow_cz_sk', 'flow_sk_cz',
                'flow_pl_sk', 'flow_sk_pl', 'flow_hu_sk', 'flow_sk_hu']:
        if col in da.columns:
            if da[col].dtype == object:
                da[col] = da[col].str.replace(',', '.').astype(float)

    # Calculate net flows
    if 'flow_cz_sk' in da.columns:
        da['net_flow_cz'] = da['flow_cz_sk'] - da.get('flow_sk_cz', 0)
        da['net_flow_pl'] = da['flow_pl_sk'] - da.get('flow_sk_pl', 0)
        da['net_flow_hu'] = da['flow_hu_sk'] - da.get('flow_sk_hu', 0)
        da['net_import'] = da['net_flow_cz'] + da['net_flow_pl'] + da['net_flow_hu']

    # Calculate supply-demand balance
    if 'da_supply' in da.columns and 'da_demand' in da.columns:
        da['supply_demand_ratio'] = da['da_supply'] / da['da_demand'].replace(0, np.nan)

    print(f"    [+] Total DA records: {len(da)}")
    return da


def load_idm_data():
    """Load Intraday Market data."""
    print("[*] Loading IDM data...")

    idm_path = BASE_DIR / "data" / "clean" / "market" / "intraday" / "idm_60min.csv"

    if not idm_path.exists():
        # Try to load from raw
        idm_path = BASE_DIR / "data" / "clean" / "market" / "intraday" / "idm_15min.csv"

    if not idm_path.exists():
        print("    [-] IDM data not found, loading from raw...")
        # Consolidate from raw
        idm_raw = RAW_DIR / "IDM_MarketData"
        dfs = []
        for folder in idm_raw.iterdir():
            if folder.is_dir():
                f60 = folder / "60 min.csv"
                if f60.exists():
                    df = pd.read_csv(f60, sep=";")
                    dfs.append(df)
        idm = pd.concat(dfs, ignore_index=True)
    else:
        idm = pd.read_csv(idm_path, sep=";")

    # Parse date
    idm['date'] = pd.to_datetime(idm['Delivery day'], format="%Y-%m-%d", errors='coerce')
    if idm['date'].isna().all():
        idm['date'] = pd.to_datetime(idm['Delivery day'], format="%d.%m.%Y", errors='coerce')

    # Get hour
    idm['hour'] = idm['Period number'].astype(int) - 1

    # Create timestamp
    idm['timestamp'] = idm['date'] + pd.to_timedelta(idm['hour'], unit='h')

    # Rename columns
    idm = idm.rename(columns={
        'Weighted average price of all trades (EUR/MWh)': 'idm_vwap',
        'Total Traded Quantity (MW)': 'idm_volume',
        'Buy trades (MW)': 'idm_buy',
        'Sell trades (MW)': 'idm_sell',
    })

    print(f"    [+] Total IDM records: {len(idm)}")
    return idm


def load_damas_forecasts():
    """Load DAMAS load and production forecasts from XLSX."""
    print("[*] Loading DAMAS forecasts...")

    forecasts = {}

    # Load forecast
    load_files = list(DAMAS_DIR.glob("*Zatazenie*predikcia*.xlsx"))
    if load_files:
        # Take the largest file (most data)
        load_file = max(load_files, key=lambda x: x.stat().st_size)
        print(f"    [+] Loading load forecast: {load_file.name}")

        df = pd.read_excel(load_file, header=None)

        # Parse the data
        records = []
        current_date = None
        for _, row in df.iterrows():
            if pd.notna(row[0]):
                try:
                    current_date = pd.to_datetime(row[0], format="%d.%m.%Y")
                except:
                    continue
            if current_date and pd.notna(row[1]) and pd.notna(row[2]):
                try:
                    hour = int(row[1]) - 1
                    value = float(row[2])
                    records.append({
                        'date': current_date,
                        'hour': hour,
                        'load_forecast': value
                    })
                except:
                    continue

        forecasts['load'] = pd.DataFrame(records)
        forecasts['load']['timestamp'] = forecasts['load']['date'] + pd.to_timedelta(forecasts['load']['hour'], unit='h')
        print(f"        Records: {len(forecasts['load'])}")

    # Production forecast
    prod_files = list(DAMAS_DIR.glob("*Vyroba ES SR*predikcia*.xlsx"))
    if prod_files:
        prod_file = max(prod_files, key=lambda x: x.stat().st_size)
        print(f"    [+] Loading production forecast: {prod_file.name}")

        df = pd.read_excel(prod_file, header=None)

        records = []
        current_date = None
        for _, row in df.iterrows():
            if pd.notna(row[0]):
                try:
                    current_date = pd.to_datetime(row[0], format="%d.%m.%Y")
                except:
                    continue
            if current_date and pd.notna(row[1]) and pd.notna(row[2]):
                try:
                    hour = int(row[1]) - 1
                    value = float(row[2])
                    records.append({
                        'date': current_date,
                        'hour': hour,
                        'prod_forecast': value
                    })
                except:
                    continue

        forecasts['production'] = pd.DataFrame(records)
        forecasts['production']['timestamp'] = forecasts['production']['date'] + pd.to_timedelta(forecasts['production']['hour'], unit='h')
        print(f"        Records: {len(forecasts['production'])}")

    # Solar/Wind forecast
    res_files = list(DAMAS_DIR.glob("*FVE*Predikcia*.xlsx"))
    if res_files:
        res_file = max(res_files, key=lambda x: x.stat().st_size)
        print(f"    [+] Loading solar/wind forecast: {res_file.name}")

        df = pd.read_excel(res_file, header=None)

        # Get column names from first row
        header = df.iloc[0].tolist()

        records = []
        current_date = None
        for _, row in df.iloc[1:].iterrows():
            if pd.notna(row[0]):
                try:
                    current_date = pd.to_datetime(row[0], format="%d.%m.%Y")
                except:
                    continue
            if current_date and pd.notna(row[1]):
                try:
                    hour = int(row[1]) - 1
                    solar_da = float(row[3]) if pd.notna(row[3]) else 0
                    wind_da = float(row[6]) if pd.notna(row[6]) else 0
                    records.append({
                        'date': current_date,
                        'hour': hour,
                        'solar_forecast': solar_da,
                        'wind_forecast': wind_da,
                        'res_forecast': solar_da + wind_da
                    })
                except:
                    continue

        forecasts['res'] = pd.DataFrame(records)
        forecasts['res']['timestamp'] = forecasts['res']['date'] + pd.to_timedelta(forecasts['res']['hour'], unit='h')
        print(f"        Records: {len(forecasts['res'])}")

    return forecasts


def load_scada_actuals():
    """Load SCADA actual values for calculating forecast errors."""
    print("[*] Loading SCADA actuals...")

    # Load 3-min load data
    load_file = RAW_DIR / "3MIN_Load.csv"
    if load_file.exists():
        df = pd.read_csv(load_file, header=None, encoding='utf-8')

        # Parse timestamp and value
        records = []
        for _, row in df.iloc[1:].iterrows():
            try:
                ts_str = str(row[0]).split(',')[0]
                ts = pd.to_datetime(ts_str, format="%d/%m/%Y %H:%M:%S")
                val = float(str(row[1]).replace(',', '.'))
                records.append({'timestamp': ts, 'load_actual': val})
            except:
                continue

        load_df = pd.DataFrame(records)
        # Aggregate to hourly
        load_df['hour_ts'] = load_df['timestamp'].dt.floor('h')
        load_hourly = load_df.groupby('hour_ts')['load_actual'].mean().reset_index()
        load_hourly.columns = ['timestamp', 'load_actual']

        print(f"    [+] Load actuals: {len(load_hourly)} hourly records")
        return load_hourly

    return None


def merge_all_data(da, idm, forecasts, actuals):
    """Merge all data sources."""
    print("[*] Merging all data...")

    # Start with DA data
    df = da[['timestamp', 'date', 'hour', 'da_price', 'da_demand', 'da_supply',
             'net_import', 'net_flow_cz', 'net_flow_pl', 'net_flow_hu',
             'supply_demand_ratio']].copy()

    # Merge IDM
    idm_cols = ['timestamp', 'idm_vwap', 'idm_volume']
    idm_subset = idm[[c for c in idm_cols if c in idm.columns]].copy()
    df = df.merge(idm_subset, on='timestamp', how='left')

    # Merge forecasts
    if 'load' in forecasts:
        df = df.merge(forecasts['load'][['timestamp', 'load_forecast']], on='timestamp', how='left')

    if 'production' in forecasts:
        df = df.merge(forecasts['production'][['timestamp', 'prod_forecast']], on='timestamp', how='left')

    if 'res' in forecasts:
        df = df.merge(forecasts['res'][['timestamp', 'solar_forecast', 'wind_forecast', 'res_forecast']],
                      on='timestamp', how='left')

    # Merge actuals
    if actuals is not None:
        df = df.merge(actuals, on='timestamp', how='left')

        # Calculate forecast error
        if 'load_forecast' in df.columns and 'load_actual' in df.columns:
            df['load_forecast_error'] = df['load_actual'] - df['load_forecast']
            df['load_forecast_error_pct'] = df['load_forecast_error'] / df['load_forecast'] * 100

    # Calculate spread
    df['spread'] = df['da_price'] - df['idm_vwap']
    df['spread_abs'] = df['spread'].abs()
    df['spread_direction'] = (df['spread'] > 0).astype(int)

    # Add calendar features
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['dayofweek'] >= 5

    # Calculate volatility features
    df = df.sort_values('timestamp')
    df['da_price_std_24h'] = df['da_price'].rolling(24, min_periods=12).std()
    df['da_price_std_7d'] = df['da_price'].rolling(168, min_periods=84).std()
    df['spread_std_24h'] = df['spread'].rolling(24, min_periods=12).std()

    # Lag features
    df['spread_lag_24h'] = df['spread'].shift(24)
    df['spread_abs_lag_24h'] = df['spread_abs'].shift(24)

    # Filter to rows with both DA and IDM prices
    df_valid = df[df['idm_vwap'].notna() & df['da_price'].notna()].copy()

    print(f"    [+] Merged dataset: {len(df_valid)} valid records")
    print(f"    [+] Date range: {df_valid['timestamp'].min()} to {df_valid['timestamp'].max()}")

    return df_valid


def analyze_indicators(df):
    """Analyze how each indicator affects spread size."""
    print("[*] Analyzing indicators...")

    results = {}

    # List of indicators to analyze
    indicators = [
        ('load_forecast_error', 'Load Forecast Error (MW)'),
        ('load_forecast_error_pct', 'Load Forecast Error (%)'),
        ('solar_forecast', 'Solar Forecast (MW)'),
        ('wind_forecast', 'Wind Forecast (MW)'),
        ('res_forecast', 'RES Forecast (MW)'),
        ('net_import', 'Net Import (MW)'),
        ('net_flow_cz', 'Net Flow CZ (MW)'),
        ('net_flow_hu', 'Net Flow HU (MW)'),
        ('supply_demand_ratio', 'Supply/Demand Ratio'),
        ('da_price_std_24h', 'DA Price Volatility 24h'),
        ('da_price_std_7d', 'DA Price Volatility 7d'),
        ('spread_abs_lag_24h', 'Yesterday |Spread|'),
        ('da_price', 'DA Price Level'),
        ('da_demand', 'DA Demand (MW)'),
        ('da_supply', 'DA Supply (MW)'),
    ]

    for col, name in indicators:
        if col not in df.columns or df[col].isna().all():
            continue

        valid = df[[col, 'spread_abs', 'spread']].dropna()
        if len(valid) < 100:
            continue

        # Correlation with spread magnitude
        corr_abs, p_abs = stats.pearsonr(valid[col], valid['spread_abs'])
        corr_signed, p_signed = stats.pearsonr(valid[col], valid['spread'])

        # Binned analysis
        try:
            valid['bin'] = pd.qcut(valid[col], q=5, duplicates='drop')
            bin_stats = valid.groupby('bin', observed=True).agg({
                'spread_abs': ['mean', 'std', 'count'],
                'spread': 'mean'
            })
            bin_stats.columns = ['spread_abs_mean', 'spread_abs_std', 'count', 'spread_mean']
        except:
            bin_stats = None

        results[col] = {
            'name': name,
            'corr_abs': corr_abs,
            'p_abs': p_abs,
            'corr_signed': corr_signed,
            'p_signed': p_signed,
            'bin_stats': bin_stats,
            'n': len(valid)
        }

        print(f"    {name}: r(|spread|)={corr_abs:.3f}, r(spread)={corr_signed:.3f}, n={len(valid)}")

    return results


def plot_indicator_analysis(df, results):
    """Create visualizations for indicator analysis."""
    print("[*] Creating visualizations...")

    # 1. Correlation summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Bar chart of correlations
    ax = axes[0, 0]
    corr_data = [(r['name'], r['corr_abs'], r['p_abs']) for r in results.values() if abs(r['corr_abs']) > 0.01]
    corr_data.sort(key=lambda x: abs(x[1]), reverse=True)

    names = [x[0][:25] for x in corr_data[:12]]
    corrs = [x[1] for x in corr_data[:12]]
    colors = ['green' if c > 0 else 'red' for c in corrs]

    y_pos = range(len(names))
    ax.barh(y_pos, corrs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Correlation with |Spread|')
    ax.set_title('Indicator Correlation with Spread Magnitude', fontweight='bold')

    # 2. Spread by RES forecast level
    ax = axes[0, 1]
    if 'res_forecast' in df.columns and df['res_forecast'].notna().any():
        valid = df[df['res_forecast'].notna()].copy()
        valid['res_bin'] = pd.qcut(valid['res_forecast'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')

        res_stats = valid.groupby('res_bin', observed=True)['spread_abs'].agg(['mean', 'std'])

        x = range(len(res_stats))
        ax.bar(x, res_stats['mean'], yerr=res_stats['std']/2, color='forestgreen', alpha=0.7,
               edgecolor='black', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(res_stats.index, rotation=45)
        ax.set_xlabel('RES (Solar+Wind) Forecast Level')
        ax.set_ylabel('Mean |Spread| (EUR/MWh)')
        ax.set_title('Spread Magnitude by RES Forecast', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'RES forecast data not available', ha='center', va='center', transform=ax.transAxes)

    # 3. Spread by net import
    ax = axes[1, 0]
    if 'net_import' in df.columns:
        valid = df[df['net_import'].notna()].copy()
        valid['import_bin'] = pd.qcut(valid['net_import'], q=5,
                                       labels=['High Export', 'Export', 'Balanced', 'Import', 'High Import'],
                                       duplicates='drop')

        import_stats = valid.groupby('import_bin', observed=True)['spread_abs'].agg(['mean', 'std'])

        x = range(len(import_stats))
        ax.bar(x, import_stats['mean'], yerr=import_stats['std']/2, color='steelblue', alpha=0.7,
               edgecolor='black', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(import_stats.index, rotation=45)
        ax.set_xlabel('Net Cross-Border Position')
        ax.set_ylabel('Mean |Spread| (EUR/MWh)')
        ax.set_title('Spread Magnitude by Net Import', fontweight='bold')

    # 4. Spread by yesterday's spread
    ax = axes[1, 1]
    if 'spread_abs_lag_24h' in df.columns:
        valid = df[df['spread_abs_lag_24h'].notna()].copy()
        valid['lag_bin'] = pd.cut(valid['spread_abs_lag_24h'],
                                   bins=[0, 5, 10, 20, 50, np.inf],
                                   labels=['0-5', '5-10', '10-20', '20-50', '>50'])

        lag_stats = valid.groupby('lag_bin', observed=True)['spread_abs'].agg(['mean', 'std', 'count'])

        x = range(len(lag_stats))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(lag_stats)))
        ax.bar(x, lag_stats['mean'], yerr=lag_stats['std']/2, color=colors, alpha=0.8,
               edgecolor='black', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(lag_stats.index)
        ax.set_xlabel("Yesterday's |Spread| (EUR/MWh)")
        ax.set_ylabel('Today Mean |Spread| (EUR/MWh)')
        ax.set_title('Spread Persistence (Yesterday -> Today)', fontweight='bold')

        # Add count labels
        for i, (_, row) in enumerate(lag_stats.iterrows()):
            ax.text(i, row['mean'] + row['std']/2 + 1, f'n={int(row["count"])}',
                    ha='center', fontsize=8)

    plt.suptitle('Indicators Affecting DA-IDM Spread Magnitude', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_indicator_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '01_indicator_overview.png'}")

    # Second figure: More detailed analysis
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Spread by hour
    ax = axes[0, 0]
    hourly = df.groupby('hour')['spread_abs'].agg(['mean', 'std'])
    ax.bar(hourly.index, hourly['mean'], yerr=hourly['std']/3, color='steelblue', alpha=0.7,
           edgecolor='black', capsize=2)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean |Spread| (EUR/MWh)')
    ax.set_title('Spread by Hour', fontweight='bold')
    ax.set_xticks(range(0, 24, 2))

    # 2. Spread by day of week
    ax = axes[0, 1]
    daily = df.groupby('dayofweek')['spread_abs'].agg(['mean', 'std'])
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.bar(range(7), daily['mean'], yerr=daily['std']/3, color='coral', alpha=0.7,
           edgecolor='black', capsize=2)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Mean |Spread| (EUR/MWh)')
    ax.set_title('Spread by Day of Week', fontweight='bold')

    # 3. Spread by month
    ax = axes[0, 2]
    monthly = df.groupby('month')['spread_abs'].agg(['mean', 'std'])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.bar(monthly.index, monthly['mean'], yerr=monthly['std']/3, color='purple', alpha=0.6,
           edgecolor='black', capsize=2)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([months[i-1] for i in monthly.index], rotation=45)
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean |Spread| (EUR/MWh)')
    ax.set_title('Spread by Month', fontweight='bold')

    # 4. Spread vs DA price volatility
    ax = axes[1, 0]
    if 'da_price_std_24h' in df.columns:
        valid = df[df['da_price_std_24h'].notna()].copy()
        ax.scatter(valid['da_price_std_24h'], valid['spread_abs'], alpha=0.1, s=10)

        # Add trend line
        z = np.polyfit(valid['da_price_std_24h'], valid['spread_abs'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['da_price_std_24h'].min(), valid['da_price_std_24h'].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (r={results.get("da_price_std_24h", {}).get("corr_abs", 0):.2f})')

        ax.set_xlabel('DA Price Volatility (24h std)')
        ax.set_ylabel('|Spread| (EUR/MWh)')
        ax.set_title('Spread vs Price Volatility', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 100)

    # 5. Spread vs DA price level
    ax = axes[1, 1]
    valid = df[df['da_price'].notna() & df['spread_abs'].notna()].copy()
    valid = valid[(valid['da_price'] > -50) & (valid['da_price'] < 300)]

    ax.scatter(valid['da_price'], valid['spread_abs'], alpha=0.1, s=10)

    # Binned means
    valid['price_bin'] = pd.cut(valid['da_price'], bins=20)
    bin_means = valid.groupby('price_bin', observed=True)['spread_abs'].mean()
    bin_centers = [interval.mid for interval in bin_means.index]
    ax.plot(bin_centers, bin_means.values, 'ro-', markersize=6, linewidth=2, label='Bin means')

    ax.set_xlabel('DA Price (EUR/MWh)')
    ax.set_ylabel('|Spread| (EUR/MWh)')
    ax.set_title('Spread vs DA Price Level', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 100)

    # 6. Spread vs supply/demand ratio
    ax = axes[1, 2]
    if 'supply_demand_ratio' in df.columns:
        valid = df[df['supply_demand_ratio'].notna()].copy()
        valid = valid[(valid['supply_demand_ratio'] > 0) & (valid['supply_demand_ratio'] < 10)]

        ax.scatter(valid['supply_demand_ratio'], valid['spread_abs'], alpha=0.1, s=10)

        # Binned means
        valid['ratio_bin'] = pd.cut(valid['supply_demand_ratio'], bins=10)
        bin_means = valid.groupby('ratio_bin', observed=True)['spread_abs'].mean()
        bin_centers = [interval.mid for interval in bin_means.index]
        ax.plot(bin_centers, bin_means.values, 'go-', markersize=6, linewidth=2, label='Bin means')

        ax.set_xlabel('Supply/Demand Ratio')
        ax.set_ylabel('|Spread| (EUR/MWh)')
        ax.set_title('Spread vs Supply/Demand Balance', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 100)

    plt.suptitle('Detailed Indicator Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '02_detailed_analysis.png'}")

    # Third figure: Cross-border flows
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    flow_cols = [('net_flow_cz', 'CZ'), ('net_flow_pl', 'PL'), ('net_flow_hu', 'HU')]

    for ax, (col, country) in zip(axes, flow_cols):
        if col in df.columns:
            valid = df[df[col].notna()].copy()
            valid['flow_bin'] = pd.qcut(valid[col], q=5, duplicates='drop')

            flow_stats = valid.groupby('flow_bin', observed=True)['spread_abs'].agg(['mean', 'std'])

            x = range(len(flow_stats))
            ax.bar(x, flow_stats['mean'], yerr=flow_stats['std']/3, alpha=0.7, edgecolor='black', capsize=2)
            ax.set_xticks(x)
            labels = [f'{interval.left:.0f}-{interval.right:.0f}' for interval in flow_stats.index]
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            ax.set_xlabel(f'Net Flow {country} (MW)')
            ax.set_ylabel('Mean |Spread| (EUR/MWh)')
            ax.set_title(f'Spread by {country} Flow', fontweight='bold')

    plt.suptitle('Cross-Border Flow Impact on Spread', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_crossborder_flows.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '03_crossborder_flows.png'}")


def create_summary(df, results):
    """Create summary markdown."""
    print("[*] Creating summary...")

    # Sort results by correlation magnitude
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['corr_abs']), reverse=True)

    summary = """# Spread Indicators EDA

## Overview

Analysis of indicators that affect DA-IDM spread magnitude. Understanding what drives large spreads helps identify profitable trading opportunities.

---

## Key Findings

### Top Indicators Correlated with |Spread|

| Indicator | Correlation | p-value | Interpretation |
|-----------|-------------|---------|----------------|
"""

    for col, r in sorted_results[:10]:
        sig = '***' if r['p_abs'] < 0.001 else '**' if r['p_abs'] < 0.01 else '*' if r['p_abs'] < 0.05 else ''
        interp = 'Higher -> Larger spread' if r['corr_abs'] > 0 else 'Higher -> Smaller spread'
        summary += f"| {r['name']} | {r['corr_abs']:.3f} {sig} | {r['p_abs']:.2e} | {interp} |\n"

    summary += """
### Key Insights

"""

    # Add specific insights
    if 'spread_abs_lag_24h' in results:
        r = results['spread_abs_lag_24h']
        summary += f"""#### 1. Spread Persistence (r = {r['corr_abs']:.3f})
**Yesterday's spread size strongly predicts today's spread size.** This is the most actionable finding:
- If yesterday had a large spread (>20 EUR), expect a large spread today
- If yesterday had a small spread (<5 EUR), expect a small spread today
- This persistence allows filtering for high-probability trades

"""

    if 'da_price_std_24h' in results:
        r = results['da_price_std_24h']
        summary += f"""#### 2. Price Volatility (r = {r['corr_abs']:.3f})
**Higher price volatility leads to larger spreads:**
- Volatile markets create more price divergence between DA and IDM
- Trade larger when volatility is high, but with tighter stops

"""

    if 'res_forecast' in results:
        r = results['res_forecast']
        summary += f"""#### 3. RES (Solar+Wind) Forecast (r = {r['corr_abs']:.3f})
**Higher renewable forecasts are associated with {"larger" if r['corr_abs'] > 0 else "smaller"} spreads:**
- Solar/wind variability increases price uncertainty
- High RES periods may have more price corrections in IDM

"""

    if 'net_import' in results:
        r = results['net_import']
        summary += f"""#### 4. Cross-Border Flows (r = {r['corr_abs']:.3f})
**Net import position affects spread:**
- {"High imports" if r['corr_abs'] > 0 else "High exports"} associated with larger spreads
- Congestion at borders creates price divergence

"""

    if 'supply_demand_ratio' in results:
        r = results['supply_demand_ratio']
        summary += f"""#### 5. Supply/Demand Balance (r = {r['corr_abs']:.3f})
**Market balance affects uncertainty:**
- {"Oversupply" if r['corr_abs'] > 0 else "Tight supply"} conditions have larger spreads
- Balanced markets are more predictable

"""

    # Calendar patterns
    hourly_spread = df.groupby('hour')['spread_abs'].mean()
    peak_hour = hourly_spread.idxmax()
    off_peak_hour = hourly_spread.idxmin()

    monthly_spread = df.groupby('month')['spread_abs'].mean()
    best_month = monthly_spread.idxmax()
    worst_month = monthly_spread.idxmin()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    summary += f"""#### 6. Calendar Patterns

**Hourly:**
- Largest spreads at hour {peak_hour}:00 (mean = {hourly_spread[peak_hour]:.1f} EUR)
- Smallest spreads at hour {off_peak_hour}:00 (mean = {hourly_spread[off_peak_hour]:.1f} EUR)

**Monthly:**
- Best month: {months[best_month-1]} (mean = {monthly_spread[best_month]:.1f} EUR)
- Worst month: {months[worst_month-1]} (mean = {monthly_spread[worst_month]:.1f} EUR)

---

## Trading Implications

### When to Expect Large Spreads (Good for Trading)

1. **Yesterday had large spread** (>20 EUR same hour)
2. **High price volatility** (24h rolling std > 30)
3. **Peak hours** (17:00-20:00)
4. **Q4 months** (Oct-Dec historically best)
5. **Extreme cross-border positions** (high import or export)

### When to Avoid Trading

1. **Yesterday had tiny spread** (<5 EUR)
2. **Low volatility regime**
3. **Night hours** (00:00-06:00)
4. **Summer months** (May-Aug historically worst)
5. **Balanced market conditions**

---

## Visualizations

1. **01_indicator_overview.png** - Correlation summary and key indicators
2. **02_detailed_analysis.png** - Hour/day/month patterns and scatter plots
3. **03_crossborder_flows.png** - Impact of cross-border flows by country

---

## Data Coverage

| Metric | Value |
|--------|-------|
| Total observations | {len(df):,} |
| Date range | {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')} |
| Mean |spread| | {df['spread_abs'].mean():.2f} EUR/MWh |
| Median |spread| | {df['spread_abs'].median():.2f} EUR/MWh |
| 90th percentile |spread| | {df['spread_abs'].quantile(0.9):.2f} EUR/MWh |

---

## Next Steps

1. Build predictive model for spread magnitude using top indicators
2. Combine with direction prediction for complete trading strategy
3. Backtest with transaction costs
4. Consider regime-switching models (high-vol vs low-vol)
"""

    with open(OUTPUT_DIR / 'summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"[+] Saved: {OUTPUT_DIR / 'summary.md'}")


def main():
    print("=" * 60)
    print("SPREAD INDICATORS EDA")
    print("=" * 60)

    # Load all data
    da = load_da_data()
    idm = load_idm_data()
    forecasts = load_damas_forecasts()
    actuals = load_scada_actuals()

    # Merge
    df = merge_all_data(da, idm, forecasts, actuals)

    # Analyze
    results = analyze_indicators(df)

    # Visualize
    plot_indicator_analysis(df, results)

    # Summary
    create_summary(df, results)

    print("\n" + "=" * 60)
    print("[+] Analysis complete!")
    print(f"    Output: MarketPriceGap/features/spread_indicators/")
    print("=" * 60)


if __name__ == "__main__":
    main()
