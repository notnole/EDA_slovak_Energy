"""
Phase 1+2: IDM Order Book Data Overview & Price Formation Analysis

Queries the local DB directly to understand:
  1. Data coverage (date range, row counts, completeness)
  2. Order book structure (products, depth, update frequency)
  3. Price convergence curves (IDM price evolution vs time-to-delivery)
  4. Bid-ask spread dynamics
  5. DA vs IDM price relationship over trading horizon

Tables used:
  - db_ems.vdt_isot_knihaobjednavok_best (order book snapshots)
  - db_ems.vdt_isot_lasttrades (executed trades)
"""

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import timedelta

# --- Config ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "DB_EMS",
    "user": "postgres",
    "password": "Kapitan1",
}

# DA prices path
DA_PRICES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed" / "hourly_market_prices.csv"

plt.rcParams.update({
    "figure.figsize": (14, 8),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def get_conn():
    return psycopg2.connect(**DB)


# ============================================================
# PHASE 1: Data Coverage & Structure
# ============================================================

def phase1_data_overview():
    """Understand what we have in the order book."""
    print("=" * 70)
    print("PHASE 1: IDM Order Book Data Overview")
    print("=" * 70)

    with get_conn() as conn:
        # 1a. Total row count and date range
        print("\n[*] Order book (knihaobjednavok_best):")
        df = pd.read_sql("""
            SELECT
                COUNT(*) as total_rows,
                MIN(tradeday::date) as min_date,
                MAX(tradeday::date) as max_date,
                COUNT(DISTINCT tradeday::date) as n_days
            FROM db_ems.vdt_isot_knihaobjednavok_best
        """, conn)
        print(f"    Total rows: {df['total_rows'].iloc[0]:,.0f}")
        print(f"    Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
        print(f"    Unique days: {df['n_days'].iloc[0]}")

        # 1b. Breakdown by delivery duration
        print("\n[*] By delivery duration:")
        df_dur = pd.read_sql("""
            SELECT
                deliverydur,
                COUNT(*) as rows,
                COUNT(DISTINCT tradeday::date) as days,
                MIN(tradeday::date) as from_date,
                MAX(tradeday::date) as to_date
            FROM db_ems.vdt_isot_knihaobjednavok_best
            GROUP BY deliverydur
            ORDER BY deliverydur
        """, conn)
        for _, r in df_dur.iterrows():
            print(f"    {r['deliverydur']}min: {r['rows']:>12,} rows, {r['days']} days ({r['from_date']} to {r['to_date']})")

        # 1c. Depth levels
        print("\n[*] By depth level:")
        df_depth = pd.read_sql("""
            SELECT id_depth, COUNT(*) as rows
            FROM db_ems.vdt_isot_knihaobjednavok_best
            GROUP BY id_depth
            ORDER BY id_depth
        """, conn)
        for _, r in df_depth.iterrows():
            print(f"    Depth {r['id_depth']}: {r['rows']:>12,} rows")

        # 1d. Trade types
        print("\n[*] By trade type:")
        df_tt = pd.read_sql("""
            SELECT tradetype, COUNT(*) as rows
            FROM db_ems.vdt_isot_knihaobjednavok_best
            GROUP BY tradetype
        """, conn)
        for _, r in df_tt.iterrows():
            label = "Bid (buy)" if r['tradetype'] == 'N' else "Ask (sell)"
            print(f"    {r['tradetype']} ({label}): {r['rows']:>12,} rows")

        # 1e. Updates per day distribution
        print("\n[*] Updates per day (for 60min products, depth=1):")
        df_daily = pd.read_sql("""
            SELECT
                tradeday::date as day,
                COUNT(*) as updates
            FROM db_ems.vdt_isot_knihaobjednavok_best
            WHERE deliverydur = 60 AND id_depth = 1
            GROUP BY tradeday::date
            ORDER BY tradeday::date
        """, conn)
        print(f"    Mean: {df_daily['updates'].mean():,.0f} updates/day")
        print(f"    Median: {df_daily['updates'].median():,.0f}")
        print(f"    Min: {df_daily['updates'].min():,.0f}")
        print(f"    Max: {df_daily['updates'].max():,.0f}")

        # 1f. Last trades table (may not exist in this backup)
        print("\n[*] Last trades (lasttrades):")
        try:
            df_lt = pd.read_sql("""
                SELECT
                    COUNT(*) as total_rows,
                    MIN(tradeday::date) as min_date,
                    MAX(tradeday::date) as max_date,
                    COUNT(DISTINCT tradeday::date) as n_days
                FROM db_ems.vdt_isot_lasttrades
            """, conn)
            print(f"    Total rows: {df_lt['total_rows'].iloc[0]:,.0f}")
            print(f"    Date range: {df_lt['min_date'].iloc[0]} to {df_lt['max_date'].iloc[0]}")
            print(f"    Unique days: {df_lt['n_days'].iloc[0]}")
        except Exception as e:
            print(f"    [-] Table not available in this backup")
            conn.rollback()

        # 1g. Monthly coverage heatmap data
        print("\n[*] Monthly row counts (order book, 60min, depth=1):")
        df_monthly = pd.read_sql("""
            SELECT
                DATE_TRUNC('month', tradeday)::date as month,
                COUNT(*) as rows,
                COUNT(DISTINCT tradeday::date) as days
            FROM db_ems.vdt_isot_knihaobjednavok_best
            WHERE deliverydur = 60 AND id_depth = 1
            GROUP BY DATE_TRUNC('month', tradeday)
            ORDER BY month
        """, conn)
        for _, r in df_monthly.iterrows():
            print(f"    {r['month']}: {r['rows']:>10,} rows ({r['days']} days)")

    # Save daily coverage
    df_daily.to_csv(DATA_DIR / "daily_coverage.csv", index=False)
    df_monthly.to_csv(DATA_DIR / "monthly_coverage.csv", index=False)

    # Plot: daily update counts
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(pd.to_datetime(df_daily['day']), df_daily['updates'], width=1, alpha=0.7, color='steelblue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Order book updates per day")
    ax.set_title("IDM Order Book: Daily Update Count (60min products, best depth)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_daily_coverage.png", dpi=150)
    plt.close()
    print(f"\n[+] Saved: {PLOT_DIR / '01_daily_coverage.png'}")

    return df_monthly


# ============================================================
# PHASE 2: Price Formation & Convergence
# ============================================================

def phase2_price_formation():
    """Analyze how IDM prices evolve relative to delivery time and DA price."""
    print("\n" + "=" * 70)
    print("PHASE 2: IDM Price Formation & DA Relationship")
    print("=" * 70)

    # Load DA prices for comparison
    da_path = BASE_DIR.parent.parent / "data" / "processed" / "hourly_market_prices.csv"
    if not da_path.exists():
        # Try alternate path
        da_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "hourly_market_prices.csv"

    da_df = None
    if da_path.exists():
        da_df = pd.read_csv(da_path)
        # Handle both 'datetime' and 'timestamp_hour' column names
        time_col = 'datetime' if 'datetime' in da_df.columns else 'timestamp_hour'
        da_df[time_col] = pd.to_datetime(da_df[time_col])
        da_df['tradeday'] = da_df[time_col].dt.date
        if 'hour' not in da_df.columns:
            da_df['hour'] = da_df[time_col].dt.hour
        print(f"[+] Loaded DA prices: {len(da_df)} rows, {da_df['tradeday'].nunique()} days")
    else:
        print(f"[-] DA prices not found at {da_path}, will skip DA comparison")

    with get_conn() as conn:
        # 2a. Price convergence: sample hourly snapshots at time-to-delivery windows
        # For each delivery hour, get the mid price at various lead times before delivery
        print("\n[*] Computing price convergence curves...")
        print("    (aggregating order book snapshots by time-to-delivery)")

        # We'll bucket by hours-before-delivery: 24h, 12h, 6h, 3h, 1h, 0.5h
        # delivery_start = tradeday + periodfrom * 15min (for hourly: periodfrom is the start QH)
        # time_to_delivery = delivery_start - lastupdate (in hours)

        df_conv = pd.read_sql("""
            WITH ob AS (
                SELECT
                    tradeday::date as tradeday,
                    periodfrom as hour,
                    tradetype,
                    price,
                    lastupdate,
                    -- For 60min products, periodfrom IS the hour (0-23)
                    tradeday::date + periodfrom * INTERVAL '1 hour' as delivery_start
                FROM db_ems.vdt_isot_knihaobjednavok_best
                WHERE deliverydur = 60
                AND id_depth = 1
            ),
            with_ttd AS (
                SELECT *,
                    EXTRACT(EPOCH FROM (delivery_start - lastupdate)) / 3600.0 as hours_to_delivery
                FROM ob
                WHERE lastupdate < delivery_start  -- only pre-delivery snapshots
            ),
            bucketed AS (
                SELECT *,
                    CASE
                        WHEN hours_to_delivery >= 20 THEN 24
                        WHEN hours_to_delivery >= 9 THEN 12
                        WHEN hours_to_delivery >= 4.5 THEN 6
                        WHEN hours_to_delivery >= 2 THEN 3
                        WHEN hours_to_delivery >= 0.75 THEN 1
                        ELSE 0.5
                    END as ttd_bucket
                FROM with_ttd
            )
            SELECT
                tradeday,
                hour,
                ttd_bucket,
                AVG(CASE WHEN tradetype = 'N' THEN price END) as avg_bid,
                AVG(CASE WHEN tradetype = 'P' THEN price END) as avg_ask,
                COUNT(*) as n_snapshots
            FROM bucketed
            GROUP BY tradeday, hour, ttd_bucket
            ORDER BY tradeday, hour, ttd_bucket DESC
        """, conn)

        print(f"[+] Price convergence data: {len(df_conv)} rows")

        # Compute mid price
        df_conv['mid_price'] = (df_conv['avg_bid'].fillna(0) + df_conv['avg_ask'].fillna(0)) / 2
        df_conv['mid_price'] = df_conv.apply(
            lambda r: r['avg_ask'] if pd.isna(r['avg_bid']) else (r['avg_bid'] if pd.isna(r['avg_ask']) else r['mid_price']),
            axis=1
        )
        df_conv['spread'] = df_conv['avg_ask'] - df_conv['avg_bid']

        # Save raw convergence data
        df_conv.to_csv(DATA_DIR / "price_convergence_by_ttd.csv", index=False)

        # 2b. Aggregate convergence stats
        print("\n[*] Average price & spread by time-to-delivery bucket:")
        agg = df_conv.groupby('ttd_bucket').agg(
            avg_mid=('mid_price', 'mean'),
            std_mid=('mid_price', 'std'),
            avg_spread=('spread', 'mean'),
            median_spread=('spread', 'median'),
            avg_snapshots=('n_snapshots', 'mean'),
            n_observations=('mid_price', 'count'),
        ).sort_index(ascending=False)
        print(agg.to_string())
        agg.to_csv(DATA_DIR / "convergence_stats.csv")

        # 2c. If we have DA prices, compute the DA-IDM deviation by TTD
        if da_df is not None:
            print("\n[*] Computing DA-IDM deviation by time-to-delivery...")
            df_conv['tradeday'] = pd.to_datetime(df_conv['tradeday']).dt.date
            df_conv['hour'] = df_conv['hour'].astype(int)

            merged = df_conv.merge(
                da_df[['tradeday', 'hour', 'da_price']].drop_duplicates(),
                on=['tradeday', 'hour'],
                how='inner'
            )
            merged['idm_da_diff'] = merged['mid_price'] - merged['da_price']
            merged['idm_da_pct'] = merged['idm_da_diff'] / merged['da_price'] * 100

            print(f"[+] Merged with DA: {len(merged)} rows")

            # Deviation stats by TTD bucket
            dev_stats = merged.groupby('ttd_bucket').agg(
                mean_diff=('idm_da_diff', 'mean'),
                median_diff=('idm_da_diff', 'median'),
                std_diff=('idm_da_diff', 'std'),
                mean_pct=('idm_da_pct', 'mean'),
                n=('idm_da_diff', 'count'),
            ).sort_index(ascending=False)
            print("\n    DA-IDM deviation by hours-to-delivery:")
            print(dev_stats.to_string())
            dev_stats.to_csv(DATA_DIR / "da_idm_deviation_by_ttd.csv")

            merged.to_csv(DATA_DIR / "convergence_with_da.csv", index=False)
        else:
            merged = None

        # 2d. Bid-ask spread evolution by time of day and TTD
        print("\n[*] Bid-ask spread by hour of day and TTD...")
        spread_by_hour = df_conv.groupby(['hour', 'ttd_bucket']).agg(
            avg_spread=('spread', 'mean'),
            median_spread=('spread', 'median'),
        ).reset_index()
        spread_by_hour.to_csv(DATA_DIR / "spread_by_hour_ttd.csv", index=False)

    # ---- PLOTS ----

    # Plot 2a: Price convergence curve (average mid price by TTD)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Average price level by TTD bucket
    ax = axes[0, 0]
    ttd_labels = sorted(agg.index, reverse=True)
    ax.plot(ttd_labels, [agg.loc[t, 'avg_mid'] for t in ttd_labels], 'bo-', linewidth=2, markersize=8)
    ax.fill_between(ttd_labels,
                    [agg.loc[t, 'avg_mid'] - agg.loc[t, 'std_mid'] for t in ttd_labels],
                    [agg.loc[t, 'avg_mid'] + agg.loc[t, 'std_mid'] for t in ttd_labels],
                    alpha=0.2, color='blue')
    ax.set_xlabel("Hours to delivery")
    ax.set_ylabel("Average mid price [EUR/MWh]")
    ax.set_title("IDM Price Level by Time-to-Delivery")
    ax.invert_xaxis()

    # Top-right: Bid-ask spread by TTD
    ax = axes[0, 1]
    ax.plot(ttd_labels, [agg.loc[t, 'avg_spread'] for t in ttd_labels], 'ro-', linewidth=2, markersize=8)
    ax.plot(ttd_labels, [agg.loc[t, 'median_spread'] for t in ttd_labels], 'r--', linewidth=1, alpha=0.7, label='median')
    ax.set_xlabel("Hours to delivery")
    ax.set_ylabel("Bid-ask spread [EUR/MWh]")
    ax.set_title("Bid-Ask Spread by Time-to-Delivery")
    ax.legend()
    ax.invert_xaxis()

    # Bottom-left: DA-IDM deviation by TTD
    ax = axes[1, 0]
    if merged is not None:
        ax.bar(range(len(dev_stats)), dev_stats['mean_diff'], alpha=0.7, color='teal')
        ax.errorbar(range(len(dev_stats)), dev_stats['mean_diff'],
                   yerr=dev_stats['std_diff'], fmt='none', color='black', capsize=3)
        ax.set_xticks(range(len(dev_stats)))
        ax.set_xticklabels([f"{t}h" for t in dev_stats.index], rotation=0)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Hours to delivery")
        ax.set_ylabel("IDM - DA price [EUR/MWh]")
        ax.set_title("IDM Deviation from DA by Time-to-Delivery")
    else:
        ax.text(0.5, 0.5, "DA prices not available", transform=ax.transAxes, ha='center')

    # Bottom-right: Spread by hour of day (for TTD=1h)
    ax = axes[1, 1]
    for ttd_val, color, label in [(0.5, 'red', '30min'), (1, 'orange', '1h'), (3, 'green', '3h'), (12, 'blue', '12h')]:
        subset = spread_by_hour[spread_by_hour['ttd_bucket'] == ttd_val]
        if len(subset) > 0:
            ax.plot(subset['hour'], subset['avg_spread'], 'o-', color=color, label=f'TTD={label}', markersize=4)
    ax.set_xlabel("Delivery hour")
    ax.set_ylabel("Avg bid-ask spread [EUR/MWh]")
    ax.set_title("Bid-Ask Spread by Hour of Day")
    ax.legend()
    ax.set_xticks(range(0, 24, 2))

    fig.suptitle("IDM Order Book: Price Formation Analysis", fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_price_formation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Saved: {PLOT_DIR / '02_price_formation.png'}")

    # Plot 2b: DA-IDM deviation distribution
    if merged is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Distribution of IDM-DA diff at 1h TTD
        for i, (ttd_val, label) in enumerate([(12, '12h before'), (3, '3h before'), (0.5, '30min before')]):
            ax = axes[i]
            subset = merged[merged['ttd_bucket'] == ttd_val]['idm_da_diff'].dropna()
            if len(subset) > 0:
                ax.hist(subset, bins=80, alpha=0.7, color='steelblue', edgecolor='white')
                ax.axvline(subset.mean(), color='red', linestyle='--', label=f'mean={subset.mean():.1f}')
                ax.axvline(subset.median(), color='orange', linestyle='--', label=f'median={subset.median():.1f}')
                ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel("IDM - DA [EUR/MWh]")
                ax.set_ylabel("Count")
                ax.set_title(f"IDM-DA Deviation at {label}")
                ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(PLOT_DIR / "03_da_idm_deviation_dist.png", dpi=150)
        plt.close()
        print(f"[+] Saved: {PLOT_DIR / '03_da_idm_deviation_dist.png'}")

        # Plot 2c: Hourly heatmap of DA-IDM deviation
        fig, ax = plt.subplots(figsize=(14, 6))
        hourly_ttd = merged.groupby(['hour', 'ttd_bucket'])['idm_da_diff'].mean().unstack('ttd_bucket')
        hourly_ttd = hourly_ttd[sorted(hourly_ttd.columns, reverse=True)]
        im = ax.imshow(hourly_ttd.T, aspect='auto', cmap='RdBu_r', vmin=-20, vmax=20)
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        ax.set_yticks(range(len(hourly_ttd.columns)))
        ax.set_yticklabels([f"{t}h" for t in hourly_ttd.columns])
        ax.set_xlabel("Delivery Hour")
        ax.set_ylabel("Hours to Delivery")
        ax.set_title("IDM - DA Price Deviation [EUR/MWh] by Hour and Time-to-Delivery")
        plt.colorbar(im, ax=ax, label="EUR/MWh")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "04_da_idm_heatmap.png", dpi=150)
        plt.close()
        print(f"[+] Saved: {PLOT_DIR / '04_da_idm_heatmap.png'}")

    return df_conv, merged


def phase2b_update_frequency():
    """Analyze order book update frequency / activity patterns."""
    print("\n" + "-" * 70)
    print("PHASE 2b: Order Book Activity Patterns")
    print("-" * 70)

    with get_conn() as conn:
        # Snapshot count by hour-of-day (when updates happen, not delivery hour)
        print("\n[*] Update activity by hour of day...")
        df_activity = pd.read_sql("""
            SELECT
                EXTRACT(HOUR FROM lastupdate)::int as update_hour,
                COUNT(*) as n_updates,
                COUNT(DISTINCT tradeday::date) as n_days
            FROM db_ems.vdt_isot_knihaobjednavok_best
            WHERE deliverydur = 60 AND id_depth = 1
            GROUP BY EXTRACT(HOUR FROM lastupdate)::int
            ORDER BY update_hour
        """, conn)
        df_activity['avg_per_day'] = df_activity['n_updates'] / df_activity['n_days']
        print(df_activity.to_string(index=False))
        df_activity.to_csv(DATA_DIR / "update_activity_by_hour.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df_activity['update_hour'], df_activity['avg_per_day'], color='steelblue', alpha=0.8)
    ax.set_xlabel("Hour of day (when update was recorded)")
    ax.set_ylabel("Avg order book updates per day")
    ax.set_title("IDM Order Book: Update Activity by Time of Day")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_update_activity.png", dpi=150)
    plt.close()
    print(f"\n[+] Saved: {PLOT_DIR / '05_update_activity.png'}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("[*] IDM Order Book Analysis - Phase 1 & 2")
    print("[*] Querying local PostgreSQL (localhost:5432/DB_EMS)")
    print()

    # Phase 1: What do we have?
    monthly = phase1_data_overview()

    # Phase 2: Price formation
    conv_data, merged_data = phase2_price_formation()

    # Phase 2b: Activity patterns
    phase2b_update_frequency()

    print("\n" + "=" * 70)
    print("[+] Phase 1+2 complete. Check plots/ and data/ for outputs.")
    print("=" * 70)
