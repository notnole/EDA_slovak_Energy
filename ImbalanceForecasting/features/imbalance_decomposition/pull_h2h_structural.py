"""
Structural Imbalance Decomposition (Approach B)

Pull hub-to-hub cross-border flow data from ISOT database to test:
  System Imbalance = Load Surprise - Net Cross-Border IDM Correction

Key insight: Domestic IDM trades are zero-sum within Slovakia.
Only cross-border IDM trades change the total system position.
The H2H available capacity reduction = actual cross-border IDM flow.

READ-ONLY queries on production database.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT = Path(__file__).resolve().parent
DATA_OUT = OUT / "data"
DATA_OUT.mkdir(exist_ok=True)

DB_CONFIG = {
    "host": "10.4.1.66",
    "port": 5432,
    "dbname": "DB_EMS",
    "user": "pnacek",
    "password": "Kapitan4478",
    "options": "-c search_path=db_ems,public",
}

# Area IDs (SK perspective: import from / export to each neighbor)
AREAS = {
    6: ("CEPS", "CZ"),
    8: ("APG", "AT"),
    30: ("TTG", "DE"),
    33: ("50HzT", "DE"),
    36: ("MAVIR", "HU"),
    44: ("PSE", "PL"),
}


def query_db(sql, params=None):
    """Execute read-only query and return DataFrame."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql_query(sql, conn, params=params)


# =========================================================================
# 1. Explore H2H snapshot pattern - understand the data
# =========================================================================
print("[*] Exploring H2H snapshot pattern (CEPS, 2026-02-10)...")

snapshot_check = query_db("""
    SELECT id_area, tradeday, periodfrom, periodto,
           available_in, available_out, date_in
    FROM isot_vdt_hub2hub
    WHERE tradeday = '2026-02-10'
      AND id_area = 6
    ORDER BY periodfrom, date_in
    LIMIT 200
""")

print(f"  Rows for CEPS on 2026-02-10: {len(snapshot_check)}")
n_periods = snapshot_check['periodfrom'].nunique()
snapshots_per = snapshot_check.groupby('periodfrom')['date_in'].nunique()
print(f"  Unique periods: {n_periods}")
print(f"  Snapshots per period: min={snapshots_per.min()}, max={snapshots_per.max()}, "
      f"mean={snapshots_per.mean():.1f}")

# Show capacity reduction for one period
sample_period = snapshot_check['periodfrom'].unique()[0]
sample = snapshot_check[snapshot_check['periodfrom'] == sample_period].sort_values('date_in')
print(f"\n  Sample period {sample_period}:")
print(f"    First snapshot: in={sample.iloc[0]['available_in']}, out={sample.iloc[0]['available_out']} at {sample.iloc[0]['date_in']}")
print(f"    Last snapshot:  in={sample.iloc[-1]['available_in']}, out={sample.iloc[-1]['available_out']} at {sample.iloc[-1]['date_in']}")
cap_used_in = float(sample.iloc[0]['available_in']) - float(sample.iloc[-1]['available_in'])
cap_used_out = float(sample.iloc[0]['available_out']) - float(sample.iloc[-1]['available_out'])
print(f"    Capacity USED: import={cap_used_in:.1f} MW, export={cap_used_out:.1f} MW")
print(f"    -> Net cross-border import via CEPS = {cap_used_in - cap_used_out:.1f} MW")

# =========================================================================
# 2. Pull H2H first/last snapshots for ALL periods and borders
#    This gives us: initial ATC, final available, and capacity used
# =========================================================================
print("\n[*] Pulling H2H capacity reduction (first vs last snapshot per period)...")
print("    This is the KEY data: capacity used = cross-border IDM flow")

h2h_reduction = query_db("""
    WITH snapshots AS (
        SELECT id_area, tradeday, periodfrom, periodto,
               available_in, available_out, date_in,
               ROW_NUMBER() OVER (
                   PARTITION BY id_area, tradeday, periodfrom
                   ORDER BY date_in ASC
               ) as rn_first,
               ROW_NUMBER() OVER (
                   PARTITION BY id_area, tradeday, periodfrom
                   ORDER BY date_in DESC
               ) as rn_last,
               COUNT(*) OVER (
                   PARTITION BY id_area, tradeday, periodfrom
               ) as n_snapshots
        FROM isot_vdt_hub2hub
        WHERE tradeday >= '2025-01-01'
    ),
    first_snap AS (
        SELECT id_area, tradeday, periodfrom, periodto, n_snapshots,
               available_in as initial_in, available_out as initial_out,
               date_in as first_date
        FROM snapshots WHERE rn_first = 1
    ),
    last_snap AS (
        SELECT id_area, tradeday, periodfrom,
               available_in as final_in, available_out as final_out,
               date_in as last_date
        FROM snapshots WHERE rn_last = 1
    )
    SELECT f.id_area, f.tradeday, f.periodfrom, f.periodto,
           f.initial_in, f.initial_out, l.final_in, l.final_out,
           (f.initial_in - l.final_in) as cap_used_in,
           (f.initial_out - l.final_out) as cap_used_out,
           f.n_snapshots, f.first_date, l.last_date
    FROM first_snap f
    JOIN last_snap l ON f.id_area = l.id_area
                    AND f.tradeday = l.tradeday
                    AND f.periodfrom = l.periodfrom
    ORDER BY f.tradeday, f.periodfrom, f.id_area
""")

# Map area names
h2h_reduction['area_name'] = h2h_reduction['id_area'].map({k: v[0] for k, v in AREAS.items()})
h2h_reduction['country'] = h2h_reduction['id_area'].map({k: v[1] for k, v in AREAS.items()})

print(f"  [+] H2H reduction: {len(h2h_reduction)} rows")
print(f"  Date range: {h2h_reduction['tradeday'].min()} to {h2h_reduction['tradeday'].max()}")
print(f"  Areas: {h2h_reduction['area_name'].unique()}")
print(f"  Avg snapshots: {h2h_reduction['n_snapshots'].mean():.1f}")

# Summary stats per border
print("\n  Capacity utilization by border (MW, mean):")
for area_id, (name, cc) in AREAS.items():
    sub = h2h_reduction[h2h_reduction['id_area'] == area_id]
    if len(sub) > 0:
        print(f"    {name:6s} ({cc}): import_used={sub['cap_used_in'].mean():+.1f}, "
              f"export_used={sub['cap_used_out'].mean():+.1f}, "
              f"net_import={sub['cap_used_in'].mean() - sub['cap_used_out'].mean():+.1f} MW "
              f"(n={len(sub)})")

h2h_reduction.to_csv(DATA_OUT / "h2h_capacity_reduction.csv", index=False)
print(f"  [+] Saved {DATA_OUT / 'h2h_capacity_reduction.csv'}")

# =========================================================================
# 3. Pivot to wide format and compute total cross-border flow into SK
# =========================================================================
print("\n[*] Computing total cross-border IDM flow into SK...")

# Pivot: one row per (tradeday, periodfrom) with per-border capacity used
pivot = h2h_reduction.pivot_table(
    index=['tradeday', 'periodfrom'],
    columns='area_name',
    values=['cap_used_in', 'cap_used_out', 'initial_in', 'initial_out', 'final_in', 'final_out'],
    aggfunc='first'
)
pivot.columns = [f"{area}_{metric}" for metric, area in pivot.columns]
pivot = pivot.reset_index()

# Total cross-border import used (across all borders)
in_used_cols = [c for c in pivot.columns if c.endswith('_cap_used_in')]
out_used_cols = [c for c in pivot.columns if c.endswith('_cap_used_out')]

pivot['total_xborder_import_used'] = pivot[in_used_cols].sum(axis=1)
pivot['total_xborder_export_used'] = pivot[out_used_cols].sum(axis=1)
# Net cross-border flow INTO Slovakia (positive = net import)
pivot['net_xborder_import'] = pivot['total_xborder_import_used'] - pivot['total_xborder_export_used']

# Build proper timestamp
pivot['timestamp'] = pd.to_datetime(pivot['tradeday']) + \
    pd.to_timedelta((pivot['periodfrom'] - pd.to_datetime(pivot['tradeday'])).dt.total_seconds(), unit='s')
# Actually periodfrom IS a timestamp, so just use it directly
pivot['timestamp'] = pd.to_datetime(pivot['periodfrom'])

print(f"  [+] Wide format: {len(pivot)} rows")
print(f"  Net cross-border import: mean={pivot['net_xborder_import'].mean():.1f}, "
      f"std={pivot['net_xborder_import'].std():.1f} MW")

pivot.to_csv(DATA_OUT / "h2h_xborder_flow.csv", index=False)
print(f"  [+] Saved {DATA_OUT / 'h2h_xborder_flow.csv'}")

# =========================================================================
# 4. Pull IDM last traded data
# =========================================================================
print("\n[*] Pulling IDM last traded data...")

idm_volume = query_db("""
    WITH ranked AS (
        SELECT tradeday, periodfrom, periodto, deliverydur,
               price, amount, total_amount, lastupdate, date_in,
               ROW_NUMBER() OVER (
                   PARTITION BY tradeday, periodfrom, deliverydur
                   ORDER BY lastupdate DESC
               ) as rn
        FROM vdt_isot_lasttrades
        WHERE tradeday >= '2025-01-01'
    )
    SELECT tradeday, periodfrom, periodto, deliverydur,
           price as last_price, amount as last_amount,
           total_amount, lastupdate
    FROM ranked
    WHERE rn = 1
    ORDER BY tradeday, periodfrom, deliverydur
""")

print(f"  [+] IDM volume: {len(idm_volume)} rows")
print(f"  Date range: {idm_volume['tradeday'].min()} to {idm_volume['tradeday'].max()}")
print(f"  Delivery durations: {idm_volume['deliverydur'].value_counts().to_dict()}")

idm_volume.to_csv(DATA_OUT / "idm_lasttrades.csv", index=False)
print(f"  [+] Saved {DATA_OUT / 'idm_lasttrades.csv'}")

# =========================================================================
# 5. Pull executed trades (Buy/Sell direction)
# =========================================================================
print("\n[*] Pulling executed trades by direction...")

trades = query_db("""
    SELECT tradeday, periodfrom, periodto, deliverydur,
           tradetype, profilerole,
           SUM(amount) as total_amount,
           COUNT(*) as n_trades,
           AVG(price) as avg_price
    FROM isot_vdt_trade
    WHERE tradeday >= '2025-01-01'
      AND ano_platny = 1
    GROUP BY tradeday, periodfrom, periodto, deliverydur, tradetype, profilerole
    ORDER BY tradeday, periodfrom, tradetype
""")

print(f"  [+] Trades: {len(trades)} rows")
if len(trades) > 0:
    print(f"  Trade types: {trades['tradetype'].value_counts().to_dict()}")
    print(f"  Profile roles: {trades['profilerole'].value_counts().to_dict()}")

trades.to_csv(DATA_OUT / "isot_trades_aggregated.csv", index=False)
print(f"  [+] Saved {DATA_OUT / 'isot_trades_aggregated.csv'}")

# =========================================================================
# 6. Check legacy h2h_v2 for actual from->to flow amounts
# =========================================================================
print("\n[*] Checking vdt_isot_h2h_v2 (legacy flow data)...")

h2h_v2_check = query_db("""
    SELECT MIN(tradeday) as min_date, MAX(tradeday) as max_date,
           COUNT(*) as n_rows,
           COUNT(DISTINCT tradeday) as n_days
    FROM vdt_isot_h2h_v2
""")
print(f"  {h2h_v2_check.to_string(index=False)}")

if h2h_v2_check['n_rows'].iloc[0] > 0:
    # Get area definitions for v2
    h2h_v2_areas = query_db("SELECT * FROM vdt_isot_h2h_area ORDER BY id_area")
    print(f"\n  H2H v2 areas:")
    print(h2h_v2_areas.to_string(index=False))

    h2h_v2_roles = query_db("SELECT * FROM vdt_isot_h2h_profilerole ORDER BY id_profilerole")
    print(f"\n  H2H v2 profile roles:")
    print(h2h_v2_roles.to_string(index=False))

    # Pull a sample
    h2h_v2_sample = query_db("""
        SELECT h.tradeday, h.deliverydur, h.id_areafrom, h.id_areato,
               h.id_profilerole, h.periodfrom, h.periodto, h.amount, h.date_in
        FROM vdt_isot_h2h_v2 h
        WHERE h.tradeday = (SELECT MAX(tradeday) FROM vdt_isot_h2h_v2)
        ORDER BY h.periodfrom, h.id_areafrom, h.id_areato
        LIMIT 50
    """)
    print(f"\n  Sample from latest day:")
    print(h2h_v2_sample.head(20).to_string(index=False))
    h2h_v2_sample.to_csv(DATA_OUT / "h2h_v2_sample.csv", index=False)

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 80)
print("DATA PULL COMPLETE")
print("=" * 80)
print(f"\nKey finding: H2H has {h2h_reduction['n_snapshots'].mean():.0f} snapshots per period per border")
print(f"  -> We CAN track capacity reduction = actual cross-border IDM flow")
print(f"  -> Net cross-border import: mean={pivot['net_xborder_import'].mean():.1f} MW, "
      f"std={pivot['net_xborder_import'].std():.1f} MW")
print(f"\nFiles saved to: {DATA_OUT}")
print(f"  h2h_capacity_reduction.csv - Per-border capacity used (first-last snapshot)")
print(f"  h2h_xborder_flow.csv      - Total cross-border flow per period (pivoted)")
print(f"  idm_lasttrades.csv         - IDM cumulative volume")
print(f"  isot_trades_aggregated.csv - Executed trades by type/role")
print(f"\n[+] Next: merge with load surprise + imbalance for structural regression")
