"""
Extract Hub-to-Hub cross-border capacity and IDM price data.

Sources:
  - H2H capacity: Production DB (10.4.1.66) db_ems.isot_vdt_hub2hub
  - IDM best bid/ask: Local DB (localhost) db_ems.vdt_isot_knihaobjednavok_best
  - IDM last traded: Production DB (10.4.1.66) db_ems.vdt_isot_lasttrades

Output: data/h2h_idm_hourly.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "h2h_idm_hourly.csv"

# --- DB connections ---
PROD_DB = {
    "host": "10.4.1.66",
    "port": 5432,
    "dbname": "DB_EMS",
    "user": "pnacek",
    "password": "Kapitan4478",
}
LOCAL_DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "DB_EMS",
    "user": "postgres",
    "password": "Kapitan1",
}

# Area ID to short name mapping
AREA_MAP = {
    6: "ceps",    # CZ - direct neighbor
    8: "apg",     # AT - direct neighbor
    36: "mavir",  # HU - direct neighbor
    44: "pse",    # PL - direct neighbor
    33: "de_50hz", # DE (50HzT) - indirect
    30: "de_ttg",  # DE (TTG) - indirect
}


def extract_h2h(conn_params):
    """Extract last pre-delivery H2H capacity snapshot per (area, tradeday, hour)."""
    print("[*] Extracting H2H capacity data from production DB...")

    query = """
    WITH ranked AS (
        SELECT
            id_area,
            tradeday::date as tradeday,
            EXTRACT(HOUR FROM periodfrom)::int as hour,
            available_in,
            available_out,
            date_in,
            ROW_NUMBER() OVER (
                PARTITION BY id_area, tradeday, periodfrom
                ORDER BY date_in DESC
            ) as rn
        FROM db_ems.isot_vdt_hub2hub
        WHERE date_in < periodfrom
        AND id_area IN (6, 8, 30, 33, 36, 44)
    )
    SELECT id_area, tradeday, hour, available_in, available_out
    FROM ranked
    WHERE rn = 1
    ORDER BY tradeday, hour, id_area
    """

    with psycopg2.connect(**conn_params) as conn:
        df = pd.read_sql(query, conn)

    print(f"[+] H2H raw: {len(df)} rows, {df['tradeday'].nunique()} days")
    return df


def pivot_h2h(df):
    """Pivot H2H long format to wide: one column per area per direction."""
    print("[*] Pivoting H2H data to wide format...")

    # Map area IDs to names
    df["area"] = df["id_area"].map(AREA_MAP)

    # Pivot to wide
    rows = []
    for (tradeday, hour), grp in df.groupby(["tradeday", "hour"]):
        row = {"tradeday": tradeday, "hour": hour}
        for _, r in grp.iterrows():
            area = r["area"]
            row[f"{area}_in"] = r["available_in"]
            row[f"{area}_out"] = r["available_out"]
        rows.append(row)

    wide = pd.DataFrame(rows)

    # Merge DE areas (50HzT + TTG) into single DE feature
    for direction in ["in", "out"]:
        col_50hz = f"de_50hz_{direction}"
        col_ttg = f"de_ttg_{direction}"
        col_de = f"de_{direction}"
        if col_50hz in wide.columns and col_ttg in wide.columns:
            wide[col_de] = wide[[col_50hz, col_ttg]].mean(axis=1)
            wide.drop(columns=[col_50hz, col_ttg], inplace=True)

    # Derived features
    in_cols = [c for c in wide.columns if c.endswith("_in")]
    out_cols = [c for c in wide.columns if c.endswith("_out")]

    wide["total_import_cap"] = wide[in_cols].sum(axis=1)
    wide["total_export_cap"] = wide[out_cols].sum(axis=1)
    wide["net_capacity"] = wide["total_import_cap"] - wide["total_export_cap"]

    # Congestion flags (import direction = 0 means congested for imports into SK)
    for col in in_cols:
        area = col.replace("_in", "")
        wide[f"{area}_congested_in"] = (wide[col] == 0).astype(int)

    wide["congested_in_count"] = wide[[c for c in wide.columns if c.endswith("_congested_in")]].sum(axis=1)

    print(f"[+] H2H wide: {len(wide)} rows, {len(wide.columns)} columns")
    return wide


def extract_idm_orderbook(conn_params, date_range):
    """Extract last pre-delivery bid/ask from local order book, aggregate to hourly."""
    print("[*] Extracting IDM best bid/ask from local DB (day by day)...")

    all_days = []
    dates = pd.date_range(date_range[0], date_range[1], freq="D")

    for i, day in enumerate(dates):
        day_str = day.strftime("%Y-%m-%d")
        query = f"""
        WITH ranked AS (
            SELECT
                tradeday::date as tradeday,
                periodfrom,
                tradetype,
                price,
                lastupdate,
                ROW_NUMBER() OVER (
                    PARTITION BY periodfrom, tradetype
                    ORDER BY lastupdate DESC
                ) as rn
            FROM db_ems.vdt_isot_knihaobjednavok_best
            WHERE tradeday::date = '{day_str}'
            AND id_depth = 1
            AND deliverydur = 15
            AND lastupdate < tradeday + periodfrom * INTERVAL '15 minutes'
        )
        SELECT tradeday, periodfrom, tradetype, price
        FROM ranked
        WHERE rn = 1
        """
        try:
            with psycopg2.connect(**conn_params) as conn:
                df_day = pd.read_sql(query, conn)
            if len(df_day) > 0:
                all_days.append(df_day)
        except Exception as e:
            print(f"[-] Warning: failed for {day_str}: {e}")

        if (i + 1) % 10 == 0:
            print(f"    ... processed {i + 1}/{len(dates)} days")

    if not all_days:
        print("[!] No IDM order book data extracted")
        return pd.DataFrame()

    df = pd.concat(all_days, ignore_index=True)
    print(f"[+] IDM order book raw: {len(df)} rows")

    # Pivot bid/ask
    bid = df[df["tradetype"] == "N"][["tradeday", "periodfrom", "price"]].rename(columns={"price": "bid"})
    ask = df[df["tradetype"] == "P"][["tradeday", "periodfrom", "price"]].rename(columns={"price": "ask"})
    merged = bid.merge(ask, on=["tradeday", "periodfrom"], how="outer")

    # Compute mid price and spread
    merged["mid_price"] = (merged["bid"].fillna(0) + merged["ask"].fillna(0)) / 2
    merged["mid_price"] = merged.apply(
        lambda r: r["ask"] if pd.isna(r["bid"]) else (r["bid"] if pd.isna(r["ask"]) else r["mid_price"]),
        axis=1,
    )
    merged["spread"] = merged["ask"] - merged["bid"]

    # Map periodfrom (0-95) to hour (0-23)
    merged["hour"] = merged["periodfrom"] // 4

    # Aggregate 4 quarter-hours to hourly
    hourly = merged.groupby(["tradeday", "hour"]).agg(
        idm_mid_price=("mid_price", "mean"),
        idm_bid=("bid", "mean"),
        idm_ask=("ask", "mean"),
        idm_spread=("spread", "mean"),
    ).reset_index()

    print(f"[+] IDM hourly from order book: {len(hourly)} rows")
    return hourly


def extract_idm_lasttrades(conn_params):
    """Extract last traded price per (tradeday, hour) from production DB."""
    print("[*] Extracting IDM last traded prices from production DB...")

    query = """
    WITH ranked AS (
        SELECT
            tradeday::date as tradeday,
            periodfrom as hour,
            price as last_price,
            total_amount,
            lastupdate,
            ROW_NUMBER() OVER (
                PARTITION BY tradeday, periodfrom
                ORDER BY lastupdate DESC
            ) as rn
        FROM db_ems.vdt_isot_lasttrades
        WHERE deliverydur = 60
    )
    SELECT tradeday, hour, last_price, total_amount
    FROM ranked
    WHERE rn = 1
    ORDER BY tradeday, hour
    """

    with psycopg2.connect(**conn_params) as conn:
        df = pd.read_sql(query, conn)

    print(f"[+] IDM last trades: {len(df)} rows, {df['tradeday'].nunique()} days")
    return df


def main():
    print("=" * 60)
    print("H2H Capacity + IDM Price Data Extraction")
    print("=" * 60)

    # --- Part A: H2H from production DB ---
    h2h_raw = extract_h2h(PROD_DB)
    h2h_wide = pivot_h2h(h2h_raw)

    # --- Part B1: IDM last traded from production DB ---
    idm_trades = extract_idm_lasttrades(PROD_DB)

    # --- Part B2: IDM order book from local DB ---
    # Determine overlap date range
    h2h_min = h2h_wide["tradeday"].min()
    h2h_max = h2h_wide["tradeday"].max()
    print(f"[*] H2H date range: {h2h_min} to {h2h_max}")

    idm_ob = extract_idm_orderbook(LOCAL_DB, (h2h_min, h2h_max))

    # --- Part C: Merge ---
    print("[*] Merging datasets...")

    # Ensure types match
    h2h_wide["tradeday"] = pd.to_datetime(h2h_wide["tradeday"]).dt.date
    idm_trades["tradeday"] = pd.to_datetime(idm_trades["tradeday"]).dt.date
    h2h_wide["hour"] = h2h_wide["hour"].astype(int)
    idm_trades["hour"] = idm_trades["hour"].astype(int)

    # Start with h2h, merge last traded prices
    master = h2h_wide.merge(idm_trades, on=["tradeday", "hour"], how="left")

    # Merge order book prices if available
    if len(idm_ob) > 0:
        idm_ob["tradeday"] = pd.to_datetime(idm_ob["tradeday"]).dt.date
        idm_ob["hour"] = idm_ob["hour"].astype(int)
        master = master.merge(idm_ob, on=["tradeday", "hour"], how="left")

    # Add time features
    master["tradeday"] = pd.to_datetime(master["tradeday"])
    master["dayofweek"] = master["tradeday"].dt.dayofweek
    master["is_weekend"] = (master["dayofweek"] >= 5).astype(int)
    master["month"] = master["tradeday"].dt.month

    # Sort
    master.sort_values(["tradeday", "hour"], inplace=True)
    master.reset_index(drop=True, inplace=True)

    # Use best available price column
    if "idm_mid_price" in master.columns:
        master["idm_price"] = master["idm_mid_price"].fillna(master.get("last_price", np.nan))
    elif "last_price" in master.columns:
        master["idm_price"] = master["last_price"]

    # Save
    master.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[+] Saved: {OUTPUT_PATH}")
    print(f"    Rows: {len(master)}")
    print(f"    Days: {master['tradeday'].nunique()}")
    print(f"    Columns: {list(master.columns)}")

    # Quick stats
    price_col = "idm_price" if "idm_price" in master.columns else "last_price"
    if price_col in master.columns:
        valid = master[price_col].notna()
        print(f"\n[*] IDM price stats ({valid.sum()} valid obs):")
        print(f"    Mean: {master.loc[valid, price_col].mean():.1f} EUR/MWh")
        print(f"    Std:  {master.loc[valid, price_col].std():.1f} EUR/MWh")
        print(f"    Min:  {master.loc[valid, price_col].min():.1f}")
        print(f"    Max:  {master.loc[valid, price_col].max():.1f}")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
