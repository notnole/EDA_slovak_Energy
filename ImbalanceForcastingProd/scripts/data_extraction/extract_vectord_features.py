"""
Extract candidate features from vectord for the spread model experiment.

Pulls:
  - Sk.A.NatGas       (hourly, full range) — gas generation, marginal-fuel signal
  - SK.F.ResL.M.1     (15-min, Sep 2025+) — residual load forecast
  - DE.F.M.1.Spot     (15-min, Sep 2025+) — German DA spot forecast (coupling)

Saves to: ImbalanceForcastingProd/data/features/vectord_features.csv
  Index: 15-min UTC timestamps
  Columns: natgas_mw, resl_mw, de_spot

Requires SSH tunnel:
    ssh -L8080:10.100.0.70:8080 noel@greenbat1.vps.wbsprt.com
"""

import pandas as pd
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from vectord import VectordClient

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2025-01-01T00:00:00Z"
END = "2026-04-15T00:00:00Z"

VECTORS = [
    # (vector_name, column_name, cadence, preserve_nan)
    # Generation mix actuals (ENTSO-E) — apply shift(LEAD+4) in training for publication delay
    ("Sk.A.NatGas", "natgas_mw", "hourly", False),
    ("Sk.A.HardCoal", "hardcoal_mw", "hourly", False),
    ("Sk.A.Lignite", "lignite_mw", "hourly", False),
    ("Sk.A.Biomas", "biomas_mw", "hourly", False),
    ("SK.A.Nuclear", "nuclear_mw", "hourly", False),
    ("Sk.A.FosilOil", "fosiloil_mw", "hourly", False),
    ("Sk.A.HydroReservoir", "hydroreservoir_mw", "hourly", False),
    ("Sk.A.HydroRunRiver", "hydrorunriver_mw", "hourly", False),
    ("Sk.A.HydroPump", "hydropump_mw", "hourly", False),
    # Per-unit pumped-storage generation (Čierny Váh units CVTG2-6)
    ("Sk.A.HydroPump.CVTG2", "hydropump_cvtg2_mw", "hourly", False),
    ("Sk.A.HydroPump.CVTG3", "hydropump_cvtg3_mw", "hourly", False),
    ("Sk.A.HydroPump.CVTG4", "hydropump_cvtg4_mw", "hourly", False),
    ("Sk.A.HydroPump.CVTG5", "hydropump_cvtg5_mw", "hourly", False),
    ("Sk.A.HydroPump.CVTG6", "hydropump_cvtg6_mw", "hourly", False),
    # D-1 known forecasts (no shift — forward-looking OK since published D-1)
    ("SK.F.ResL.M.1", "resl_mw", "15min", False),
    ("SK.F.ResL.M.ISR.1", "resl_isr_mw", "15min", False),
    ("DE.F.M.1.Spot", "de_spot", "15min", False),
    ("SK.F.M.1.Spot", "sk_spot_m1", "15min", False),
    ("SK.F.M.2.Spot", "sk_spot_m2", "15min", False),
    ("SK.F.M.Merged.Spot", "sk_spot_merged", "15min", False),
    # EQ consumption forecasts (multiple weather models — D-1)
    ("SK.F.Cons.M.1", "cons_gfs", "15min", False),
    ("SK.F.Cons.M.Icon.1", "cons_icon", "15min", False),
    ("SK.F.Cons.ECMF.1", "cons_ecmf", "15min", False),
    ("SK.F.Cons.SEPS.1", "cons_seps", "hourly", False),
    ("Sk.F.Solar", "solar_fcst", "hourly", False),
    # EQ temperature forecasts (multi-model — D-1)
    ("SK.T.GFS.1", "temp_gfs", "15min", False),
    ("SK.T.ECM.1", "temp_ecm", "15min", False),
    ("SK.T.Icon.1", "temp_icon", "15min", False),
    # EQ cloud cover forecasts (D-1)
    ("SK.Cloud.EC.Merged", "cloud_ec", "15min", False),
    ("SK.Cloud.Icon.Merged", "cloud_icon", "15min", False),
    # PICASSO marginal prices (15-min, published ~15min later)
    # preserve_nan=True: NaN means only one side of marginal price was activated
    # in that period — the missing-ness itself is informative.
    ("PICASSO.MarginalPricess.SEPS_POS.Avg", "picasso_pos", "15min", True),
    ("PICASSO.MarginalPricess.SEPS_NEG.Avg", "picasso_neg", "15min", True),
]


def pull_chunked(client, vec_name, start, end, chunk_days=30):
    """Pull a vector in chunks to avoid timeouts."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_ts)
        s = cur.strftime("%Y-%m-%dT%H:%M:%SZ")
        e = nxt.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            df = client.read_df(vec_name, s, e)
            if len(df) > 0:
                chunks.append(df)
                print(f"    {s[:10]} -> {e[:10]}: {len(df):>5d} pts")
        except Exception as exc:
            print(f"    {s[:10]} -> {e[:10]}: ERROR {exc}")
        cur = nxt
    if not chunks:
        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], tz="UTC"))
    out = pd.concat(chunks)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def resample_to_15min(df, cadence, preserve_nan=False):
    """Resample to 15-min on the 15-min grid.

    preserve_nan=False: ffill small gaps (normal timeseries).
    preserve_nan=True:  leave NaN in place — missing-ness is meaningful.
    """
    if len(df) == 0:
        return df
    target_idx = pd.date_range(df.index.min().floor("15min"),
                                df.index.max().ceil("15min"),
                                freq="15min", tz="UTC")
    if preserve_nan:
        # Snap to 15-min grid, drop duplicates, reindex WITHOUT filling
        out = df.copy()
        out.index = out.index.floor("15min")
        out = out[~out.index.duplicated(keep="last")]
        out = out.reindex(target_idx)
    elif cadence == "hourly":
        out = df.reindex(target_idx, method="ffill", limit=4)
    else:
        out = df.copy()
        out.index = out.index.floor("15min")
        out = out[~out.index.duplicated(keep="last")]
        out = out.reindex(target_idx, method="ffill", limit=1)
    return out


def main():
    print("=" * 70)
    print("VECTORD FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Range: {START} -> {END}")

    client = VectordClient()
    frames = {}

    for vec_name, col_name, cadence, preserve_nan in VECTORS:
        print(f"\n[*] {vec_name} ({cadence}{', preserve NaN' if preserve_nan else ''}) -> {col_name}")
        raw = pull_chunked(client, vec_name, START, END)
        print(f"    total raw: {len(raw)} pts")
        if len(raw) == 0:
            print(f"    [!] No data — skipping")
            continue
        res = resample_to_15min(raw, cadence, preserve_nan=preserve_nan)
        res = res.rename(columns={"value": col_name})
        frames[col_name] = res[[col_name]]
        print(f"    resampled to 15-min: {len(res)} pts, "
              f"{res.index.min()} -> {res.index.max()}, "
              f"coverage {res[col_name].notna().mean():.1%}")

    if not frames:
        print("\n[!] No data pulled, aborting")
        return

    # Outer-join all frames on 15-min index
    merged = pd.concat(frames.values(), axis=1)
    print(f"\n[+] Merged: {len(merged)} rows, columns={list(merged.columns)}")
    print(f"    Coverage per column:")
    for c in merged.columns:
        nz = merged[c].notna().sum()
        print(f"      {c:<15s} {nz:>6d} non-null ({nz/len(merged):.1%})")

    out_path = OUT_DIR / "vectord_features.csv"
    merged.to_csv(out_path)
    print(f"\n[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
