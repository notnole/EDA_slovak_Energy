"""
DA Price Seasonality Decomposition
===================================
Decomposes Slovak DA electricity prices into trend, seasonal, and residual.
Loads directly from raw OKTE DA files in RawData/DA_market/.
Handles both hourly (2024-2025) and QH 15-min (2026) resolution.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = Path(__file__).resolve().parents[3] / "RawData" / "DA_market"


def load_raw_da():
    """Load all raw DA CSVs, aggregate QH to hourly."""
    files = sorted(RAW_DIR.glob("Celkove_vysledky_DT_*.csv"))
    frames = []
    for f in files:
        print(f"[*] Loading {f.name}")
        df = pd.read_csv(f, sep=";", encoding="utf-8")
        # European decimal -> float
        df["price"] = df["Cena SK (EUR/MWh)"].astype(str).str.replace(",", ".").astype(float)
        # Parse date (D.M.YYYY)
        df["date"] = pd.to_datetime(df["Obchodny den"] if "Obchodny den" in df.columns
                                    else df.iloc[:, 0], format="%d.%m.%Y", dayfirst=True)
        resolution = df["Perioda (min)"].astype(int) if "Perioda (min)" in df.columns \
            else df.iloc[:, 3].astype(int)
        period_num = df.iloc[:, 1].astype(int)
        # Map period to hour
        df["hour"] = np.where(resolution == 60, period_num - 1, (period_num - 1) // 4)
        frames.append(df[["date", "hour", "price"]])

    all_df = pd.concat(frames, ignore_index=True)
    # Aggregate QH -> hourly (mean)
    hourly = all_df.groupby(["date", "hour"])["price"].mean().reset_index()
    hourly["datetime"] = hourly["date"] + pd.to_timedelta(hourly["hour"], unit="h")
    hourly = hourly.sort_values("datetime").set_index("datetime")
    hourly.rename(columns={"price": "da_price"}, inplace=True)

    print(f"[+] {len(hourly)} hourly obs from {hourly.index.min()} to {hourly.index.max()}")
    return hourly


def sequential_multi_stl(da):
    """Sequential STL: daily (24h) then weekly (168h) on remainder."""
    print("\n" + "=" * 60)
    print("[*] SEQUENTIAL STL (daily + weekly)")
    print("=" * 60)

    stl_d = STL(da, period=24, robust=True).fit()
    remainder = stl_d.trend + stl_d.resid
    stl_w = STL(remainder, period=168, robust=True).fit()

    return stl_d.seasonal, stl_w.seasonal, stl_w.trend, stl_w.resid


def main():
    print("=" * 60)
    print("[*] DA PRICE DECOMPOSITION (from raw OKTE files)")
    print("=" * 60)

    hourly = load_raw_da()
    da = hourly["da_price"].copy()

    # Fill small gaps
    full_idx = pd.date_range(da.index.min(), da.index.max(), freq="h")
    missing = len(full_idx) - len(da)
    print(f"[*] Missing hours: {missing} / {len(full_idx)}")
    if missing > 0:
        da = da.reindex(full_idx).interpolate(method="linear", limit=3).dropna()
        print(f"[*] After interpolation: {len(da)} hours")

    total_var = np.var(da)
    total_std = np.std(da)

    # --- Single STL comparisons ---
    for period, label in [(24, "Daily"), (168, "Weekly")]:
        stl = STL(da, period=period, robust=True).fit()
        tv = np.var(stl.trend)
        sv = np.var(stl.seasonal)
        rv = np.var(stl.resid)
        print(f"\n--- STL period={period} ({label}) ---")
        print(f"  Trend:    {100*tv/total_var:5.1f}%   Seasonal: {100*sv/total_var:5.1f}%   Residual: {100*rv/total_var:5.1f}%")

    # --- Sequential multi-period STL ---
    daily_s, weekly_s, trend, resid = sequential_multi_stl(da)
    trend_var = np.var(trend)
    daily_var = np.var(daily_s)
    weekly_var = np.var(weekly_s)
    resid_var = np.var(resid)
    combined_s = daily_s + weekly_s
    combined_var = np.var(combined_s)

    print(f"\n{'='*60}")
    print(f"  VARIANCE DECOMPOSITION")
    print(f"{'='*60}")
    print(f"  Total variance:  {total_var:.1f} (EUR/MWh)^2    std = {total_std:.1f}")
    print(f"")
    print(f"  {'Component':<28} {'Var':>8} {'%':>7}  {'StdDev':>7}")
    print(f"  {'-'*52}")
    for name, v in [("Trend", trend_var), ("Daily seasonal (24h)", daily_var),
                    ("Weekly seasonal (168h)", weekly_var),
                    ("Combined seasonal", combined_var), ("Residual", resid_var)]:
        print(f"  {name:<28} {v:>8.1f} {100*v/total_var:>6.1f}%  {np.sqrt(v):>7.1f}")

    # R-squared
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((da - da.mean())**2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n  R-squared (trend+seasonal): {r2:.4f}")

    # --- Hourly profile ---
    print(f"\n{'='*60}")
    print(f"  HOURLY PROFILE")
    print(f"{'='*60}")
    hp = da.groupby(da.index.hour).agg(["mean", "std"])
    hp.columns = ["mean", "std"]
    for h in range(24):
        bar = "#" * int(hp.loc[h, "mean"] / 4)
        print(f"  H{h:02d}: {hp.loc[h,'mean']:6.1f} +/- {hp.loc[h,'std']:5.1f}  {bar}")

    # --- Monthly profile ---
    print(f"\n{'='*60}")
    print(f"  MONTHLY PROFILE")
    print(f"{'='*60}")
    mp = da.groupby(da.index.month).agg(["mean", "std"])
    mp.columns = ["mean", "std"]
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        if m in mp.index:
            bar = "#" * int(mp.loc[m, "mean"] / 4)
            print(f"  {months[m-1]}: {mp.loc[m,'mean']:6.1f} +/- {mp.loc[m,'std']:5.1f}  {bar}")

    # --- Weekday vs weekend ---
    wd = da[da.index.dayofweek < 5]
    we = da[da.index.dayofweek >= 5]
    print(f"\n  Weekday mean: {wd.mean():.1f}   Weekend mean: {we.mean():.1f}   Diff: {wd.mean()-we.mean():.1f} EUR/MWh")

    # --- Day of week ---
    print(f"\n{'='*60}")
    print(f"  DAY-OF-WEEK PROFILE")
    print(f"{'='*60}")
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow = da.groupby(da.index.dayofweek).mean()
    for d in range(7):
        if d in dow.index:
            print(f"  {dow_names[d]}: {dow[d]:6.1f}")

    # --- Final answer ---
    print(f"\n{'='*60}")
    print(f"  FINAL ANSWER")
    print(f"{'='*60}")
    print(f"  Trend:       {100*trend_var/total_var:.1f}% of price variance")
    print(f"  Seasonality: {100*combined_var/total_var:.1f}% of price variance")
    print(f"    - Daily:   {100*daily_var/total_var:.1f}%")
    print(f"    - Weekly:  {100*weekly_var/total_var:.1f}%")
    print(f"  Residual:    {100*resid_var/total_var:.1f}% of price variance")
    print(f"")
    print(f"  Predictable (trend+seasonal): {100*(trend_var+combined_var)/total_var:.1f}%")
    print(f"  Unpredictable (residual):     {100*resid_var/total_var:.1f}%")


if __name__ == "__main__":
    main()
