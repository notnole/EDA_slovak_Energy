"""Validate IDM and Imbalance data for correctness."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"

print("=" * 80)
print("DATA VALIDATION")
print("=" * 80)

# 1. Load and inspect IDM data
print("\n[1] IDM DATA")
print("-" * 40)

all_data = []
for folder in IDM_PATH.iterdir():
    if folder.is_dir() and folder.name.startswith("IDM_total"):
        csv_path = folder / "15 min.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=";", decimal=",")
            all_data.append(df)
            print(f"  {folder.name}: {len(df)} rows")

idm = pd.concat(all_data, ignore_index=True)
print(f"\nTotal IDM rows: {len(idm)}")

# Show column names
print(f"\nIDM Columns:")
for col in idm.columns:
    print(f"  - {col}")

# Check price column
price_col = "Weighted average price of all trades (EUR/MWh)"
print(f"\nIDM Price column: '{price_col}'")
idm["idm_price"] = pd.to_numeric(idm[price_col], errors="coerce")
print(f"  Non-null values: {idm['idm_price'].notna().sum()}")
print(f"  Null values: {idm['idm_price'].isna().sum()}")
print(f"  Min: {idm['idm_price'].min()}")
print(f"  Max: {idm['idm_price'].max()}")
print(f"  Mean: {idm['idm_price'].mean():.2f}")

# Show sample values
print(f"\nSample IDM rows:")
idm["date"] = pd.to_datetime(idm["Delivery day"], format="%d.%m.%Y")
sample = idm[idm["date"] >= "2025-09-01"].head(10)
print(sample[["Delivery day", "Period number", price_col]].to_string())

# 2. Load and inspect Imbalance data
print("\n\n[2] IMBALANCE DATA")
print("-" * 40)

imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
print(f"Total rows: {len(imb)}")
print(f"\nImbalance Columns:")
for col in imb.columns:
    print(f"  - {col}")

price_col_imb = "Imbalance Settlement Price (EUR/MWh)"
print(f"\nImbalance Price column: '{price_col_imb}'")
print(f"  Non-null values: {imb[price_col_imb].notna().sum()}")
print(f"  Null values: {imb[price_col_imb].isna().sum()}")
print(f"  Min: {imb[price_col_imb].min()}")
print(f"  Max: {imb[price_col_imb].max()}")
print(f"  Mean: {imb[price_col_imb].mean():.2f}")

# Show sample values
print(f"\nSample Imbalance rows (Sep 2025):")
sample_imb = imb[(imb["datetime"] >= "2025-09-01") & (imb["datetime"] < "2025-09-02")]
print(sample_imb[["datetime", "System Imbalance (MWh)", price_col_imb]].head(10).to_string())

# 3. Check extreme values
print("\n\n[3] EXTREME VALUES CHECK")
print("-" * 40)

print("\nIDM prices > 500 EUR/MWh:")
extreme_idm = idm[idm["idm_price"] > 500]
print(f"  Count: {len(extreme_idm)}")
if len(extreme_idm) > 0:
    print(extreme_idm[["Delivery day", "Period number", price_col]].head(5).to_string())

print("\nIDM prices < 0 EUR/MWh:")
negative_idm = idm[idm["idm_price"] < 0]
print(f"  Count: {len(negative_idm)}")

print("\nImbalance prices > 500 EUR/MWh:")
extreme_imb = imb[imb[price_col_imb] > 500]
print(f"  Count: {len(extreme_imb)}")

print("\nImbalance prices < -100 EUR/MWh:")
negative_imb = imb[imb[price_col_imb] < -100]
print(f"  Count: {len(negative_imb)}")

# 4. Merge and compare specific examples
print("\n\n[4] MERGED DATA VALIDATION")
print("-" * 40)

# Parse IDM datetime
idm["period_num"] = idm["Period number"]
idm["hour"] = (idm["period_num"] - 1) // 4
idm["qh_in_hour"] = ((idm["period_num"] - 1) % 4) + 1
idm["datetime"] = idm["date"] + pd.to_timedelta(idm["hour"], unit="h") + pd.to_timedelta((idm["qh_in_hour"]-1)*15, unit="m")

# Filter QH1-2
idm_filtered = idm[idm["qh_in_hour"].isin([1, 2])].copy()

# Filter imbalance
imb_clean = imb[imb["System Imbalance (MWh)"].abs() <= 300].copy()
imb_clean = imb_clean.rename(columns={price_col_imb: "imb_price"})

# Merge
merged = pd.merge(
    idm_filtered[["datetime", "hour", "idm_price"]],
    imb_clean[["datetime", "imb_price"]],
    on="datetime",
    how="inner"
)
merged = merged.dropna()
merged["spread"] = merged["idm_price"] - merged["imb_price"]
merged["year"] = merged["datetime"].dt.year
merged["month"] = merged["datetime"].dt.month

print(f"Merged rows: {len(merged)}")
print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")

# Show specific examples
print("\nSample merged data (Sep 1, 2025):")
sample_merged = merged[(merged["datetime"] >= "2025-09-01") & (merged["datetime"] < "2025-09-02")]
print(sample_merged[["datetime", "idm_price", "imb_price", "spread"]].head(10).to_string())

# 5. Monthly statistics
print("\n\n[5] MONTHLY PRICE STATISTICS")
print("-" * 40)

print(f"\n{'Month':<12} {'N':>8} {'IDM Avg':>10} {'Imb Avg':>10} {'Spread':>10} {'Spread%':>10}")
print("-" * 65)

for year in [2025, 2026]:
    for month in range(1, 13):
        subset = merged[(merged["year"] == year) & (merged["month"] == month)]
        if len(subset) < 50:
            continue
        month_name = f"{pd.Timestamp(year=year, month=month, day=1).strftime('%b')} {year}"
        idm_avg = subset["idm_price"].mean()
        imb_avg = subset["imb_price"].mean()
        spread = subset["spread"].mean()
        spread_pct = spread / idm_avg * 100 if idm_avg != 0 else 0
        print(f"{month_name:<12} {len(subset):>8} {idm_avg:>8.1f}   {imb_avg:>8.1f}   {spread:>8.1f}   {spread_pct:>8.1f}%")

# 6. Check for suspicious patterns
print("\n\n[6] SUSPICIOUS PATTERNS CHECK")
print("-" * 40)

# Check if IDM and Imbalance prices are correlated
corr = merged["idm_price"].corr(merged["imb_price"])
print(f"Correlation IDM vs Imbalance price: {corr:.3f}")

# Check spread distribution
print(f"\nSpread (IDM - Imbalance) distribution:")
print(f"  Mean: {merged['spread'].mean():.1f}")
print(f"  Std: {merged['spread'].std():.1f}")
print(f"  5th percentile: {merged['spread'].quantile(0.05):.1f}")
print(f"  95th percentile: {merged['spread'].quantile(0.95):.1f}")

# Check win rate consistency
print(f"\nWin rate (spread > 0) by month:")
for year in [2025, 2026]:
    for month in range(1, 13):
        subset = merged[(merged["year"] == year) & (merged["month"] == month)]
        if len(subset) < 50:
            continue
        month_name = f"{pd.Timestamp(year=year, month=month, day=1).strftime('%b')} {year}"
        win_rate = (subset["spread"] > 0).mean() * 100
        print(f"  {month_name}: {win_rate:.0f}%")
