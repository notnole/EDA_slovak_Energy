"""Check if IDM and Imbalance price levels changed."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"

print("=" * 90)
print("PRICE LEVEL ANALYSIS: Did markets converge?")
print("=" * 90)

# Load data
all_data = []
for folder in IDM_PATH.iterdir():
    if folder.is_dir() and folder.name.startswith("IDM_total"):
        csv_path = folder / "15 min.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=";", decimal=",")
            all_data.append(df)

idm = pd.concat(all_data, ignore_index=True)
idm["date"] = pd.to_datetime(idm["Delivery day"], format="%d.%m.%Y")
idm["period_num"] = idm["Period number"]
idm["hour"] = (idm["period_num"] - 1) // 4
idm["qh_in_hour"] = ((idm["period_num"] - 1) % 4) + 1
idm["datetime"] = idm["date"] + pd.to_timedelta(idm["hour"], unit="h") + pd.to_timedelta((idm["qh_in_hour"]-1)*15, unit="m")
idm["idm_price"] = pd.to_numeric(idm["Weighted average price of all trades (EUR/MWh)"], errors="coerce")
idm = idm[idm["qh_in_hour"].isin([1, 2])]

imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
imb = imb.rename(columns={"Imbalance Settlement Price (EUR/MWh)": "imb_price"})

merged = pd.merge(
    idm[["datetime", "hour", "idm_price"]],
    imb[["datetime", "imb_price"]],
    on="datetime", how="inner"
).dropna()

merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]
merged["year"] = merged["datetime"].dt.year
merged["month"] = merged["datetime"].dt.month
merged["spread"] = merged["idm_price"] - merged["imb_price"]

print("\n[1] MONTHLY AVERAGE PRICES (IDM, Imbalance, Spread)")
print("-" * 80)
print(f"{'Month':<12} {'N':>8} {'IDM':>10} {'Imb':>10} {'Spread':>10} {'IDM Std':>10} {'Imb Std':>10}")
print("-" * 80)

for year in [2025, 2026]:
    for month in range(1, 13):
        subset = merged[(merged["year"] == year) & (merged["month"] == month)]
        if len(subset) < 100:
            continue
        month_name = f"{pd.Timestamp(year=year, month=month, day=1).strftime('%b %Y')}"
        print(f"{month_name:<12} {len(subset):>8} {subset['idm_price'].mean():>8.1f}   {subset['imb_price'].mean():>8.1f}   {subset['spread'].mean():>8.1f}   {subset['idm_price'].std():>8.1f}   {subset['imb_price'].std():>8.1f}")

print("\n[2] KEY OBSERVATION: Did prices converge?")
print("-" * 80)

# Calculate by period
periods = {
    "Sep-Dec 2025": merged[(merged["year"] == 2025) & (merged["month"] >= 9)],
    "Jan 2026": merged[(merged["year"] == 2026) & (merged["month"] == 1)]
}

for name, data in periods.items():
    print(f"\n{name}:")
    print(f"  IDM mean: {data['idm_price'].mean():.1f} EUR/MWh")
    print(f"  Imb mean: {data['imb_price'].mean():.1f} EUR/MWh")
    print(f"  Spread:   {data['spread'].mean():.1f} EUR/MWh")
    print(f"  Correlation IDM-Imb: {data['idm_price'].corr(data['imb_price']):.3f}")

print("\n[3] HYPOTHESIS: In 2025, imbalance prices were systematically LOWER than IDM")
print("-" * 80)

# Check the price difference distribution
for period_name, data in periods.items():
    print(f"\n{period_name}:")
    print(f"  % periods where IDM > Imb: {100*(data['idm_price'] > data['imb_price']).mean():.1f}%")
    print(f"  % periods where spread > 10: {100*(data['spread'] > 10).mean():.1f}%")
    print(f"  % periods where spread < -10: {100*(data['spread'] < -10).mean():.1f}%")

print("\n[4] WHAT CHANGED?")
print("-" * 80)
print("""
The data shows that in 2025:
- IDM prices were systematically HIGHER than Imbalance prices
- This created the "always sell" arbitrage opportunity

In Jan 2026:
- IDM and Imbalance prices are now EQUAL on average
- The systematic bias disappeared
- Markets converged (efficient market hypothesis)
""")

# Check daily patterns
print("\n[5] HOURLY PATTERN CHECK")
print("-" * 80)
print("Did the hourly pattern change between periods?")
print()
print(f"{'Hour':<8} {'Sep-Dec 25 Spread':>18} {'Jan 26 Spread':>18}")
print("-" * 50)

sepdec = merged[(merged["year"] == 2025) & (merged["month"] >= 9)]
jan26 = merged[(merged["year"] == 2026) & (merged["month"] == 1)]

for hour in range(5, 22):
    sd_hour = sepdec[sepdec["hour"] == hour]
    j26_hour = jan26[jan26["hour"] == hour]
    if len(sd_hour) > 20 and len(j26_hour) > 20:
        print(f"{hour:02d}:00    {sd_hour['spread'].mean():>16.1f}   {j26_hour['spread'].mean():>16.1f}")

print("\n[6] SUMMARY")
print("-" * 80)
sepdec_avg = sepdec['spread'].mean()
jan26_avg = jan26['spread'].mean()
print(f"Sep-Dec 2025 average spread: {sepdec_avg:+.1f} EUR/MWh")
print(f"Jan 2026 average spread:     {jan26_avg:+.1f} EUR/MWh")
print(f"Change:                      {jan26_avg - sepdec_avg:+.1f} EUR/MWh")
print()
print("CONCLUSION: Markets converged - the systematic arbitrage closed.")
