"""Check extreme price periods and filter impact."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"

print("=" * 90)
print("EXTREME PRICE ANALYSIS & FILTER IMPACT")
print("=" * 90)

# Load IDM
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

# Load imbalance WITHOUT filtering
imb_raw = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
imb_raw = imb_raw[["datetime", "System Imbalance (MWh)", "Imbalance Settlement Price (EUR/MWh)"]].copy()
imb_raw.columns = ["datetime", "imbalance", "imb_price"]

print("\n[1] IMBALANCE PRICE EXTREMES (unfiltered)")
print("-" * 60)
print(f"Total rows: {len(imb_raw)}")
print(f"Price range: {imb_raw['imb_price'].min():.0f} to {imb_raw['imb_price'].max():.0f} EUR/MWh")

# Show most extreme
print("\nMost extreme positive prices:")
extreme_high = imb_raw.nlargest(10, "imb_price")
print(extreme_high[["datetime", "imbalance", "imb_price"]].to_string())

print("\nMost extreme negative prices:")
extreme_low = imb_raw.nsmallest(10, "imb_price")
print(extreme_low[["datetime", "imbalance", "imb_price"]].to_string())

# Check QH distribution of extremes
print("\n[2] EXTREME PRICES BY QH POSITION")
print("-" * 60)
imb_raw["qh_in_hour"] = ((imb_raw["datetime"].dt.minute // 15) % 4) + 1

for qh in [1, 2, 3, 4]:
    qh_data = imb_raw[imb_raw["qh_in_hour"] == qh]
    n_extreme = len(qh_data[qh_data["imb_price"].abs() > 500])
    print(f"QH{qh}: {n_extreme} periods with |price| > 500 EUR/MWh")

# We only trade QH1-2, check if we're missing extreme prices
print("\n[3] COMPARISON: QH1-2 vs QH3-4")
print("-" * 60)
qh12 = imb_raw[imb_raw["qh_in_hour"].isin([1, 2])]
qh34 = imb_raw[imb_raw["qh_in_hour"].isin([3, 4])]
print(f"QH1-2 price range: {qh12['imb_price'].min():.0f} to {qh12['imb_price'].max():.0f}")
print(f"QH3-4 price range: {qh34['imb_price'].min():.0f} to {qh34['imb_price'].max():.0f}")

# Impact of imbalance filter
print("\n[4] IMPACT OF |IMBALANCE| <= 300 FILTER")
print("-" * 60)
imb_filtered = imb_raw[imb_raw["imbalance"].abs() <= 300]
print(f"Rows before filter: {len(imb_raw)}")
print(f"Rows after filter:  {len(imb_filtered)}")
print(f"Removed: {len(imb_raw) - len(imb_filtered)} ({(len(imb_raw) - len(imb_filtered))/len(imb_raw)*100:.1f}%)")

print("\nPrice stats comparison:")
print(f"{'Metric':<20} {'Unfiltered':>15} {'Filtered':>15}")
print("-" * 55)
print(f"{'Mean':<20} {imb_raw['imb_price'].mean():>13.1f}   {imb_filtered['imb_price'].mean():>13.1f}")
print(f"{'Std':<20} {imb_raw['imb_price'].std():>13.1f}   {imb_filtered['imb_price'].std():>13.1f}")
print(f"{'Min':<20} {imb_raw['imb_price'].min():>13.1f}   {imb_filtered['imb_price'].min():>13.1f}")
print(f"{'Max':<20} {imb_raw['imb_price'].max():>13.1f}   {imb_filtered['imb_price'].max():>13.1f}")
print(f"{'5th pct':<20} {imb_raw['imb_price'].quantile(0.05):>13.1f}   {imb_filtered['imb_price'].quantile(0.05):>13.1f}")
print(f"{'95th pct':<20} {imb_raw['imb_price'].quantile(0.95):>13.1f}   {imb_filtered['imb_price'].quantile(0.95):>13.1f}")

# Merge WITH and WITHOUT filter
print("\n[5] TRADING PROFIT COMPARISON: WITH vs WITHOUT FILTER")
print("-" * 60)

# Only QH1-2 for IDM
idm_qh12 = idm[idm["qh_in_hour"].isin([1, 2])].copy()

# Merge unfiltered
merged_unfiltered = pd.merge(
    idm_qh12[["datetime", "hour", "idm_price"]],
    imb_raw[["datetime", "imb_price"]],
    on="datetime", how="inner"
).dropna()

# Merge filtered
merged_filtered = pd.merge(
    idm_qh12[["datetime", "hour", "idm_price"]],
    imb_filtered[["datetime", "imb_price"]],
    on="datetime", how="inner"
).dropna()

# Filter for trading hours
merged_unfiltered = merged_unfiltered[(merged_unfiltered["hour"] >= 5) & (merged_unfiltered["hour"] < 22)]
merged_filtered = merged_filtered[(merged_filtered["hour"] >= 5) & (merged_filtered["hour"] < 22)]

merged_unfiltered["spread"] = merged_unfiltered["idm_price"] - merged_unfiltered["imb_price"]
merged_filtered["spread"] = merged_filtered["idm_price"] - merged_filtered["imb_price"]

merged_unfiltered["year"] = merged_unfiltered["datetime"].dt.year
merged_unfiltered["month"] = merged_unfiltered["datetime"].dt.month
merged_filtered["year"] = merged_filtered["datetime"].dt.year
merged_filtered["month"] = merged_filtered["datetime"].dt.month

print(f"\n{'Period':<15} {'Unfiltered Total':>18} {'Filtered Total':>18} {'Difference':>15}")
print("-" * 70)

# Full 2025
uf_2025 = merged_unfiltered[merged_unfiltered["year"] == 2025]
f_2025 = merged_filtered[merged_filtered["year"] == 2025]
diff_2025 = uf_2025["spread"].sum() - f_2025["spread"].sum()
print(f"{'Full 2025':<15} {uf_2025['spread'].sum():>16.0f}   {f_2025['spread'].sum():>16.0f}   {diff_2025:>13.0f}")

# Jan 2025
uf_jan25 = merged_unfiltered[(merged_unfiltered["year"] == 2025) & (merged_unfiltered["month"] == 1)]
f_jan25 = merged_filtered[(merged_filtered["year"] == 2025) & (merged_filtered["month"] == 1)]
diff_jan25 = uf_jan25["spread"].sum() - f_jan25["spread"].sum()
print(f"{'Jan 2025':<15} {uf_jan25['spread'].sum():>16.0f}   {f_jan25['spread'].sum():>16.0f}   {diff_jan25:>13.0f}")

# Feb 2025
uf_feb25 = merged_unfiltered[(merged_unfiltered["year"] == 2025) & (merged_unfiltered["month"] == 2)]
f_feb25 = merged_filtered[(merged_filtered["year"] == 2025) & (merged_filtered["month"] == 2)]
diff_feb25 = uf_feb25["spread"].sum() - f_feb25["spread"].sum()
print(f"{'Feb 2025':<15} {uf_feb25['spread'].sum():>16.0f}   {f_feb25['spread'].sum():>16.0f}   {diff_feb25:>13.0f}")

# Jan 2026
uf_jan26 = merged_unfiltered[(merged_unfiltered["year"] == 2026) & (merged_unfiltered["month"] == 1)]
f_jan26 = merged_filtered[(merged_filtered["year"] == 2026) & (merged_filtered["month"] == 1)]
diff_jan26 = uf_jan26["spread"].sum() - f_jan26["spread"].sum()
print(f"{'Jan 2026':<15} {uf_jan26['spread'].sum():>16.0f}   {f_jan26['spread'].sum():>16.0f}   {diff_jan26:>13.0f}")

# Check extreme spread values
print("\n[6] MOST EXTREME SPREAD VALUES (UNFILTERED)")
print("-" * 60)

print("\nLargest positive spreads (sell IDM, huge profit):")
top_pos = merged_unfiltered.nlargest(10, "spread")
print(top_pos[["datetime", "idm_price", "imb_price", "spread"]].to_string())

print("\nLargest negative spreads (sell IDM, huge loss):")
top_neg = merged_unfiltered.nsmallest(10, "spread")
print(top_neg[["datetime", "idm_price", "imb_price", "spread"]].to_string())

# Win rate comparison
print("\n[7] WIN RATE COMPARISON BY MONTH")
print("-" * 60)
print(f"{'Month':<12} {'Unfiltered':>12} {'Filtered':>12} {'N Unf':>10} {'N Filt':>10}")
print("-" * 60)

for year in [2025, 2026]:
    for month in range(1, 13):
        uf = merged_unfiltered[(merged_unfiltered["year"] == year) & (merged_unfiltered["month"] == month)]
        f = merged_filtered[(merged_filtered["year"] == year) & (merged_filtered["month"] == month)]
        if len(uf) < 50 or len(f) < 50:
            continue
        uf_win = (uf["spread"] > 0).mean() * 100
        f_win = (f["spread"] > 0).mean() * 100
        month_name = f"{pd.Timestamp(year=year, month=month, day=1).strftime('%b')} {year}"
        print(f"{month_name:<12} {uf_win:>10.0f}%   {f_win:>10.0f}%   {len(uf):>10} {len(f):>10}")

print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)
print("""
The |imbalance| <= 300 filter removes extreme events, which may include
both extreme profits AND extreme losses. The net effect depends on whether
extreme events are more often in our favor (sell IDM > buy imbalance) or
against us.
""")
