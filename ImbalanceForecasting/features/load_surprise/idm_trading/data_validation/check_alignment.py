"""Check fundamental alignment between IDM and Imbalance data."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"

print("=" * 90)
print("FUNDAMENTAL ALIGNMENT CHECK")
print("=" * 90)

# Load raw IDM - show exact columns
folder = list(IDM_PATH.iterdir())[0]
csv_path = folder / "15 min.csv"
idm_sample = pd.read_csv(csv_path, sep=";", decimal=",", nrows=20)

print("\n[1] RAW IDM DATA SAMPLE")
print("-" * 60)
print(idm_sample[["Delivery day", "Period number", "Period", "Weighted average price of all trades (EUR/MWh)"]].head(10).to_string())

# Check period interpretation
print("\n[2] PERIOD NUMBER INTERPRETATION")
print("-" * 60)
print("Period 1 = 00:00-00:15 (QH1 of hour 0)")
print("Period 2 = 00:15-00:30 (QH2 of hour 0)")
print("Period 3 = 00:30-00:45 (QH3 of hour 0)")
print("Period 4 = 00:45-01:00 (QH4 of hour 0)")
print("Period 5 = 01:00-01:15 (QH1 of hour 1)")
print("...")
print("Period 96 = 23:45-00:00 (QH4 of hour 23)")

# Load imbalance - show raw structure
imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"], nrows=20)
print("\n[3] RAW IMBALANCE DATA SAMPLE")
print("-" * 60)
print(imb[["datetime", "Date", "Settlement Term", "Imbalance Settlement Price (EUR/MWh)"]].head(10).to_string())

# Check: What does imbalance datetime represent?
print("\n[4] IMBALANCE DATETIME MEANING")
print("-" * 60)
print("datetime column represents: START of 15-min period")
print("  2025-01-01 00:00:00 = period 00:00-00:15")
print("  2025-01-01 00:15:00 = period 00:15-00:30")

# Now check the actual conversion
print("\n[5] IDM DATETIME CONVERSION CHECK")
print("-" * 60)

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

# Current conversion formula
idm["hour"] = (idm["period_num"] - 1) // 4
idm["qh_in_hour"] = ((idm["period_num"] - 1) % 4) + 1
idm["datetime"] = idm["date"] + pd.to_timedelta(idm["hour"], unit="h") + pd.to_timedelta((idm["qh_in_hour"]-1)*15, unit="m")

print("Sample conversions:")
for pn in [1, 2, 3, 4, 5, 6, 7, 8, 93, 94, 95, 96]:
    sample = idm[idm["period_num"] == pn].iloc[0]
    print(f"  Period {pn:2d} -> Hour {int(sample['hour']):2d}, QH{int(sample['qh_in_hour'])} -> {sample['datetime'].strftime('%H:%M')}")

# Verify a specific date matches
print("\n[6] SPECIFIC DATE VERIFICATION: Jan 2, 2025")
print("-" * 60)

# IDM for Jan 2, 2025, period 1
idm_jan2 = idm[(idm["date"] == "2025-01-02") & (idm["period_num"] == 1)]
print(f"IDM: Date={idm_jan2['Delivery day'].values[0]}, Period={idm_jan2['Period number'].values[0]}")
print(f"     Converted datetime: {idm_jan2['datetime'].values[0]}")
print(f"     Price: {idm_jan2['Weighted average price of all trades (EUR/MWh)'].values[0]}")

# Imbalance for same period
imb_full = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
imb_jan2 = imb_full[imb_full["datetime"] == "2025-01-02 00:00:00"]
print(f"\nImbalance: datetime={imb_jan2['datetime'].values[0]}")
print(f"     Settlement Term: {imb_jan2['Settlement Term'].values[0]}")
print(f"     Price: {imb_jan2['Imbalance Settlement Price (EUR/MWh)'].values[0]:.2f}")

# Check if these represent the SAME 15-min period
print("\n[7] CRITICAL CHECK: Are these the same 15-min period?")
print("-" * 60)
print("IDM 'Delivery day' + Period 1 = delivery at 00:00-00:15 on that day")
print("Imbalance datetime 00:00:00 = settlement for 00:00-00:15")
print("\n=> YES, they should match")

# Now let's check a few merged examples
print("\n[8] MERGED EXAMPLES VERIFICATION")
print("-" * 60)

idm["idm_price"] = pd.to_numeric(idm["Weighted average price of all trades (EUR/MWh)"], errors="coerce")
imb_full = imb_full.rename(columns={"Imbalance Settlement Price (EUR/MWh)": "imb_price"})

merged = pd.merge(
    idm[["datetime", "Delivery day", "Period number", "hour", "qh_in_hour", "idm_price"]],
    imb_full[["datetime", "imb_price"]],
    on="datetime", how="inner"
)

# Show a few examples
print(merged[merged["datetime"].dt.date == pd.Timestamp("2025-01-02").date()].head(10).to_string())

# Check for any datetime mismatches
print("\n[9] DATE RANGE CHECK")
print("-" * 60)
print(f"IDM datetime range: {idm['datetime'].min()} to {idm['datetime'].max()}")
print(f"Imbalance datetime range: {imb_full['datetime'].min()} to {imb_full['datetime'].max()}")
print(f"Merged rows: {len(merged)}")
print(f"IDM rows: {len(idm)}")
print(f"Imbalance rows: {len(imb_full)}")
