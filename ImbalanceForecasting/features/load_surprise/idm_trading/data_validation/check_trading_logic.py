"""Check if trading logic is fundamentally correct."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"

print("=" * 90)
print("TRADING LOGIC VERIFICATION")
print("=" * 90)

# Load imbalance data
imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])

print("\n[1] IMBALANCE SETTLEMENT PRICE MEANING")
print("-" * 60)
print("When System Imbalance > 0 (surplus): Grid has excess power")
print("When System Imbalance < 0 (deficit): Grid lacks power")
print()
print("Question: Is Imbalance Settlement Price the price we PAY or RECEIVE?")
print()

# Check correlation between imbalance and price
corr = imb["System Imbalance (MWh)"].corr(imb["Imbalance Settlement Price (EUR/MWh)"])
print(f"Correlation between Imbalance and Price: {corr:.3f}")

# Check typical prices for surplus vs deficit
surplus = imb[imb["System Imbalance (MWh)"] > 50]
deficit = imb[imb["System Imbalance (MWh)"] < -50]
print(f"\nTypical price when SURPLUS (imb > 50 MWh): {surplus['Imbalance Settlement Price (EUR/MWh)'].mean():.1f} EUR/MWh")
print(f"Typical price when DEFICIT (imb < -50 MWh): {deficit['Imbalance Settlement Price (EUR/MWh)'].mean():.1f} EUR/MWh")

print("\n[2] TRADING STRATEGY LOGIC")
print("-" * 60)
print("""
STRATEGY: Sell on IDM, settle at imbalance

1. At time T-2h: Sell 1 MWh on IDM for period T at price P_idm
2. At time T: We don't deliver (intentional short position)
3. Settlement: We're short 1 MWh, so we pay imbalance price P_imb

PROFIT = P_idm - P_imb (what we received minus what we paid)

If P_idm > P_imb: We profit (sold high, bought back low)
If P_idm < P_imb: We lose (sold low, bought back high)
""")

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

imb = imb.rename(columns={"Imbalance Settlement Price (EUR/MWh)": "imb_price",
                           "System Imbalance (MWh)": "imbalance"})

# Merge
merged = pd.merge(
    idm[["datetime", "hour", "qh_in_hour", "idm_price"]],
    imb[["datetime", "imbalance", "imb_price"]],
    on="datetime", how="inner"
).dropna()

print("\n[3] PRICE COMPARISON BY PERIOD")
print("-" * 60)
merged["year"] = merged["datetime"].dt.year
merged["month"] = merged["datetime"].dt.month
merged["spread"] = merged["idm_price"] - merged["imb_price"]

print(f"{'Period':<15} {'N':>8} {'IDM Mean':>12} {'Imb Mean':>12} {'Spread':>10}")
print("-" * 60)

# Check ALL QH positions
for qh in [1, 2, 3, 4]:
    qh_data = merged[(merged["qh_in_hour"] == qh) & (merged["year"] == 2025)]
    if len(qh_data) > 100:
        print(f"QH{qh}           {len(qh_data):>8} {qh_data['idm_price'].mean():>10.1f}   {qh_data['imb_price'].mean():>10.1f}   {qh_data['spread'].mean():>8.1f}")

print("\n[4] WHY QH1-2 ONLY?")
print("-" * 60)
print("""
We filter for QH1-2 because:
- IDM trading closes ~1h before delivery
- For prediction at H-2, we can only trade QH1-2 of the next hour
- QH3-4 would require H-1 prediction (different model)
""")

# Check if QH filtering is causing issues
print("\n[5] QH1-2 vs QH3-4 PROFIT COMPARISON")
print("-" * 60)
qh12 = merged[merged["qh_in_hour"].isin([1, 2])]
qh34 = merged[merged["qh_in_hour"].isin([3, 4])]

for year in [2025, 2026]:
    qh12_year = qh12[qh12["year"] == year]
    qh34_year = qh34[qh34["year"] == year]
    if len(qh12_year) > 100 and len(qh34_year) > 100:
        print(f"{year} QH1-2: n={len(qh12_year):>5}, spread={qh12_year['spread'].mean():>6.1f}, win={100*(qh12_year['spread']>0).mean():>4.0f}%")
        print(f"{year} QH3-4: n={len(qh34_year):>5}, spread={qh34_year['spread'].mean():>6.1f}, win={100*(qh34_year['spread']>0).mean():>4.0f}%")
        print()

print("\n[6] SANITY CHECK: Specific Extreme Examples")
print("-" * 60)

# Find most profitable trades
top_profit = merged.nlargest(5, "spread")
print("TOP 5 MOST PROFITABLE 'SELL IDM' TRADES:")
for _, row in top_profit.iterrows():
    print(f"  {row['datetime']}: IDM={row['idm_price']:.1f}, Imb={row['imb_price']:.1f}, Profit={row['spread']:.1f}")
    print(f"    -> Sold on IDM at {row['idm_price']:.1f}, bought back from imbalance at {row['imb_price']:.1f}")

print("\nTOP 5 WORST 'SELL IDM' TRADES:")
top_loss = merged.nsmallest(5, "spread")
for _, row in top_loss.iterrows():
    print(f"  {row['datetime']}: IDM={row['idm_price']:.1f}, Imb={row['imb_price']:.1f}, Loss={row['spread']:.1f}")
    print(f"    -> Sold on IDM at {row['idm_price']:.1f}, had to buy back from imbalance at {row['imb_price']:.1f}")

print("\n[7] DOES THE PROFIT DIRECTION MAKE SENSE?")
print("-" * 60)
print("""
When imbalance price is NEGATIVE (grid pays you to take power):
  - If we SOLD on IDM at positive price, then we profit massively
  - Because we get paid twice: IDM sale + imbalance payment

Check: Are negative imbalance prices giving us big profits?
""")

neg_imb = merged[merged["imb_price"] < -100]
if len(neg_imb) > 0:
    print(f"Trades with imb_price < -100: {len(neg_imb)}")
    print(f"Average IDM price: {neg_imb['idm_price'].mean():.1f}")
    print(f"Average imb price: {neg_imb['imb_price'].mean():.1f}")
    print(f"Average spread: {neg_imb['spread'].mean():.1f} (should be very positive)")
    print(f"Win rate: {100*(neg_imb['spread']>0).mean():.0f}%")
