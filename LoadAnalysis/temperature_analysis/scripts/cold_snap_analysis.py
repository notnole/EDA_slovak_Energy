"""
Cold snap analysis for Slovak temperature data.
Identifies all periods where daily mean temp < 0C for 3+ consecutive days.
"""
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Load the 15-minute temperature data
# ---------------------------------------------------------------
print("[*] Loading temperature data from data/clean/weather/temperature_15min.csv")
df = pd.read_csv("data/clean/weather/temperature_15min.csv", parse_dates=["datetime"])

# Use temp_actual as primary, fill gaps with GFS forecast
df["temp"] = df["temp_actual"].fillna(df["temp_forecast_gfs"])
df = df.dropna(subset=["temp"])

print("[*] Data range: {} to {}".format(df.datetime.min(), df.datetime.max()))
print("[*] Total 15-min records with valid temp: {}".format(len(df)))

# ---------------------------------------------------------------
# 1. Daily mean temperature
# ---------------------------------------------------------------
df["date"] = df["datetime"].dt.date
daily = df.groupby("date")["temp"].agg(["mean", "min", "max", "count"]).reset_index()
daily.columns = ["date", "mean_temp", "min_temp", "max_temp", "n_obs"]

# Filter out days with too few observations (< 48 = half a day of 15-min data)
daily = daily[daily["n_obs"] >= 48].copy()

print("[*] Full days with sufficient data: {}".format(len(daily)))
print()
print("=" * 80)
print("  DAILY TEMPERATURE SUMMARY")
print("=" * 80)
print("  Period:       {} to {}".format(daily.date.min(), daily.date.max()))
print("  Overall mean: {:.1f} C".format(daily.mean_temp.mean()))
print("  Overall min:  {:.1f} C".format(daily.min_temp.min()))
print("  Overall max:  {:.1f} C".format(daily.max_temp.max()))
print("  Days below 0 mean: {}".format((daily.mean_temp < 0).sum()))
print()

# Monthly summary
daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M")
monthly = daily.groupby("month").agg(
    mean_temp=("mean_temp", "mean"),
    min_temp=("min_temp", "min"),
    max_temp=("max_temp", "max"),
    days=("date", "count"),
).reset_index()

print("--- Monthly Temperature Summary ---")
print("{:<12} {:>8} {:>8} {:>8} {:>6}".format("Month", "Mean", "Min", "Max", "Days"))
print("-" * 46)
for _, row in monthly.iterrows():
    print("{:<12} {:>8.1f} {:>8.1f} {:>8.1f} {:>6}".format(
        str(row.month), row.mean_temp, row.min_temp, row.max_temp, row.days
    ))
print()

# ---------------------------------------------------------------
# 2. Cold snap detection: daily mean < 0 for 3+ consecutive days
# ---------------------------------------------------------------
daily = daily.sort_values("date").reset_index(drop=True)
daily["below_zero"] = daily["mean_temp"] < 0

# Find consecutive runs
runs = []
current_start = None
current_count = 0

for i, row in daily.iterrows():
    if row["below_zero"]:
        if current_start is None:
            current_start = i
        current_count += 1
    else:
        if current_count >= 3:
            runs.append((current_start, i - 1))
        current_start = None
        current_count = 0

# Check if last run extends to end
if current_count >= 3:
    runs.append((current_start, len(daily) - 1))

print("=" * 105)
print("  COLD SNAP PERIODS (daily mean temp < 0 C for 3+ consecutive days)")
print("=" * 105)

if not runs:
    print("  [!] No cold snap periods found in the data.")
else:
    print("  Found {} cold snap period(s)".format(len(runs)))
    print()
    header = "{:<4} {:<14} {:<14} {:<10} {:>10} {:>10} {:<14} {:>16}".format(
        "#", "Start Date", "End Date", "Duration", "Min Temp", "Mean Temp",
        "Coldest Day", "Coldest DayTemp"
    )
    print(header)
    print("-" * 105)

    for idx, (start_i, end_i) in enumerate(runs, 1):
        snap = daily.iloc[start_i:end_i + 1]
        start_date = snap["date"].iloc[0]
        end_date = snap["date"].iloc[-1]
        duration = len(snap)
        min_temp = snap["min_temp"].min()
        mean_temp = snap["mean_temp"].mean()
        coldest_day_idx = snap["mean_temp"].idxmin()
        coldest_day = daily.loc[coldest_day_idx, "date"]
        coldest_day_mean = daily.loc[coldest_day_idx, "mean_temp"]

        print("{:<4} {:<14} {:<14} {:<10} {:>10.1f} {:>10.1f} {:<14} {:>16.1f}".format(
            idx, str(start_date), str(end_date), "{} days".format(duration),
            min_temp, mean_temp, str(coldest_day), coldest_day_mean
        ))

    print()
    # Detailed daily breakdown for each cold snap
    for idx, (start_i, end_i) in enumerate(runs, 1):
        snap = daily.iloc[start_i:end_i + 1]
        print("--- Cold Snap #{}: {} to {} (daily detail) ---".format(
            idx, snap.date.iloc[0], snap.date.iloc[-1]
        ))
        print("{:<14} {:>8} {:>8} {:>8}".format("Date", "Mean", "Min", "Max"))
        print("-" * 42)
        for _, row in snap.iterrows():
            print("{:<14} {:>8.1f} {:>8.1f} {:>8.1f}".format(
                str(row.date), row.mean_temp, row.min_temp, row.max_temp
            ))
        print()

# ---------------------------------------------------------------
# 3. Weekly rolling average & cold weeks
# ---------------------------------------------------------------
daily["date_dt"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date_dt")
daily["weekly_rolling_mean"] = daily["mean_temp"].rolling(7, center=True, min_periods=4).mean()

cold_weeks = daily[daily["weekly_rolling_mean"] < 0].copy()

print("=" * 80)
print("  WEEKLY ROLLING AVERAGE TEMPERATURE (7-day centered)")
print("=" * 80)

if cold_weeks.empty:
    print("  [!] No weeks with rolling average below 0 C.")
else:
    print("  Days where 7-day rolling mean < 0 C: {}".format(len(cold_weeks)))
    print()

    # Group into contiguous periods
    cold_weeks = cold_weeks.reset_index(drop=True)
    cold_weeks["gap"] = (pd.to_datetime(cold_weeks["date"]).diff().dt.days > 1).cumsum()

    cold_periods = cold_weeks.groupby("gap").agg(
        start=("date", "first"),
        end=("date", "last"),
        days=("date", "count"),
        mean_rolling=("weekly_rolling_mean", "mean"),
        min_rolling=("weekly_rolling_mean", "min"),
    ).reset_index()

    print("{:<4} {:<14} {:<14} {:>6} {:>10} {:>10}".format(
        "#", "Start", "End", "Days", "Mean Roll", "Min Roll"
    ))
    print("-" * 62)
    for i, row in cold_periods.iterrows():
        print("{:<4} {:<14} {:<14} {:>6} {:>10.1f} {:>10.1f}".format(
            i + 1, str(row.start), str(row.end), row.days,
            row.mean_rolling, row.min_rolling
        ))

print()

# ---------------------------------------------------------------
# 4. Also show near-zero periods (daily mean < 2C for 5+ days)
# ---------------------------------------------------------------
daily["below_2"] = daily["mean_temp"] < 2

runs_cool = []
current_start = None
current_count = 0

for i, row in daily.iterrows():
    if row["below_2"]:
        if current_start is None:
            current_start = i
        current_count += 1
    else:
        if current_count >= 5:
            runs_cool.append((current_start, i - 1))
        current_start = None
        current_count = 0

if current_count >= 5:
    runs_cool.append((current_start, len(daily) - 1))

print("=" * 100)
print("  EXTENDED COLD PERIODS (daily mean temp < 2 C for 5+ consecutive days)")
print("=" * 100)

if not runs_cool:
    print("  [!] No extended cold periods found.")
else:
    print("  Found {} period(s)".format(len(runs_cool)))
    print()
    print("{:<4} {:<14} {:<14} {:<10} {:>10} {:>10}".format(
        "#", "Start Date", "End Date", "Duration", "Min Temp", "Mean Temp"
    ))
    print("-" * 66)

    for idx, (start_i, end_i) in enumerate(runs_cool, 1):
        snap = daily.iloc[start_i:end_i + 1]
        start_date = snap["date"].iloc[0]
        end_date = snap["date"].iloc[-1]
        duration = len(snap)
        min_temp = snap["min_temp"].min()
        mean_temp = snap["mean_temp"].mean()
        print("{:<4} {:<14} {:<14} {:<10} {:>10.1f} {:>10.1f}".format(
            idx, str(start_date), str(end_date), "{} days".format(duration),
            min_temp, mean_temp
        ))

print()

# ---------------------------------------------------------------
# 5. Comparison with the reference Jan 2026 cold snap
# ---------------------------------------------------------------
jan_2026 = daily[pd.to_datetime(daily["date"]).dt.to_period("M") == "2026-01"]
if not jan_2026.empty:
    print("=" * 80)
    print("  REFERENCE: January 2026 Summary")
    print("=" * 80)
    print("  Mean temp:  {:.1f} C".format(jan_2026.mean_temp.mean()))
    print("  Min temp:   {:.1f} C".format(jan_2026.min_temp.min()))
    print("  Max temp:   {:.1f} C".format(jan_2026.max_temp.max()))
    print("  Days below 0 mean: {} / {}".format(
        (jan_2026.mean_temp < 0).sum(), len(jan_2026)
    ))
    print()

print("[+] Cold snap analysis complete.")
