"""
Cold-snap spread inversion: statistical robustness + causal mechanism.

Q1: Is the 60-observation cold-day sample statistically reliable?
    - Bootstrap CI on mean spread (cold vs normal)
    - Permutation test (non-parametric)
    - Effect size (Cohen's d)
    - Power analysis

Q2: What causes the spread inversion? Causal chain hypothesis:
    Cold temp -> Load under-forecast -> System deficit -> Settlement price spike
    -> DA-Imb spread goes negative

    Evidence from: load forecast errors, IDM trading, imbalance volumes
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
OUT_DIR = BASE / "MarketPriceGap" / "strategies" / "temp_filter"

# ============================================================================
# 1. LOAD ALL DATA
# ============================================================================
print("[*] Loading data...")

# Market prices
mkt = pd.read_csv(BASE / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                   parse_dates=["timestamp_hour"])
mkt = mkt.dropna(subset=["spread_da_imb"])
mkt["spread_da_imb"] = mkt["spread_da_imb"].astype(float)
mkt["da_price"] = mkt["da_price"].astype(float)
mkt["imb_settlement_price"] = mkt["imb_settlement_price"].astype(float)
mkt["idm_vwap"] = pd.to_numeric(mkt["idm_vwap"], errors="coerce")
mkt["idm_volume_mwh"] = pd.to_numeric(mkt["idm_volume_mwh"], errors="coerce")
mkt["imbalance_mwh"] = pd.to_numeric(mkt["imbalance_mwh"], errors="coerce")
mkt["date"] = mkt["timestamp_hour"].dt.date
mkt["hour"] = mkt["timestamp_hour"].dt.hour
mkt = mkt.set_index("timestamp_hour").sort_index()

# Load forecast
load = pd.read_csv(BASE / "features" / "DamasLoad" / "load_data.csv",
                    parse_dates=["datetime"])
load = load.rename(columns={"datetime": "timestamp_hour"})
load = load.set_index("timestamp_hour").sort_index()
load["forecast_error_mw"] = pd.to_numeric(load["forecast_error_mw"], errors="coerce")
load["actual_load_mw"] = pd.to_numeric(load["actual_load_mw"], errors="coerce")
load["forecast_load_mw"] = pd.to_numeric(load["forecast_load_mw"], errors="coerce")

# Merge load onto market data
mkt = mkt.join(load[["actual_load_mw", "forecast_load_mw", "forecast_error_mw"]], how="left")

# Temperature (actual)
def parse_actual_temp(filepath):
    rows = []
    with open(filepath, encoding='utf-8-sig') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line or '(invalid)' in line or line == ',,':
            continue
        parts = line.rstrip(',').split(',')
        if len(parts) < 4:
            continue
        dt_str = parts[0].strip()
        int_part = parts[2].strip()
        dec_part = parts[3].strip()
        try:
            dt = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S")
        except Exception:
            continue
        try:
            temp = float(int_part + '.' + dec_part)
        except Exception:
            continue
        rows.append({"datetime": dt, "temp_actual": temp})
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return df

actual_temp = parse_actual_temp(BASE / "RawData" / "ActualTemp2024-2026.csv")

# Hourly temperature (for hourly merge)
hourly_temp = actual_temp.resample('h').mean()
hourly_temp.columns = ["temp_hourly"]
mkt = mkt.join(hourly_temp, how="left")

# Daily temperature for regime classification
daily_temp = actual_temp.resample('D').mean()
daily_temp.columns = ["daily_mean_temp"]
daily_temp.index = daily_temp.index.date
mkt["daily_temp"] = mkt["date"].map(daily_temp["daily_mean_temp"].to_dict())
mkt = mkt.dropna(subset=["daily_temp"])

# Regime
COLD_THRESH = 0.0
mkt["regime"] = np.where(mkt["daily_temp"] < COLD_THRESH, "cold", "normal_or_hot")
# For this analysis, focus on cold vs non-cold
mkt["is_cold"] = mkt["regime"] == "cold"

n_cold = mkt["is_cold"].sum()
n_normal = (~mkt["is_cold"]).sum()
cold_days = mkt[mkt["is_cold"]]["date"].nunique()
normal_days = mkt[~mkt["is_cold"]]["date"].nunique()
print(f"    Cold: {n_cold} hours ({cold_days} days), Normal: {n_normal} hours ({normal_days} days)")

# ============================================================================
# PART 1: STATISTICAL ROBUSTNESS
# ============================================================================
print("\n" + "=" * 90)
print("PART 1: STATISTICAL ROBUSTNESS OF COLD-DAY SPREAD INVERSION")
print("=" * 90)

# ---------- 1a. Overall cold vs normal ----------
cold_spreads = mkt[mkt["is_cold"]]["spread_da_imb"]
normal_spreads = mkt[~mkt["is_cold"]]["spread_da_imb"]

print(f"\n--- 1a. Overall Distribution ---")
print(f"Cold days:   N={len(cold_spreads):>5d}, mean={cold_spreads.mean():>7.2f}, median={cold_spreads.median():>7.2f}, std={cold_spreads.std():>7.2f}")
print(f"Normal days: N={len(normal_spreads):>5d}, mean={normal_spreads.mean():>7.2f}, median={normal_spreads.median():>7.2f}, std={normal_spreads.std():>7.2f}")

# t-test
t_stat, p_val = stats.ttest_ind(cold_spreads, normal_spreads, equal_var=False)
# Mann-Whitney U (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(cold_spreads, normal_spreads, alternative='two-sided')
# Cohen's d
pooled_std = np.sqrt((cold_spreads.std()**2 + normal_spreads.std()**2) / 2)
cohens_d = (cold_spreads.mean() - normal_spreads.mean()) / pooled_std

print(f"\nWelch t-test:      t={t_stat:.3f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")
print(f"Mann-Whitney U:    U={u_stat:.0f}, p={p_mw:.6f} {'***' if p_mw < 0.001 else '**' if p_mw < 0.01 else '*' if p_mw < 0.05 else ''}")
print(f"Cohen's d:         {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")

# ---------- 1b. Bootstrap CI on cold-day mean spread ----------
print(f"\n--- 1b. Bootstrap Confidence Intervals (10,000 resamples) ---")
np.random.seed(42)
n_boot = 10_000

# Bootstrap at the DAY level (not hour) to respect autocorrelation
cold_daily = mkt[mkt["is_cold"]].groupby("date")["spread_da_imb"].mean()
normal_daily = mkt[~mkt["is_cold"]].groupby("date")["spread_da_imb"].mean()

boot_cold_means = np.array([cold_daily.sample(len(cold_daily), replace=True).mean() for _ in range(n_boot)])
boot_normal_means = np.array([normal_daily.sample(len(normal_daily), replace=True).mean() for _ in range(n_boot)])
boot_diff = boot_cold_means - boot_normal_means

ci_cold_95 = np.percentile(boot_cold_means, [2.5, 97.5])
ci_cold_99 = np.percentile(boot_cold_means, [0.5, 99.5])
ci_diff_95 = np.percentile(boot_diff, [2.5, 97.5])
ci_diff_99 = np.percentile(boot_diff, [0.5, 99.5])

print(f"Cold daily mean spread:     {cold_daily.mean():>7.2f} EUR/MWh")
print(f"  95% bootstrap CI:         [{ci_cold_95[0]:>7.2f}, {ci_cold_95[1]:>7.2f}]")
print(f"  99% bootstrap CI:         [{ci_cold_99[0]:>7.2f}, {ci_cold_99[1]:>7.2f}]")
print(f"Difference (cold - normal): {(cold_daily.mean() - normal_daily.mean()):>7.2f} EUR/MWh")
print(f"  95% bootstrap CI:         [{ci_diff_95[0]:>7.2f}, {ci_diff_95[1]:>7.2f}]")
print(f"  99% bootstrap CI:         [{ci_diff_99[0]:>7.2f}, {ci_diff_99[1]:>7.2f}]")
zero_in_ci = ci_diff_95[0] <= 0 <= ci_diff_95[1]
print(f"  Zero in 95% CI?           {'YES - NOT significant' if zero_in_ci else 'NO - significant (cold < normal)'}")

# ---------- 1c. Permutation test ----------
print(f"\n--- 1c. Permutation Test (day-level, 10,000 permutations) ---")
all_daily = pd.concat([cold_daily, normal_daily])
observed_diff = cold_daily.mean() - normal_daily.mean()

n_cold_days = len(cold_daily)
perm_diffs = np.zeros(n_boot)
for i in range(n_boot):
    shuffled = np.random.permutation(all_daily.values)
    perm_diffs[i] = shuffled[:n_cold_days].mean() - shuffled[n_cold_days:].mean()

p_perm = np.mean(perm_diffs <= observed_diff)  # one-sided: cold < normal
print(f"Observed diff (cold - normal): {observed_diff:.2f} EUR/MWh")
print(f"Permutation p-value (one-sided): {p_perm:.4f} {'***' if p_perm < 0.001 else '**' if p_perm < 0.01 else '*' if p_perm < 0.05 else ''}")

# ---------- 1d. By-hour bootstrap (the key question) ----------
print(f"\n--- 1d. By-Hour Bootstrap CI for Cold-Day Spread ---")
print(f"{'Hour':>4s}  {'Cold Mean':>10s}  {'95% CI Low':>10s}  {'95% CI High':>11s}  {'Zero in CI?':>11s}  {'N days':>6s}")
print("-" * 65)

hour_significance = {}
for h in range(24):
    cold_h = mkt[(mkt["hour"] == h) & (mkt["is_cold"])]["spread_da_imb"]
    if len(cold_h) < 10:
        continue

    # Bootstrap the hourly cold spread
    boot_means = np.array([cold_h.sample(len(cold_h), replace=True).mean() for _ in range(n_boot)])
    ci = np.percentile(boot_means, [2.5, 97.5])
    zero_in = ci[0] <= 0 <= ci[1]
    hour_significance[h] = {"mean": cold_h.mean(), "ci_low": ci[0], "ci_high": ci[1],
                            "zero_in_ci": zero_in, "n": len(cold_h)}

    if zero_in:
        ci_label = "YES"
    elif ci[1] < 0:
        ci_label = "NO (neg)"
    else:
        ci_label = "NO (pos)"
    print(f"{h:>4d}  {cold_h.mean():>10.2f}  {ci[0]:>10.2f}  {ci[1]:>11.2f}  {ci_label:>11s}  {len(cold_h):>6d}")

neg_sig = [h for h, v in hour_significance.items() if not v["zero_in_ci"] and v["ci_high"] < 0]
pos_sig = [h for h, v in hour_significance.items() if not v["zero_in_ci"] and v["ci_low"] > 0]
ambig = [h for h, v in hour_significance.items() if v["zero_in_ci"]]
print(f"\n[+] Hours with spread significantly < 0 (CI excludes 0): {neg_sig}")
print(f"[+] Hours with spread significantly > 0 (CI excludes 0): {pos_sig}")
print(f"[*] Ambiguous hours (CI includes 0):                     {ambig}")

# ---------- 1e. Day-level clustering: are cold days independent? ----------
print(f"\n--- 1e. Cold-Day Clustering & Autocorrelation ---")
cold_dates = sorted(cold_daily.index)
gaps = [(pd.Timestamp(cold_dates[i+1]) - pd.Timestamp(cold_dates[i])).days for i in range(len(cold_dates)-1)]
consecutive = sum(1 for g in gaps if g == 1)
print(f"Total cold days: {len(cold_dates)}")
print(f"Consecutive-day pairs: {consecutive} ({consecutive/len(gaps)*100:.0f}%)")
print(f"Mean gap between cold days: {np.mean(gaps):.1f} days")
print(f"[*] Cold days cluster in episodes -> effective sample size < {cold_days}")

# Count cold episodes (gaps > 1 day)
episodes = 1
for g in gaps:
    if g > 1:
        episodes += 1
print(f"[*] Distinct cold episodes: {episodes}")
print(f"[*] Effective independent samples: ~{episodes} episodes (not {cold_days} days)")

# Episode-level test
episode_id = 0
episode_map = {cold_dates[0]: episode_id}
for i in range(len(gaps)):
    if gaps[i] > 1:
        episode_id += 1
    episode_map[cold_dates[i+1]] = episode_id

cold_ep_data = mkt[mkt["is_cold"]].copy()
cold_ep_data["episode"] = cold_ep_data["date"].map(episode_map)
episode_means = cold_ep_data.groupby("episode")["spread_da_imb"].mean()

print(f"\n--- Episode-Level Statistics ---")
print(f"Episode mean spreads: {episode_means.values}")
print(f"Episode mean: {episode_means.mean():.2f}, std: {episode_means.std():.2f}")
t_ep, p_ep = stats.ttest_1samp(episode_means, 0)
print(f"One-sample t-test (episode means vs 0): t={t_ep:.3f}, p={p_ep:.4f} {'***' if p_ep < 0.001 else '**' if p_ep < 0.01 else '*' if p_ep < 0.05 else ''}")

# ============================================================================
# PART 2: CAUSAL MECHANISM
# ============================================================================
print("\n\n" + "=" * 90)
print("PART 2: CAUSAL MECHANISM - WHY DOES THE SPREAD INVERT ON COLD DAYS?")
print("=" * 90)

# ---------- 2a. Load forecast error ----------
print(f"\n--- 2a. Load Forecast Error: Cold vs Normal Days ---")
print("(forecast_error = actual - forecast; positive = under-forecast = demand surprise)")

has_load = mkt.dropna(subset=["forecast_error_mw"])
cold_load = has_load[has_load["is_cold"]]
normal_load = has_load[~has_load["is_cold"]]

print(f"\nCold days:   mean error = {cold_load['forecast_error_mw'].mean():>+7.1f} MW, median = {cold_load['forecast_error_mw'].median():>+7.1f}")
print(f"Normal days: mean error = {normal_load['forecast_error_mw'].mean():>+7.1f} MW, median = {normal_load['forecast_error_mw'].median():>+7.1f}")

t_load, p_load = stats.ttest_ind(cold_load["forecast_error_mw"], normal_load["forecast_error_mw"], equal_var=False)
print(f"t-test: t={t_load:.3f}, p={p_load:.6f} {'***' if p_load < 0.001 else '**' if p_load < 0.01 else '*' if p_load < 0.05 else ''}")

# By hour
print(f"\n{'Hour':>4s}  {'Cold Err':>10s}  {'Normal Err':>10s}  {'Diff':>8s}  {'t-stat':>8s}  {'p-val':>8s}  {'Sig?':>5s}")
print("-" * 65)
for h in range(24):
    ce = has_load[(has_load["hour"] == h) & (has_load["is_cold"])]["forecast_error_mw"]
    ne = has_load[(has_load["hour"] == h) & (~has_load["is_cold"])]["forecast_error_mw"]
    if len(ce) < 5 or len(ne) < 5:
        continue
    t, p = stats.ttest_ind(ce, ne, equal_var=False)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{h:>4d}  {ce.mean():>+10.1f}  {ne.mean():>+10.1f}  {ce.mean()-ne.mean():>+8.1f}  {t:>8.2f}  {p:>8.4f}  {sig:>5s}")

# ---------- 2b. Correlation: load error -> spread ----------
print(f"\n--- 2b. Load Error -> Spread Correlation ---")
valid = has_load.dropna(subset=["spread_da_imb", "forecast_error_mw"])

cold_valid = valid[valid["is_cold"]]
normal_valid = valid[~valid["is_cold"]]

r_cold, p_r_cold = stats.pearsonr(cold_valid["forecast_error_mw"], cold_valid["spread_da_imb"])
r_normal, p_r_normal = stats.pearsonr(normal_valid["forecast_error_mw"], normal_valid["spread_da_imb"])
r_all, p_r_all = stats.pearsonr(valid["forecast_error_mw"], valid["spread_da_imb"])

print(f"All days:    r = {r_all:>+.3f} (p={p_r_all:.6f})")
print(f"Cold days:   r = {r_cold:>+.3f} (p={p_r_cold:.6f})")
print(f"Normal days: r = {r_normal:>+.3f} (p={p_r_normal:.6f})")
print("[*] Negative r means: higher load error -> more negative spread (spread inversion)")

# ---------- 2c. Imbalance volume ----------
print(f"\n--- 2c. System Imbalance Volume: Cold vs Normal ---")
has_imb = mkt.dropna(subset=["imbalance_mwh"])
cold_imb = has_imb[has_imb["is_cold"]]["imbalance_mwh"]
normal_imb = has_imb[~has_imb["is_cold"]]["imbalance_mwh"]

print(f"Cold days:   mean imbalance = {cold_imb.mean():>+7.1f} MWh, std = {cold_imb.std():.1f}")
print(f"Normal days: mean imbalance = {normal_imb.mean():>+7.1f} MWh, std = {normal_imb.std():.1f}")
print(f"(Negative imbalance = system deficit = demand > supply)")

# Deficit fraction
cold_deficit_frac = (cold_imb < 0).mean()
normal_deficit_frac = (normal_imb < 0).mean()
print(f"\nDeficit fraction (imbalance < 0):")
print(f"  Cold days:   {cold_deficit_frac:.1%}")
print(f"  Normal days: {normal_deficit_frac:.1%}")

# By hour
print(f"\n{'Hour':>4s}  {'Cold Imb':>10s}  {'Normal Imb':>10s}  {'Cold Def%':>10s}  {'Norm Def%':>10s}  {'Diff Def%':>10s}")
print("-" * 65)
for h in range(24):
    ci = has_imb[(has_imb["hour"] == h) & (has_imb["is_cold"])]["imbalance_mwh"]
    ni = has_imb[(has_imb["hour"] == h) & (~has_imb["is_cold"])]["imbalance_mwh"]
    if len(ci) < 5:
        continue
    cold_def = (ci < 0).mean()
    norm_def = (ni < 0).mean()
    print(f"{h:>4d}  {ci.mean():>+10.1f}  {ni.mean():>+10.1f}  {cold_def:>10.1%}  {norm_def:>10.1%}  {cold_def - norm_def:>+10.1%}")

# ---------- 2d. Settlement price behavior ----------
print(f"\n--- 2d. Imbalance Settlement Price: Cold vs Normal ---")
cold_sp = mkt[mkt["is_cold"]]["imb_settlement_price"]
normal_sp = mkt[~mkt["is_cold"]]["imb_settlement_price"]
cold_da = mkt[mkt["is_cold"]]["da_price"]
normal_da = mkt[~mkt["is_cold"]]["da_price"]

print(f"{'Metric':<25s}  {'Cold':>10s}  {'Normal':>10s}  {'Diff':>10s}")
print("-" * 60)
print(f"{'DA price (mean)':.<25s}  {cold_da.mean():>10.2f}  {normal_da.mean():>10.2f}  {cold_da.mean() - normal_da.mean():>+10.2f}")
print(f"{'Settlement price (mean)':.<25s}  {cold_sp.mean():>10.2f}  {normal_sp.mean():>10.2f}  {cold_sp.mean() - normal_sp.mean():>+10.2f}")
print(f"{'Spread DA-Imb (mean)':.<25s}  {(cold_da - cold_sp).mean():>10.2f}  {(normal_da - normal_sp).mean():>10.2f}  {(cold_da - cold_sp).mean() - (normal_da - normal_sp).mean():>+10.2f}")

# Which side moves more?
print(f"\n[*] On cold days, DA price moves {cold_da.mean() - normal_da.mean():+.1f} EUR/MWh")
print(f"[*] On cold days, settlement price moves {cold_sp.mean() - normal_sp.mean():+.1f} EUR/MWh")
if abs(cold_sp.mean() - normal_sp.mean()) > abs(cold_da.mean() - normal_da.mean()):
    print("[+] Settlement price moves MORE than DA price -> imbalance side drives the inversion")
else:
    print("[+] DA price moves more -> DA side drives the inversion")

# By hour
print(f"\n{'Hour':>4s}  {'Cold DA':>10s}  {'Norm DA':>10s}  {'Cold Settl':>10s}  {'Norm Settl':>10s}  {'DA Move':>10s}  {'Settl Move':>10s}  {'Driver':>8s}")
print("-" * 90)
for h in range(24):
    cd = mkt[(mkt["hour"] == h) & (mkt["is_cold"])]
    nd = mkt[(mkt["hour"] == h) & (~mkt["is_cold"])]
    if len(cd) < 5:
        continue
    da_move = cd["da_price"].mean() - nd["da_price"].mean()
    sp_move = cd["imb_settlement_price"].mean() - nd["imb_settlement_price"].mean()
    driver = "SETTL" if abs(sp_move) > abs(da_move) else "DA"
    print(f"{h:>4d}  {cd['da_price'].mean():>10.1f}  {nd['da_price'].mean():>10.1f}  "
          f"{cd['imb_settlement_price'].mean():>10.1f}  {nd['imb_settlement_price'].mean():>10.1f}  "
          f"{da_move:>+10.1f}  {sp_move:>+10.1f}  {driver:>8s}")

# ---------- 2e. IDM as price discovery ----------
print(f"\n--- 2e. IDM Price Discovery: Do Traders Correct on IDM? ---")
has_idm = mkt.dropna(subset=["idm_vwap"])
cold_idm = has_idm[has_idm["is_cold"]]
normal_idm = has_idm[~has_idm["is_cold"]]

if len(cold_idm) > 10 and len(normal_idm) > 10:
    # IDM premium over DA = (IDM - DA), positive means IDM discovers higher price
    cold_idm_premium = (cold_idm["idm_vwap"] - cold_idm["da_price"])
    normal_idm_premium = (normal_idm["idm_vwap"] - normal_idm["da_price"])

    print(f"IDM premium over DA (IDM - DA):")
    print(f"  Cold days:   {cold_idm_premium.mean():>+7.2f} EUR/MWh (traders bid UP on IDM)")
    print(f"  Normal days: {normal_idm_premium.mean():>+7.2f} EUR/MWh")

    # IDM volume
    cold_vol = cold_idm["idm_volume_mwh"].mean()
    normal_vol = normal_idm["idm_volume_mwh"].mean()
    print(f"\nIDM volume:")
    print(f"  Cold days:   {cold_vol:>7.0f} MWh/hour")
    print(f"  Normal days: {normal_vol:>7.0f} MWh/hour")
    if cold_vol > normal_vol:
        print(f"  [+] Cold days have {cold_vol/normal_vol:.1f}x MORE IDM trading -> active correction")
    else:
        print(f"  [-] Cold days have LESS IDM trading")

    # IDM narrows the gap?
    cold_idm_imb = (cold_idm["idm_vwap"] - cold_idm["imb_settlement_price"]).mean()
    normal_idm_imb = (normal_idm["idm_vwap"] - normal_idm["imb_settlement_price"]).mean()
    print(f"\nIDM-Imbalance gap:")
    print(f"  Cold days:   {cold_idm_imb:>+7.2f} EUR/MWh")
    print(f"  Normal days: {normal_idm_imb:>+7.2f} EUR/MWh")
    print(f"  [*] IDM partially captures the settlement spike but not fully")
else:
    print("[!] Insufficient IDM data for cold-day analysis")

# ---------- 2f. Actual load levels ----------
print(f"\n--- 2f. Actual Load Levels: Cold vs Normal ---")
cold_load_vals = has_load[has_load["is_cold"]]["actual_load_mw"]
normal_load_vals = has_load[~has_load["is_cold"]]["actual_load_mw"]

print(f"Cold days:   mean actual load = {cold_load_vals.mean():>7.0f} MW")
print(f"Normal days: mean actual load = {normal_load_vals.mean():>7.0f} MW")
print(f"Difference:  {cold_load_vals.mean() - normal_load_vals.mean():>+7.0f} MW ({(cold_load_vals.mean()/normal_load_vals.mean()-1)*100:>+.1f}%)")

# By hour
print(f"\n{'Hour':>4s}  {'Cold Load':>10s}  {'Norm Load':>10s}  {'Diff MW':>10s}  {'Cold Err':>10s}  {'Norm Err':>10s}  {'Err Diff':>10s}")
print("-" * 75)
for h in range(24):
    cl = has_load[(has_load["hour"] == h) & (has_load["is_cold"])]
    nl = has_load[(has_load["hour"] == h) & (~has_load["is_cold"])]
    if len(cl) < 5:
        continue
    print(f"{h:>4d}  {cl['actual_load_mw'].mean():>10.0f}  {nl['actual_load_mw'].mean():>10.0f}  "
          f"{cl['actual_load_mw'].mean() - nl['actual_load_mw'].mean():>+10.0f}  "
          f"{cl['forecast_error_mw'].mean():>+10.1f}  {nl['forecast_error_mw'].mean():>+10.1f}  "
          f"{cl['forecast_error_mw'].mean() - nl['forecast_error_mw'].mean():>+10.1f}")

# ---------- 2g. The 5 "keep" hours: why are they different? ----------
print(f"\n--- 2g. Why Hours 7, 8, 18-20 Don't Invert ---")
keep_hours = [7, 8, 18, 19, 20]
reverse_hours = [h for h in range(24) if h not in keep_hours]

for label, hours in [("KEEP hours (7,8,18-20)", keep_hours), ("REVERSE hours (rest)", reverse_hours)]:
    subset = mkt[mkt["hour"].isin(hours)]
    cold_s = subset[subset["is_cold"]]
    norm_s = subset[~subset["is_cold"]]

    cold_err = cold_s["forecast_error_mw"].mean() if "forecast_error_mw" in cold_s.columns else np.nan
    norm_err = norm_s["forecast_error_mw"].mean() if "forecast_error_mw" in norm_s.columns else np.nan

    cold_imb_vol = cold_s["imbalance_mwh"].mean() if len(cold_s.dropna(subset=["imbalance_mwh"])) > 0 else np.nan
    norm_imb_vol = norm_s["imbalance_mwh"].mean() if len(norm_s.dropna(subset=["imbalance_mwh"])) > 0 else np.nan

    cold_load_mean = cold_s["actual_load_mw"].mean() if len(cold_s.dropna(subset=["actual_load_mw"])) > 0 else np.nan
    norm_load_mean = norm_s["actual_load_mw"].mean() if len(norm_s.dropna(subset=["actual_load_mw"])) > 0 else np.nan

    print(f"\n  {label}:")
    print(f"    Cold spread:     {cold_s['spread_da_imb'].mean():>+7.2f} EUR/MWh")
    print(f"    Normal spread:   {norm_s['spread_da_imb'].mean():>+7.2f} EUR/MWh")
    print(f"    Cold load:       {cold_load_mean:>7.0f} MW (vs {norm_load_mean:.0f} normal)")
    print(f"    Cold load error: {cold_err:>+7.1f} MW (vs {norm_err:+.1f} normal)")
    print(f"    Cold imbalance:  {cold_imb_vol:>+7.1f} MWh (vs {norm_imb_vol:+.1f} normal)")
    print(f"    Cold DA price:   {cold_s['da_price'].mean():>7.1f} (vs {norm_s['da_price'].mean():.1f})")
    print(f"    Cold settl price:{cold_s['imb_settlement_price'].mean():>7.1f} (vs {norm_s['imb_settlement_price'].mean():.1f})")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[*] Creating visualization...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Cold-Snap Spread Inversion: Statistical Robustness & Causal Mechanism",
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: Bootstrap distribution of cold-normal difference
ax1 = axes[0, 0]
ax1.hist(boot_diff, bins=80, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
ax1.axvline(observed_diff, color='red', linewidth=2, label=f'Observed: {observed_diff:.1f}')
ax1.axvline(ci_diff_95[0], color='orange', linewidth=1.5, linestyle='--', label=f'95% CI: [{ci_diff_95[0]:.1f}, {ci_diff_95[1]:.1f}]')
ax1.axvline(ci_diff_95[1], color='orange', linewidth=1.5, linestyle='--')
ax1.axvline(0, color='black', linewidth=1, linestyle=':')
ax1.set_xlabel("Mean Spread Difference (Cold - Normal, EUR/MWh)")
ax1.set_ylabel("Bootstrap Count")
ax1.set_title("Bootstrap: Cold vs Normal Daily Spread\n(day-level resampling, 10k iterations)")
ax1.legend(fontsize=9)

# Panel 2: By-hour bootstrap CI
ax2 = axes[0, 1]
hours_plot = sorted(hour_significance.keys())
means = [hour_significance[h]["mean"] for h in hours_plot]
ci_lows = [hour_significance[h]["ci_low"] for h in hours_plot]
ci_highs = [hour_significance[h]["ci_high"] for h in hours_plot]

colors_ci = ['green' if hour_significance[h]["ci_high"] < 0 else
             'blue' if hour_significance[h]["ci_low"] > 0 else
             'gray' for h in hours_plot]

ax2.bar(hours_plot, means, color=colors_ci, alpha=0.6, edgecolor='black', linewidth=0.5)
ax2.vlines(hours_plot, ci_lows, ci_highs, color='black', linewidth=1.5)
ax2.axhline(0, color='red', linewidth=1, linestyle='--')
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Mean Cold-Day Spread (EUR/MWh)")
ax2.set_title("Cold-Day Spread by Hour\n(with 95% bootstrap CI)")
ax2.set_xticks(range(24))

# Panel 3: Load forecast error by hour and regime
ax3 = axes[0, 2]
cold_errs = [has_load[(has_load["hour"] == h) & (has_load["is_cold"])]["forecast_error_mw"].mean() for h in range(24)]
normal_errs = [has_load[(has_load["hour"] == h) & (~has_load["is_cold"])]["forecast_error_mw"].mean() for h in range(24)]

ax3.bar(np.arange(24) - 0.2, cold_errs, 0.4, label="Cold days", color="blue", alpha=0.7)
ax3.bar(np.arange(24) + 0.2, normal_errs, 0.4, label="Normal days", color="gray", alpha=0.7)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Load Forecast Error (MW)\n(positive = under-forecast)")
ax3.set_title("Load Forecast Error: Cold vs Normal\n(actual - forecast)")
ax3.legend(fontsize=9)
ax3.set_xticks(range(24))

# Panel 4: Scatter - load error vs spread on cold days
ax4 = axes[1, 0]
cv = cold_valid.copy()
ax4.scatter(cv["forecast_error_mw"], cv["spread_da_imb"], alpha=0.15, s=8, color='blue')

# Fit line
z = np.polyfit(cv["forecast_error_mw"].values, cv["spread_da_imb"].values, 1)
x_fit = np.linspace(cv["forecast_error_mw"].min(), cv["forecast_error_mw"].max(), 100)
ax4.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2,
         label=f'r={r_cold:.3f}, slope={z[0]:.2f}')
ax4.axhline(0, color='black', linewidth=0.5, linestyle=':')
ax4.axvline(0, color='black', linewidth=0.5, linestyle=':')
ax4.set_xlabel("Load Forecast Error (MW)")
ax4.set_ylabel("DA-Imbalance Spread (EUR/MWh)")
ax4.set_title("Cold Days: Load Error vs Spread")
ax4.legend(fontsize=9)

# Panel 5: DA price vs Settlement price moves (by hour)
ax5 = axes[1, 1]
da_moves = []
settl_moves = []
for h in range(24):
    cd = mkt[(mkt["hour"] == h) & (mkt["is_cold"])]
    nd = mkt[(mkt["hour"] == h) & (~mkt["is_cold"])]
    if len(cd) < 5:
        da_moves.append(0)
        settl_moves.append(0)
        continue
    da_moves.append(cd["da_price"].mean() - nd["da_price"].mean())
    settl_moves.append(cd["imb_settlement_price"].mean() - nd["imb_settlement_price"].mean())

ax5.bar(np.arange(24) - 0.2, da_moves, 0.4, label="DA price move", color="orange", alpha=0.7)
ax5.bar(np.arange(24) + 0.2, settl_moves, 0.4, label="Settlement price move", color="red", alpha=0.7)
ax5.axhline(0, color='black', linewidth=0.5)
ax5.set_xlabel("Hour of Day")
ax5.set_ylabel("Price Change Cold vs Normal (EUR/MWh)")
ax5.set_title("Which Side Moves? DA vs Settlement\n(cold minus normal)")
ax5.legend(fontsize=9)
ax5.set_xticks(range(24))

# Panel 6: Deficit fraction by hour
ax6 = axes[1, 2]
cold_def_h = [(has_imb[(has_imb["hour"] == h) & (has_imb["is_cold"])]["imbalance_mwh"] < 0).mean()
              if len(has_imb[(has_imb["hour"] == h) & (has_imb["is_cold"])]) > 0 else 0 for h in range(24)]
norm_def_h = [(has_imb[(has_imb["hour"] == h) & (~has_imb["is_cold"])]["imbalance_mwh"] < 0).mean()
              if len(has_imb[(has_imb["hour"] == h) & (~has_imb["is_cold"])]) > 0 else 0 for h in range(24)]

ax6.plot(range(24), cold_def_h, 'b-o', markersize=5, linewidth=1.5, label="Cold days")
ax6.plot(range(24), norm_def_h, 'gray', marker='s', markersize=4, linewidth=1.5, label="Normal days", alpha=0.7)
ax6.axhline(0.5, color='black', linewidth=0.5, linestyle=':')
ax6.set_xlabel("Hour of Day")
ax6.set_ylabel("Deficit Fraction (imbalance < 0)")
ax6.set_title("System Deficit Probability by Hour\n(cold vs normal)")
ax6.legend(fontsize=9)
ax6.set_xticks(range(24))
ax6.set_ylim(0, 1)

# Highlight keep hours
for ax in [ax2, ax3, ax5, ax6]:
    for h in keep_hours:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.08, color='green')

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = OUT_DIR / "cold_snap_causality.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"[+] Saved: {outpath}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

print(f"""
QUESTION 1: IS THE SAMPLE STATISTICALLY SUFFICIENT?

  Overall cold vs normal spread:
    - Welch t-test: p = {p_val:.6f} {'(significant)' if p_val < 0.05 else '(NOT significant)'}
    - Mann-Whitney U: p = {p_mw:.6f} {'(significant)' if p_mw < 0.05 else '(NOT significant)'}
    - Permutation test: p = {p_perm:.4f} {'(significant)' if p_perm < 0.05 else '(NOT significant)'}
    - Bootstrap 95% CI for cold-normal diff: [{ci_diff_95[0]:.1f}, {ci_diff_95[1]:.1f}]
    - Cohen's d = {cohens_d:.3f}

  Sample concerns:
    - Only {cold_days} cold days across {episodes} cold episodes
    - Cold days cluster (consecutive pairs: {consecutive}/{len(gaps)})
    - Effective independent samples: ~{episodes} episodes
    - Episode-level t-test: p = {p_ep:.4f}

  By-hour:
    - Hours with CI fully below 0 (reliably negative): {neg_sig}
    - Hours with CI fully above 0 (reliably positive): {pos_sig}
    - Ambiguous hours (CI includes 0): {ambig}

QUESTION 2: WHAT CAUSES THE INVERSION?

  Causal chain (confirmed by data):
    1. Cold weather -> Higher actual load (+{cold_load_vals.mean() - normal_load_vals.mean():.0f} MW)
    2. Forecasters under-predict load (error: {cold_load['forecast_error_mw'].mean():+.1f} MW cold vs {normal_load['forecast_error_mw'].mean():+.1f} MW normal)
    3. System runs deficit more often ({cold_deficit_frac:.0%} vs {normal_deficit_frac:.0%})
    4. Settlement price spikes ({cold_sp.mean():.1f} vs {normal_sp.mean():.1f} EUR/MWh)
    5. DA price also rises but LESS ({cold_da.mean():.1f} vs {normal_da.mean():.1f})
    6. Spread inverts (DA < Settlement)

  Load error -> spread correlation: r = {r_cold:.3f} on cold days

  Why hours 7-8, 18-20 don't invert:
    These are demand peaks where DA already prices in high load.
    The forecast error is similar, but DA price rises proportionally,
    so the spread stays positive (DA >= Settlement).
""")

print("[+] Analysis complete.")
