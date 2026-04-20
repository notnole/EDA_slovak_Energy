"""
Quantile Spread Model — Experiment
===================================

Parallel variant of train_stacked_model.py that fits q10/q50/q90 heads
instead of only the q50 point estimate. Interval width (q90-q10) is used
as a conviction signal.

Three backtest variants compared against the point-MAE baseline:
  BASELINE: |q50| >= threshold, size = |q50|.clip(<=5)          [= current prod]
  EXCLUDE : q10 > threshold (deficit) OR q90 < -threshold (surplus),
            size = |q50|.clip(<=5)                              [filter by conviction]
  WIDTH   : |q50| >= threshold, size = base * clip(w_med/w, 0.3, 1.5)
            where w = q90-q10 per sample, w_med = train-fold median  [scale by conviction]

Everything else (folds, features, LGB params, execution) is identical
to train_stacked_model.py so the comparison is clean.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml
from train_stacked_model import SELECTED_FEATURES, LGB_PARAMS, LEAD

# Regime-aware evaluation: IDM/imbalance multiplier changed on 2025-09-01.
# Pre-Sep 2025 P&L is under old regime (multiplier 1.5) and not directly
# comparable to current production. Evaluate walk-forward monthly from
# 2025-09 onwards — each fold trains on ALL data before its pred_start.
# Every fold is a test fold in this design (no warmup-only folds).
FOLDS = [
    # (train_end,   pred_start,   pred_end)
    ('2025-09-01', '2025-09-01', '2025-10-01'),   # Sep 2025 (first post-regime month)
    ('2025-10-01', '2025-10-01', '2025-11-01'),   # Oct 2025
    ('2025-11-01', '2025-11-01', '2025-12-01'),   # Nov 2025
    ('2025-12-01', '2025-12-01', '2026-01-01'),   # Dec 2025
    ('2026-01-01', '2026-01-01', '2026-02-01'),   # Jan 2026
    ('2026-02-01', '2026-02-01', '2026-03-01'),   # Feb 2026
    ('2026-03-01', '2026-03-01', '2026-04-10'),   # Mar-Apr 2026
]

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent

QUANTILES = [0.30, 0.50, 0.70]
Q_COLS = {0.30: 'q10', 0.50: 'q50', 0.70: 'q90'}  # names kept for downstream compat

VECTORD_PATH = BASE_DIR / "data" / "features" / "vectord_features.csv"

# Signal categories with their leakage shifts (in 15-min periods):
# - Gen mix: ENTSO-E actuals with publication delay → shift(LEAD+4)
# - D-1 forecasts: known day-ahead → no shift
# - PICASSO: 15-min period ends at T+15min, published T+30min. At trade time
#   (T - LEAD*15min), we need period_end + 30min <= trade_time, i.e. period
#   must have started >= 165min before delivery T → shift(LEAD+3).
GEN_MIX_COLS = [
    'natgas_mw', 'hardcoal_mw', 'lignite_mw', 'nuclear_mw',
    'fosiloil_mw', 'hydroreservoir_mw', 'hydrorunriver_mw', 'hydropump_mw',
    'hydropump_cvtg2_mw', 'hydropump_cvtg3_mw', 'hydropump_cvtg4_mw',
    'hydropump_cvtg5_mw', 'hydropump_cvtg6_mw',
]
DA_FORECAST_COLS = [
    'resl_mw', 'resl_isr_mw', 'de_spot',
    'sk_spot_m1', 'sk_spot_m2', 'sk_spot_merged',
    'cons_gfs', 'cons_icon', 'cons_ecmf', 'cons_seps', 'solar_fcst',
    'temp_gfs', 'temp_ecm', 'temp_icon',
    'cloud_ec', 'cloud_icon',
]
PICASSO_RAW_COLS = ['picasso_pos', 'picasso_neg']
PICASSO_DERIVED_COLS = [
    'picasso_pos_last', 'picasso_neg_last',
    'picasso_pos_mean4h', 'picasso_neg_mean4h',
    'picasso_pos_std4h', 'picasso_neg_std4h',
    'picasso_pos_nan1h', 'picasso_neg_nan1h',
]
# Regulatory regime: IDM-to-imbalance multiplier changed on 2025-09-01 from 1.5 to 1.1
REG_MULTIPLIER_CHANGE_DATE = '2025-09-01'
REG_MULTIPLIER_BEFORE = 1.5
REG_MULTIPLIER_AFTER = 1.1
REGIME_COLS = ['reg_multiplier']

VECTORD_FEATURES = GEN_MIX_COLS + DA_FORECAST_COLS + PICASSO_DERIVED_COLS + REGIME_COLS


def main():
    print("=" * 70)
    print("QUANTILE SPREAD MODEL — q10/q50/q90")
    print("=" * 70)

    data = load_all_data()

    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    df_base = df_base.join(ob_120, how='left')
    df_base['imb_settlement_price'] = df_base['imb_settle_price']
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    # Join vectord experimental features with leakage-safe shifts
    vec = pd.read_csv(VECTORD_PATH, parse_dates=[0], index_col=0)
    if vec.index.tz is not None:
        vec.index = vec.index.tz_convert('UTC').tz_localize(None)

    # Reindex vectord onto df_base's index (may introduce NaN where vec is missing)
    vec = vec.reindex(df_base.index)

    # 1. Gen mix: ENTSO-E actuals → shift(LEAD+4)
    for c in GEN_MIX_COLS:
        if c in vec.columns:
            df_base[c] = vec[c].shift(LEAD + 4)

    # 2. D-1 forecasts: known day-ahead → no shift
    for c in DA_FORECAST_COLS:
        if c in vec.columns:
            df_base[c] = vec[c]

    # 3. PICASSO: compute rolling features on RAW (pre-shift), then shift(LEAD+2).
    # NaN is meaningful (one-sided activation); use min_periods=1 so rolling
    # degrades gracefully. NaN count features directly encode the missingness.
    for side in ['pos', 'neg']:
        raw_col = f'picasso_{side}'
        if raw_col not in vec.columns:
            continue
        raw = vec[raw_col]
        # Rolling features (windows in 15-min periods)
        mean4h = raw.rolling(16, min_periods=1).mean()
        std4h = raw.rolling(16, min_periods=2).std()
        nan1h = raw.isna().rolling(4, min_periods=1).sum()  # 0-4 NaN count over last 1h
        # Apply LEAD+3 shift for 30min publication-to-trade buffer (see note above)
        shift = LEAD + 3
        df_base[f'picasso_{side}_last'] = raw.shift(shift)
        df_base[f'picasso_{side}_mean4h'] = mean4h.shift(shift)
        df_base[f'picasso_{side}_std4h'] = std4h.shift(shift)
        df_base[f'picasso_{side}_nan1h'] = nan1h.shift(shift)

    # 4. Regulatory regime: IDM/imbalance multiplier changed 2025-09-01 (1.5 -> 1.1)
    # Known-in-advance categorical signal; no shift needed.
    regime_cutoff = pd.Timestamp(REG_MULTIPLIER_CHANGE_DATE)
    df_base['reg_multiplier'] = np.where(
        df_base.index < regime_cutoff, REG_MULTIPLIER_BEFORE, REG_MULTIPLIER_AFTER
    )

    # Coverage report
    print("[+] Vectord + regime feature coverage in df_base:")
    for c in VECTORD_FEATURES:
        if c in df_base.columns:
            cov = df_base[c].notna().mean()
            print(f"    {c:<25s} {cov:.1%}")
        else:
            print(f"    {c:<25s} MISSING (skipping)")
    # Drop any vectord features not actually present (e.g. biomas_mw had no data)
    vectord_present = [c for c in VECTORD_FEATURES if c in df_base.columns]

    spread_features = [f for f in SELECTED_FEATURES if f in feature_cols] + vectord_present
    missing = [f for f in SELECTED_FEATURES if f not in feature_cols]
    if missing:
        print(f"[!] Warning: {len(missing)} selected features not found: {missing}")
    print(f"[+] Features: {len(spread_features)} ({len(SELECTED_FEATURES)} base + {len(vectord_present)} vectord), {len(df_base)} rows total")

    all_oof = []
    train_widths = []  # track train median interval width for width-sizing

    for fi, (train_end, pred_start, pred_end) in enumerate(FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fi+1}/{len(FOLDS)}: train < {train_end}, predict [{pred_start}, {pred_end})")
        print(f"{'='*60}")

        df = df_base.copy()
        train_mask = df.index < train_end
        pred_mask = (df.index >= pred_start) & (df.index < pred_end)
        train = df[train_mask].dropna(subset=['target', f'proxy_lag{LEAD+1}'])
        pred_data = df[pred_mask].dropna(subset=[f'proxy_lag{LEAD+1}']).copy()

        if len(train) == 0 or len(pred_data) == 0:
            print(f"  [!] Insufficient data, skipping")
            continue

        sp_train = train.dropna(subset=['spread_target'])
        sp_train = sp_train[sp_train['imb_settlement_price'].abs() <= 5000]
        print(f"  Train: {len(sp_train)}, Predict: {len(pred_data)}")

        X_train = sp_train[spread_features].values
        y_train = sp_train['spread_target'].values
        X_pred = pred_data[spread_features].values

        # Fit each quantile head
        q_train_preds = {}
        q50_model = None
        for alpha in QUANTILES:
            m = lgb.LGBMRegressor(objective='quantile', alpha=alpha, **LGB_PARAMS)
            m.fit(X_train, y_train)
            pred_data[Q_COLS[alpha]] = m.predict(X_pred)
            q_train_preds[alpha] = m.predict(X_train)
            if alpha == 0.50:
                q50_model = m

        # Print feature importance on last fold (q50 model)
        if fi == len(FOLDS) - 1 and q50_model is not None:
            imp = pd.DataFrame({'feature': spread_features,
                                'importance': q50_model.feature_importances_})
            imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
            imp = imp.sort_values('pct', ascending=False)
            print(f"\n    Top 20 features (q50, last fold):")
            for _, row in imp.head(20).iterrows():
                marker = "  <-- vectord" if row['feature'] in vectord_present else ""
                print(f"      {row['feature']:<35s} {row['pct']:5.1f}%{marker}")
            print(f"\n    All vectord feature ranks:")
            imp_reset = imp.reset_index(drop=True)
            for vf in vectord_present:
                rows = imp_reset[imp_reset['feature'] == vf]
                if len(rows) > 0:
                    rank = rows.index[0] + 1
                    print(f"      {vf:<35s} rank {rank:2d}/{len(imp)}, {rows['pct'].iloc[0]:4.2f}%")

        # Train-fold median interval width (for WIDTH sizing)
        q_lo, q_hi = QUANTILES[0], QUANTILES[-1]
        train_width = q_train_preds[q_hi] - q_train_preds[q_lo]
        w_med = float(np.median(train_width[train_width > 0])) if (train_width > 0).any() else 1.0
        train_widths.append((fi, w_med))
        print(f"  Train median interval width (q90-q10): {w_med:.2f} EUR/MWh")

        # Monotonicity: enforce q10 <= q50 <= q90 on predictions
        q10 = pred_data['q10'].values
        q50 = pred_data['q50'].values
        q90 = pred_data['q90'].values
        cross = ((q10 > q50) | (q50 > q90)).sum()
        if cross > 0:
            print(f"  [!] {cross}/{len(pred_data)} quantile crossings — sorting")
            sorted_q = np.sort(np.column_stack([q10, q50, q90]), axis=1)
            pred_data['q10'] = sorted_q[:, 0]
            pred_data['q50'] = sorted_q[:, 1]
            pred_data['q90'] = sorted_q[:, 2]

        pred_data['width'] = pred_data['q90'] - pred_data['q10']
        pred_data['w_med_train'] = w_med

        if pred_data['spread_target'].notna().sum() > 0:
            sp_nz = pred_data['spread_target'].abs() > 0.1
            for alpha in QUANTILES:
                col = Q_COLS[alpha]
                dir_acc = (np.sign(pred_data[col]) == np.sign(pred_data['spread_target']))[sp_nz].mean()
                print(f"    {col} dir_acc={dir_acc:.1%}")

        oof = pred_data[['q10', 'q50', 'q90', 'width', 'w_med_train',
                         'target', 'spread_target',
                         'exec_bid', 'exec_ask', 'exec_spread', 'imb_settlement_price']].copy()
        oof['fold'] = fi
        all_oof.append(oof)

    # ============================================================
    # TRADING BACKTEST (test period: Feb-Mar 2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("TRADING BACKTEST — Quantile variants vs point-MAE baseline")
    print("=" * 70)

    # Every fold is a test fold under the post-regime design
    test_folds = all_oof
    if not test_folds:
        print("[!] No test fold predictions")
        return

    test_df = pd.concat(test_folds)
    test_df = test_df[test_df['exec_spread'].notna() & (test_df['exec_spread'] <= 10)]
    n_days = test_df.index.normalize().nunique()
    print(f"\nTest: {len(test_df)} periods, {n_days} days")

    def pnl_frame(sub):
        """P&L from a frame already filtered to trades, with 'size' and 'direction'."""
        sub = sub.copy()
        sub['pnl'] = 0.0
        s = sub['direction'] == 'surplus'
        d = sub['direction'] == 'deficit'
        sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * sub.loc[s, 'size'] / 4
        sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * sub.loc[d, 'size'] / 4
        return sub

    def report(sub, label):
        if len(sub) < 30:
            print(f"  {label:<55s} too few trades ({len(sub)})")
            return None
        nd = sub.index.normalize().nunique()
        total = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean()
        daily = sub.groupby(sub.index.date)['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        dd = (daily.cumsum() - daily.cumsum().cummax()).min()
        avg_sz = sub['size'].mean()
        print(f"  {label:<55s} {len(sub):4d}t, avg_sz={avg_sz:.2f}, win={wr:.0%}, "
              f"{total:>+7,.0f} ({total/nd:>+4.0f}/d) Sharpe={sharpe:.1f} DD={dd:+.0f}")
        return {'total': total, 'daily': total / nd, 'sharpe': sharpe, 'dd': dd,
                'trades': len(sub), 'wr': wr, 'avg_size': avg_sz}

    results = {}

    # --- BASELINE: current prod logic on q50 ---
    print("\n--- BASELINE: |q50| >= T, size = |q50|.clip(<=5) ---")
    for T in [3, 5, 8]:
        t = test_df[test_df['q50'].abs() >= T].copy()
        t['direction'] = np.where(t['q50'] >= T, 'deficit',
                         np.where(t['q50'] <= -T, 'surplus', None))
        t['size'] = t['q50'].abs().clip(upper=5)
        t = pnl_frame(t)
        results[f'baseline_T{T}'] = report(t, f'Baseline |q50|>={T}')

    # --- EXCLUDE: trade only when interval [q10,q90] excludes 0 by threshold ---
    print("\n--- EXCLUDE: q10 > T (deficit) OR q90 < -T (surplus), size = |q50|.clip(<=5) ---")
    for T in [0, 1, 2, 3]:
        deficit = test_df['q10'] > T
        surplus = test_df['q90'] < -T
        t = test_df[deficit | surplus].copy()
        t['direction'] = np.where(t['q10'] > T, 'deficit',
                         np.where(t['q90'] < -T, 'surplus', None))
        t['size'] = t['q50'].abs().clip(upper=5)
        t = pnl_frame(t)
        results[f'exclude_T{T}'] = report(t, f'Exclude (q10>{T} or q90<-{T})')

    # --- WIDTH: |q50|>=T, size scaled by inverse interval width vs train median ---
    print("\n--- WIDTH: |q50|>=T, size = |q50|.clip(<=5) * clip(w_med/w, 0.3, 1.5) ---")
    for T in [3, 5]:
        t = test_df[test_df['q50'].abs() >= T].copy()
        t['direction'] = np.where(t['q50'] >= T, 'deficit',
                         np.where(t['q50'] <= -T, 'surplus', None))
        base_sz = t['q50'].abs().clip(upper=5)
        conv = (t['w_med_train'] / t['width'].clip(lower=0.1)).clip(lower=0.3, upper=1.5)
        t['size'] = base_sz * conv
        t = pnl_frame(t)
        results[f'width_T{T}'] = report(t, f'Width-scaled |q50|>={T}')

    # --- COMBINED: exclude filter + width sizing ---
    print("\n--- COMBINED: q10>T or q90<-T, size scaled by inverse width ---")
    for T in [0, 1, 2]:
        deficit = test_df['q10'] > T
        surplus = test_df['q90'] < -T
        t = test_df[deficit | surplus].copy()
        t['direction'] = np.where(t['q10'] > T, 'deficit',
                         np.where(t['q90'] < -T, 'surplus', None))
        base_sz = t['q50'].abs().clip(upper=5)
        conv = (t['w_med_train'] / t['width'].clip(lower=0.1)).clip(lower=0.3, upper=1.5)
        t['size'] = base_sz * conv
        t = pnl_frame(t)
        results[f'combined_T{T}'] = report(t, f'Combined (T={T})')

    # ============================================================
    # Monthly breakdown of best-looking variants (baseline T=3 + combined T=1)
    # ============================================================
    print("\n" + "=" * 70)
    print("MONTHLY BREAKDOWN")
    print("=" * 70)

    def monthly(t, label):
        if t is None or len(t) == 0:
            return
        print(f"\n{label}:")
        t = t.copy()
        t['month'] = t.index.to_period('M')
        for period in sorted(t['month'].unique().astype(str)):
            sub = t[t['month'].astype(str) == period]
            nd = sub.index.normalize().nunique()
            p = sub['pnl'].sum()
            wr = (sub['pnl'] > 0).mean()
            print(f"    {period}: {len(sub):4d}t, win={wr:.0%}, {p:>+7,.0f} ({p/nd:>+4.0f}/d)")

    # Rebuild the two reference scenarios for monthly view
    t_base = test_df[test_df['q50'].abs() >= 3].copy()
    t_base['direction'] = np.where(t_base['q50'] >= 3, 'deficit',
                           np.where(t_base['q50'] <= -3, 'surplus', None))
    t_base['size'] = t_base['q50'].abs().clip(upper=5)
    t_base = pnl_frame(t_base)
    monthly(t_base, "Baseline |q50|>=3")

    for T in [1, 2]:
        deficit = test_df['q10'] > T
        surplus = test_df['q90'] < -T
        t = test_df[deficit | surplus].copy()
        t['direction'] = np.where(t['q10'] > T, 'deficit',
                         np.where(t['q90'] < -T, 'surplus', None))
        base_sz = t['q50'].abs().clip(upper=5)
        conv = (t['w_med_train'] / t['width'].clip(lower=0.1)).clip(lower=0.3, upper=1.5)
        t['size'] = base_sz * conv
        t = pnl_frame(t)
        monthly(t, f"Combined T={T}")

    # ============================================================
    # Save predictions
    # ============================================================
    out_path = DATA_DIR / "predictions" / "quantile_spread_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_oof).to_csv(out_path)
    print(f"\n[+] Saved: {out_path}")

    # Summary table of best variants
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'variant':<30s} {'trades':>7s} {'avg_sz':>7s} {'EUR/d':>8s} {'Sharpe':>7s} {'DD':>7s} {'win':>5s}")
    print("-" * 75)
    for k, v in results.items():
        if v is None:
            continue
        print(f"{k:<30s} {v['trades']:>7d} {v['avg_size']:>7.2f} "
              f"{v['daily']:>+8.0f} {v['sharpe']:>7.1f} {v['dd']:>+7.0f} {v['wr']:>5.0%}")


if __name__ == "__main__":
    main()
