"""Quick experiment: can we classify daily BESS revenue using D-1 features?"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

sys.path.insert(0, 'MarketPriceGap/da_indicator/scripts')
from build_da_indicator import load_all_data, build_features
from bess_simulation import optimize_bess_dp

# --- Load ---
df = load_all_data()
df, _ = build_features(df)
fcst = pd.read_csv('MarketPriceGap/da_forecast_analysis/data/forecast_vs_actual_merged.csv')
fcst['date'] = pd.to_datetime(fcst['date'])
fcst = fcst[fcst['sk_forecast1_spot'].notna()][['date', 'hour', 'sk_forecast1_spot']]
df['date'] = pd.to_datetime(df['date'])
df = df.merge(fcst, on=['date', 'hour'], how='left')
da_cov = df[df['sk_forecast1_spot'].notna()].groupby('date').size()
da_dates = sorted(da_cov[da_cov >= 20].index)
df_test = df[df['date'].isin(da_dates)].copy()

# --- Build daily feature table ---
rows = []
for date in da_dates:
    day = df_test[df_test['date'] == date].sort_values('hour')
    n = len(day)
    if n < 20:
        continue
    prices = day['da_price'].values[:n]
    forecasts = day['sk_forecast1_spot'].values[:n]
    loads_fcst = day['forecast_load_mw'].values[:n]
    solar = day['SolarActual'].values[:n] if 'SolarActual' in day.columns else np.full(n, np.nan)
    if np.any(np.isnan(prices)) or np.any(np.isnan(forecasts)):
        continue

    ranks = pd.Series(forecasts).rank(method='first').values
    _, rev, _ = optimize_bess_dp(prices, ranks, n_charge=4, n_discharge=4,
                                  start_soc=1, terminal_penalty=20.0)
    _, rev_pf, _ = optimize_bess_dp(prices, prices, n_charge=4, n_discharge=4,
                                     start_soc=1, terminal_penalty=20.0)
    _, rev_exp, _ = optimize_bess_dp(forecasts, ranks, n_charge=4, n_discharge=4,
                                      start_soc=1, terminal_penalty=20.0)
    dt = pd.to_datetime(date)
    rows.append({
        'date': date,
        'rev_actual': rev,
        'rev_perfect': rev_pf,
        # Forecast-based
        'fcst_spread': forecasts.max() - forecasts.min(),
        'fcst_std': forecasts.std(),
        'fcst_mean': forecasts.mean(),
        'rev_expected': rev_exp,
        # Load
        'load_fcst_mean': np.nanmean(loads_fcst),
        'load_fcst_range': np.nanmax(loads_fcst) - np.nanmin(loads_fcst)
            if not np.all(np.isnan(loads_fcst)) else np.nan,
        'load_fcst_dev': day['load_fcst_dev'].iloc[0]
            if 'load_fcst_dev' in day.columns else np.nan,
        # Solar yesterday
        'solar_yest_max': np.nanmax(solar) if not np.all(np.isnan(solar)) else np.nan,
        'solar_yest_mean': np.nanmean(solar) if not np.all(np.isnan(solar)) else np.nan,
        # Calendar
        'dow': dt.dayofweek,
        'is_weekend': int(dt.dayofweek >= 5),
        'month': dt.month,
        # Yesterday price
        'yest_spread': day['da_price_yest_h'].max() - day['da_price_yest_h'].min()
            if day['da_price_yest_h'].notna().sum() > 10 else np.nan,
        'yest_std': day['da_price_yest_h'].std()
            if day['da_price_yest_h'].notna().sum() > 10 else np.nan,
        # Nuclear
        'nuclear_mean': day['Nuclear'].mean()
            if 'Nuclear' in day.columns else np.nan,
    })

r = pd.DataFrame(rows)
r['date'] = pd.to_datetime(r['date'])
r = r.sort_values('date').reset_index(drop=True)
print(f"[+] Built daily features for {len(r)} days")

# Deviations from 30-day norms
r['load_dev_30d'] = r['load_fcst_mean'] - r['load_fcst_mean'].rolling(30, min_periods=7).mean()
r['solar_dev_30d'] = r['solar_yest_max'] - r['solar_yest_max'].rolling(30, min_periods=7).mean()
r['spread_dev_30d'] = r['fcst_spread'] - r['fcst_spread'].rolling(30, min_periods=7).mean()

# Feature sets
feat_minimal = ['fcst_spread']
feat_medium = ['fcst_spread', 'fcst_std', 'rev_expected', 'is_weekend']
feat_full = ['fcst_spread', 'fcst_std', 'fcst_mean', 'rev_expected',
             'load_fcst_range', 'load_dev_30d', 'solar_yest_max', 'solar_dev_30d',
             'is_weekend', 'dow', 'yest_spread', 'yest_std', 'spread_dev_30d']

# Targets
median_rev = r['rev_actual'].median()
r['is_good'] = (r['rev_actual'] >= median_rev).astype(int)
r['tercile'] = pd.qcut(r['rev_actual'], 3, labels=['Bottom', 'Middle', 'Top'])


def ts_cv(r, features, n_folds=3):
    """Expanding-window time-series CV."""
    n = len(r)
    fold_size = n // (n_folds + 1)

    preds_cls, actual_cls = [], []
    preds_reg, actual_reg = [], []
    fold_metrics = []

    for fold in range(n_folds):
        split = fold_size * (fold + 1)
        te_end = min(split + fold_size, n)
        if te_end <= split:
            break

        tr = r.iloc[:split]
        te = r.iloc[split:te_end]

        X_tr = tr[features].fillna(0).values
        X_te = te[features].fillna(0).values
        y_cls_tr = tr['is_good'].values
        y_cls_te = te['is_good'].values
        y_reg_tr = tr['rev_actual'].values
        y_reg_te = te['rev_actual'].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X_tr_s, y_cls_tr)
        pc = clf.predict(X_te_s)

        reg = Ridge(alpha=10.0)
        reg.fit(X_tr_s, y_reg_tr)
        pr = reg.predict(X_te_s)

        acc = accuracy_score(y_cls_te, pc)
        mae = np.mean(np.abs(pr - y_reg_te))
        rho = spearmanr(pr, y_reg_te)[0] if len(pr) > 5 else 0

        fold_metrics.append(dict(fold=fold + 1, train=split, test=te_end - split,
                                  acc=acc, mae=mae, rho=rho))
        preds_cls.extend(pc)
        actual_cls.extend(y_cls_te)
        preds_reg.extend(pr)
        actual_reg.extend(y_reg_te)

    oa = accuracy_score(actual_cls, preds_cls)
    omae = np.mean(np.abs(np.array(preds_reg) - np.array(actual_reg)))
    orho = spearmanr(preds_reg, actual_reg)[0] if len(preds_reg) > 5 else 0
    imps = dict(zip(features, clf.feature_importances_)) if hasattr(clf, 'feature_importances_') else None

    return oa, omae, orho, fold_metrics, imps


# === EVALUATE ===
print()
print("=" * 70)
print("  DAILY REVENUE PREDICTION: FEATURE SET COMPARISON")
print("  (Time-series CV: expanding window, 3 folds)")
print("=" * 70)

all_results = {}
for name, feats in [('Minimal (spread only)', feat_minimal),
                     ('Medium (+std,expected,wknd)', feat_medium),
                     ('Full (13 features)', feat_full)]:
    acc, mae, rho, folds, imps = ts_cv(r, feats)
    all_results[name] = (acc, mae, rho, folds, imps, feats)

    print(f"\n--- {name} ---")
    print(f"  Features: {', '.join(feats)}")
    print(f"  Binary accuracy: {acc:.1%}   (random=50%)")
    print(f"  Revenue MAE:     {mae:.0f} EUR/day")
    print(f"  Spearman rho:    {rho:.3f}")
    for fm in folds:
        print(f"    Fold {fm['fold']}: train={fm['train']} test={fm['test']} "
              f"acc={fm['acc']:.1%} MAE={fm['mae']:.0f} rho={fm['rho']:.3f}")


# === FEATURE IMPORTANCE ===
print()
print("--- FEATURE IMPORTANCE (GBM classifier, full model) ---")
_, _, _, _, imps_full, _ = all_results['Full (13 features)']
if imps_full:
    sorted_imps = sorted(imps_full.items(), key=lambda x: -x[1])
    for feat, imp in sorted_imps:
        bar = '#' * int(imp * 50)
        print(f"  {feat:<20s} {imp:.3f} {bar}")


# === TERCILE CLASSIFICATION ===
print()
print("--- TERCILE CLASSIFICATION (Bottom / Middle / Top) ---")
n = len(r)
fold_size = n // 4
all_pred_t, all_actual_t = [], []
for fold in range(3):
    split = fold_size * (fold + 1)
    te_end = min(split + fold_size, n)
    if te_end <= split:
        break
    tr = r.iloc[:split]
    te = r.iloc[split:te_end]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(tr[feat_full].fillna(0).values)
    X_te_s = sc.transform(te[feat_full].fillna(0).values)
    clf3 = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf3.fit(X_tr_s, tr['tercile'].values)
    pred3 = clf3.predict(X_te_s)
    all_pred_t.extend(pred3)
    all_actual_t.extend(te['tercile'].values)

acc3 = accuracy_score(all_actual_t, all_pred_t)
print(f"  Overall accuracy: {acc3:.1%}  (random=33.3%)")
for true_t in ['Bottom', 'Middle', 'Top']:
    mask = [a == true_t for a in all_actual_t]
    preds = [p for p, m in zip(all_pred_t, mask) if m]
    dist = Counter(preds)
    total = len(preds)
    if total > 0:
        correct_pct = dist.get(true_t, 0) / total * 100
        print(f"  Actual {true_t:<8s} -> Bottom={dist.get('Bottom',0)} "
              f"Middle={dist.get('Middle',0)} Top={dist.get('Top',0)} "
              f"(correct={correct_pct:.0f}%)")


# === CORRELATION MATRIX of key features ===
print()
print("--- CORRELATION OF D-1 FEATURES WITH ACTUAL REVENUE ---")
key_feats = ['fcst_spread', 'rev_expected', 'load_fcst_range', 'load_dev_30d',
             'solar_yest_max', 'solar_dev_30d', 'is_weekend', 'yest_spread',
             'spread_dev_30d']
for f in key_feats:
    if f in r.columns:
        c = r[f].corr(r['rev_actual'])
        rho_val = spearmanr(r[f].fillna(0), r['rev_actual'])[0]
        print(f"  {f:<20s}  Pearson={c:+.3f}  Spearman={rho_val:+.3f}")


# === ECONOMIC VALUE: what if we only trade on "good" predicted days? ===
print()
print("--- ECONOMIC VALUE: SELECTIVE TRADING ---")
# Use full model Ridge regression on expanding window
n = len(r)
fold_size = n // 4
r['pred_rev'] = np.nan
for fold in range(3):
    split = fold_size * (fold + 1)
    te_end = min(split + fold_size, n)
    if te_end <= split:
        break
    tr = r.iloc[:split]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(tr[feat_full].fillna(0).values)
    X_te_s = sc.transform(r.iloc[split:te_end][feat_full].fillna(0).values)
    reg = Ridge(alpha=10.0)
    reg.fit(X_tr_s, tr['rev_actual'].values)
    r.loc[r.index[split:te_end], 'pred_rev'] = reg.predict(X_te_s)

oos = r[r['pred_rev'].notna()].copy()
print(f"  Out-of-sample days: {len(oos)}")
print(f"  Total revenue (trade every day):  {oos['rev_actual'].sum():,.0f} EUR")
print(f"  Avg daily (trade every day):      {oos['rev_actual'].mean():.0f} EUR/day")

# What if we skip bottom-predicted days?
for threshold_pct in [25, 33, 50]:
    thresh = oos['pred_rev'].quantile(threshold_pct / 100)
    selected = oos[oos['pred_rev'] >= thresh]
    skipped = oos[oos['pred_rev'] < thresh]
    print(f"\n  Skip bottom {threshold_pct}% predicted days:")
    print(f"    Trade {len(selected)} days, skip {len(skipped)} days")
    print(f"    Total revenue:  {selected['rev_actual'].sum():,.0f} EUR")
    print(f"    Avg daily rev:  {selected['rev_actual'].mean():.0f} EUR/day "
          f"(vs {oos['rev_actual'].mean():.0f} all days)")
    print(f"    Revenue of skipped days: {skipped['rev_actual'].sum():,.0f} EUR "
          f"(avg {skipped['rev_actual'].mean():.0f}/day)")
    print(f"    Rev/day improvement:     {(selected['rev_actual'].mean() / oos['rev_actual'].mean() - 1)*100:+.1f}%")
