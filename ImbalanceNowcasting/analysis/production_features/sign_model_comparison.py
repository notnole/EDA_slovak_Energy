"""
Sign Prediction Model Comparison
Compares: LightGBM (magnitude), Logistic (sign), Baseline, Ensembles

Uses all 4 features where available + consecutive run feature.

The 5 observations within a 15-min settlement period:
- reg_min0:  observation at ~minute 3  -> Lead 12 (1 obs available)
- reg_min3:  observation at ~minute 6  -> Lead 9  (2 obs available)
- reg_min6:  observation at ~minute 9  -> Lead 6  (3 obs available)
- reg_min9:  observation at ~minute 12 -> Lead 3  (4 obs available)
- reg_min12: observation at ~minute 15 -> Lead 0  (5 obs available)

Regulation proxy for imbalance: imbalance_proxy = -0.25 * regulation_mean
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent.parent.parent
FEATURES_DIR = ROOT / "data" / "features"
MASTER_FILE = ROOT / "data" / "master" / "master_imbalance_data.csv"


def load_all_data():
    """Load all SCADA features and labels."""
    print("="*60)
    print("LOADING DATA")
    print("="*60)

    # Regulation (2 years)
    reg = pd.read_csv(FEATURES_DIR / "regulation_3min.csv", parse_dates=['datetime'])
    reg = reg.rename(columns={'regulation_mw': 'regulation'})
    print(f"Regulation: {len(reg)} rows, {reg['datetime'].min().date()} to {reg['datetime'].max().date()}")

    # Load (2 years)
    load = pd.read_csv(FEATURES_DIR / "load_3min.csv", parse_dates=['datetime'])
    load = load.rename(columns={'load_mw': 'load'})
    print(f"Load: {len(load)} rows")

    # Production (~109 days)
    prod = pd.read_csv(FEATURES_DIR / "production_3min.csv", parse_dates=['datetime'])
    prod = prod.rename(columns={'production_mw': 'production'})
    print(f"Production: {len(prod)} rows, {prod['datetime'].min().date()} to {prod['datetime'].max().date()}")

    # Export/Import (~109 days)
    exp = pd.read_csv(FEATURES_DIR / "export_import_3min.csv", parse_dates=['datetime'])
    exp = exp.rename(columns={'export_import_mw': 'export_import'})
    print(f"Export/Import: {len(exp)} rows")

    # Labels
    labels = pd.read_csv(MASTER_FILE, parse_dates=['datetime'])
    labels['imbalance'] = labels['System Imbalance (MWh)']
    labels['sign'] = np.sign(labels['imbalance'])
    print(f"Labels: {len(labels)} periods")

    return reg, load, prod, exp, labels


def pivot_to_settlement(df, value_col, prefix):
    """Pivot 3-min data to settlement period features."""
    df = df.copy()
    df['settlement_start'] = df['datetime'].dt.floor('15min')
    df['minute_in_qh'] = ((df['datetime'] - df['settlement_start']).dt.total_seconds() / 60).round().astype(int)

    # Keep only valid minute positions (0, 3, 6, 9, 12)
    df = df[df['minute_in_qh'].isin([0, 3, 6, 9, 12])]

    pivot = df.pivot_table(
        index='settlement_start',
        columns='minute_in_qh',
        values=value_col,
        aggfunc='first'
    ).reset_index()

    # Rename columns: min0 = obs at minute 3 (Lead 12), etc.
    pivot.columns = ['settlement_start'] + [f'{prefix}_min{int(c)}' for c in pivot.columns[1:]]

    return pivot


def compute_consecutive_run(df, proxy_col):
    """
    Compute consecutive run of same-sign proxy values.
    Uses regulation proxy: imbalance_proxy = -0.25 * regulation

    Returns: run_length (positive = consecutive positive, negative = consecutive negative)
    """
    df = df.sort_values('settlement_start').copy()
    df['proxy_sign'] = np.sign(df[proxy_col])

    # Compute run length
    run_lengths = []
    current_run = 0
    current_sign = 0

    for sign in df['proxy_sign']:
        if sign == current_sign:
            current_run += 1
        else:
            current_run = 1
            current_sign = sign
        run_lengths.append(current_run * sign)  # Positive for positive runs, negative for negative

    df['consecutive_run'] = run_lengths
    # Shift by 1 to use previous period's run (we can't know current period's final run)
    df['consecutive_run'] = df['consecutive_run'].shift(1)

    return df


def create_features(reg, load, prod, exp, labels, lead_time, use_all_features=False):
    """Create feature matrix for specific lead time."""

    # Map lead time to available observations
    # Lead 12: min0 only (1 obs)
    # Lead 9: min0, min3 (2 obs)
    # Lead 6: min0, min3, min6 (3 obs)
    # Lead 3: min0, min3, min6, min9 (4 obs)
    # Lead 0: min0, min3, min6, min9, min12 (5 obs)
    available_mins = {
        12: [0],
        9: [0, 3],
        6: [0, 3, 6],
        3: [0, 3, 6, 9],
        0: [0, 3, 6, 9, 12]
    }[lead_time]

    # Pivot each feature
    reg_pivot = pivot_to_settlement(reg, 'regulation', 'reg')
    load_pivot = pivot_to_settlement(load, 'load', 'load')

    # Merge reg and load
    df = reg_pivot.merge(load_pivot, on='settlement_start', how='inner')

    if use_all_features:
        prod_pivot = pivot_to_settlement(prod, 'production', 'prod')
        exp_pivot = pivot_to_settlement(exp, 'export_import', 'exp')
        df = df.merge(prod_pivot, on='settlement_start', how='inner')
        df = df.merge(exp_pivot, on='settlement_start', how='inner')

    # Keep only columns for available lead time
    keep_cols = ['settlement_start']
    for prefix in ['reg', 'load'] + (['prod', 'exp'] if use_all_features else []):
        for m in available_mins:
            col = f'{prefix}_min{m}'
            if col in df.columns:
                keep_cols.append(col)

    df = df[keep_cols].copy()

    # Add derived features
    n_obs = len(available_mins)

    # Regulation features
    reg_cols = [c for c in df.columns if c.startswith('reg_min')]
    if len(reg_cols) > 0:
        df['reg_mean'] = df[reg_cols].mean(axis=1)
        df['reg_first'] = df[reg_cols[0]]
        df['reg_last'] = df[reg_cols[-1]]
        if len(reg_cols) >= 2:
            df['reg_change'] = df['reg_last'] - df['reg_first']
            df['reg_trend'] = np.polyfit(range(len(reg_cols)), df[reg_cols].values.T, 1)[0] if len(reg_cols) > 1 else 0
        if len(reg_cols) >= 3:
            df['reg_std'] = df[reg_cols].std(axis=1)

    # Load features
    load_cols = [c for c in df.columns if c.startswith('load_min')]
    if len(load_cols) > 0:
        df['load_mean'] = df[load_cols].mean(axis=1)
        if len(load_cols) >= 2:
            df['load_change'] = df[load_cols[-1]] - df[load_cols[0]]

    # Baseline prediction (proxy)
    df['baseline_pred'] = -0.25 * df['reg_mean']
    df['baseline_sign'] = np.sign(df['baseline_pred'])

    # Merge with labels
    labels_renamed = labels[['datetime', 'imbalance', 'sign']].rename(columns={'datetime': 'settlement_start'})
    df = df.merge(labels_renamed, on='settlement_start', how='inner')

    # Add historical features
    df = df.sort_values('settlement_start')
    df['prev_imbalance'] = df['imbalance'].shift(1)
    df['prev_sign'] = df['sign'].shift(1)
    df['prev_abs'] = df['prev_imbalance'].abs()
    df['prev2_imbalance'] = df['imbalance'].shift(2)
    df['prev3_imbalance'] = df['imbalance'].shift(3)

    # Consecutive run feature (using regulation proxy from PREVIOUS periods)
    df = compute_consecutive_run(df, 'baseline_pred')

    # Target
    df['target'] = (df['sign'] > 0).astype(int)
    df['sign_flipped'] = df['sign'] != df['prev_sign']

    return df.dropna(subset=['prev_imbalance', 'consecutive_run'])


def train_lgb_model(X_train, y_train, X_test, y_test):
    """Train LightGBM for magnitude prediction, evaluate sign accuracy."""
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 200
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return model, pred_train, pred_test


def train_logistic_model(X_train, y_train, X_test):
    """Train logistic regression for sign prediction."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_train_s, y_train)

    pred_train = model.predict(X_train_s)
    pred_test = model.predict(X_test_s)

    return model, scaler, pred_train, pred_test


def evaluate_models(df, lead_time, use_all_features=False):
    """Train and evaluate all models for a specific lead time."""
    feature_set = "All 4 features" if use_all_features else "Reg + Load only"
    print(f"\n{'='*60}")
    print(f"LEAD {lead_time} - {feature_set}")
    print(f"{'='*60}")

    # Define feature columns
    exclude = ['settlement_start', 'imbalance', 'sign', 'target', 'baseline_pred',
               'baseline_sign', 'sign_flipped', 'prev_sign']
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"Features: {len(feature_cols)}")
    print(f"Periods: {len(df)}")

    # Time-based split
    df = df.sort_values('settlement_start').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)
    y_train_sign = train['target']
    y_test_sign = test['target']
    y_train_mag = train['imbalance']
    y_test_mag = test['imbalance']

    results = {}

    # 1. Baseline (simple formula)
    baseline_pred = (test['baseline_sign'] > 0).astype(int)
    results['baseline'] = accuracy_score(y_test_sign, baseline_pred)

    # 2. Persistence
    persist_pred = (test['prev_sign'] > 0).astype(int)
    results['persistence'] = accuracy_score(y_test_sign, persist_pred)

    # 3. LightGBM (magnitude -> sign)
    lgb_model, lgb_train, lgb_test = train_lgb_model(X_train, y_train_mag, X_test, y_test_mag)
    lgb_sign_pred = (lgb_test > 0).astype(int)
    results['lgb_magnitude'] = accuracy_score(y_test_sign, lgb_sign_pred)

    # 4. Logistic Regression (sign-focused)
    log_model, scaler, log_train, log_test = train_logistic_model(X_train, y_train_sign, X_test)
    results['logistic_sign'] = accuracy_score(y_test_sign, log_test)

    # 5. Ensemble: Majority vote (baseline, lgb, logistic)
    votes = np.column_stack([baseline_pred, lgb_sign_pred, log_test])
    ensemble_pred = (votes.sum(axis=1) >= 2).astype(int)
    results['ensemble_vote'] = accuracy_score(y_test_sign, ensemble_pred)

    # 6. Ensemble: Use logistic when lgb and baseline disagree
    lgb_baseline_agree = (lgb_sign_pred == baseline_pred)
    ensemble_smart = np.where(lgb_baseline_agree, lgb_sign_pred, log_test)
    results['ensemble_smart'] = accuracy_score(y_test_sign, ensemble_smart)

    # Print results
    print(f"\n{'Model':<25} {'Accuracy':>10} {'vs Baseline':>12}")
    print("-"*50)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        diff = acc - results['baseline']
        print(f"{name:<25} {acc*100:>9.1f}% {diff*100:>+11.1f}pp")

    # Analyze by small imbalance
    small = test[test['prev_abs'] < 3]
    if len(small) > 50:
        print(f"\n[*] Small imbalance (|prev| < 3, n={len(small)}):")
        small_X = small[feature_cols].fillna(0)
        small_y = small['target']

        small_baseline = accuracy_score(small_y, (small['baseline_sign'] > 0).astype(int))
        small_lgb = accuracy_score(small_y, (lgb_model.predict(small_X) > 0).astype(int))
        small_log = accuracy_score(small_y, log_model.predict(scaler.transform(small_X)))

        print(f"   Baseline: {small_baseline*100:.1f}%")
        print(f"   LGB:      {small_lgb*100:.1f}%")
        print(f"   Logistic: {small_log*100:.1f}%")

    # Feature importance for logistic
    print(f"\n[*] Top logistic features:")
    imp = pd.DataFrame({'feat': feature_cols, 'coef': log_model.coef_[0]})
    imp = imp.reindex(imp['coef'].abs().sort_values(ascending=False).index)
    for _, row in imp.head(6).iterrows():
        print(f"   {row['feat']:25s} {row['coef']:+.4f}")

    return results


def main():
    print("="*60)
    print("SIGN PREDICTION MODEL COMPARISON")
    print("LightGBM vs Logistic vs Baseline vs Ensembles")
    print("="*60 + "\n")

    reg, load, prod, exp, labels = load_all_data()

    all_results = []

    # Test with reg+load only (more data)
    print("\n" + "#"*60)
    print("# PART 1: Regulation + Load (longer history)")
    print("#"*60)

    for lead in [12, 9, 6, 3, 0]:
        df = create_features(reg, load, prod, exp, labels, lead, use_all_features=False)
        if len(df) < 1000:
            print(f"\nLead {lead}: Not enough data ({len(df)} periods)")
            continue
        results = evaluate_models(df, lead, use_all_features=False)
        for model, acc in results.items():
            all_results.append({'lead': lead, 'features': 'reg+load', 'model': model, 'accuracy': acc})

    # Test with all 4 features (less data but richer)
    print("\n" + "#"*60)
    print("# PART 2: All 4 Features (~109 days)")
    print("#"*60)

    for lead in [9, 6, 3, 0]:  # Skip lead 12 (usually less data)
        df = create_features(reg, load, prod, exp, labels, lead, use_all_features=True)
        if len(df) < 500:
            print(f"\nLead {lead}: Not enough data ({len(df)} periods)")
            continue
        results = evaluate_models(df, lead, use_all_features=True)
        for model, acc in results.items():
            all_results.append({'lead': lead, 'features': 'all4', 'model': model, 'accuracy': acc})

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    results_df = pd.DataFrame(all_results)

    print("\n[*] Best model by lead time (Reg+Load):")
    for lead in [12, 9, 6, 3, 0]:
        subset = results_df[(results_df['lead'] == lead) & (results_df['features'] == 'reg+load')]
        if len(subset) > 0:
            best = subset.loc[subset['accuracy'].idxmax()]
            baseline_acc = subset[subset['model'] == 'baseline']['accuracy'].values[0]
            print(f"   Lead {lead}: {best['model']} ({best['accuracy']*100:.1f}%) vs baseline {baseline_acc*100:.1f}%")

    return results_df


if __name__ == "__main__":
    results = main()
