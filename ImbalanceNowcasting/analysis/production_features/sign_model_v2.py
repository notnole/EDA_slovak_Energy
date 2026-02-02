"""
Sign Prediction Model v2
Uses only regulation + load (2 years of data) for maximum training data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent.parent.parent
FEATURES_DIR = ROOT / "data" / "features"
MASTER_FILE = ROOT / "data" / "master" / "master_imbalance_data.csv"


def load_data():
    """Load regulation and load data (both have 2 years)."""
    print("[*] Loading SCADA data...")

    reg = pd.read_csv(FEATURES_DIR / "regulation_3min.csv")
    reg['datetime'] = pd.to_datetime(reg['datetime'])
    reg = reg.rename(columns={'regulation_mw': 'regulation'})

    load = pd.read_csv(FEATURES_DIR / "load_3min.csv")
    load['datetime'] = pd.to_datetime(load['datetime'])
    load = load.rename(columns={'load_mw': 'load'})

    df = reg.merge(load, on='datetime', how='inner')
    df = df.sort_values('datetime').reset_index(drop=True)
    df['settlement_start'] = df['datetime'].dt.floor('15min')
    df['obs_num'] = df.groupby('settlement_start').cumcount() + 1

    print(f"   {len(df)} observations")
    print(f"   {df['datetime'].min()} to {df['datetime'].max()}")

    # Imbalance
    imb = pd.read_csv(MASTER_FILE)
    imb['datetime'] = pd.to_datetime(imb['datetime'])
    imb['imbalance'] = imb['System Imbalance (MWh)']
    imb['sign'] = np.sign(imb['imbalance'])

    return df, imb


def create_features(scada, imbalance, lead_time):
    """Create features for specific lead time."""
    n_obs = (12 - lead_time) // 3 + 1

    df = scada[scada['obs_num'] <= n_obs].copy()
    features_list = []

    for settlement, group in df.groupby('settlement_start'):
        if len(group) < n_obs:
            continue

        group = group.sort_values('obs_num')
        feat = {'settlement_start': settlement}

        for col in ['regulation', 'load']:
            values = group[col].values

            for i, v in enumerate(values):
                feat[f'{col}_obs{i+1}'] = v

            feat[f'{col}_mean'] = np.mean(values)

            if n_obs >= 2:
                feat[f'{col}_first'] = values[0]
                feat[f'{col}_last'] = values[-1]
                feat[f'{col}_change'] = values[-1] - values[0]

            if n_obs >= 3:
                feat[f'{col}_std'] = np.std(values)
                feat[f'{col}_min'] = np.min(values)
                feat[f'{col}_max'] = np.max(values)

        features_list.append(feat)

    feat_df = pd.DataFrame(features_list)

    # Merge with imbalance
    imb = imbalance.rename(columns={'datetime': 'settlement_start'})
    feat_df = feat_df.merge(imb[['settlement_start', 'imbalance', 'sign']], on='settlement_start', how='inner')

    # Add previous period features
    feat_df = feat_df.sort_values('settlement_start')
    feat_df['prev_imbalance'] = feat_df['imbalance'].shift(1)
    feat_df['prev_sign'] = feat_df['sign'].shift(1)
    feat_df['prev_abs'] = feat_df['prev_imbalance'].abs()
    feat_df['prev2_imbalance'] = feat_df['imbalance'].shift(2)

    # Baseline prediction: -0.25 * regulation_mean
    feat_df['baseline_pred'] = -0.25 * feat_df['regulation_mean']
    feat_df['baseline_sign'] = np.sign(feat_df['baseline_pred'])

    # Target
    feat_df['target'] = (feat_df['sign'] > 0).astype(int)

    return feat_df.dropna(subset=['prev_imbalance'])


def train_and_evaluate(df, lead_time):
    """Train and evaluate model."""
    print(f"\n{'='*60}")
    print(f"LEAD TIME {lead_time} MINUTES ({(12-lead_time)//3 + 1} observations)")
    print(f"{'='*60}")

    exclude = ['settlement_start', 'imbalance', 'sign', 'target', 'baseline_pred', 'baseline_sign']
    feature_cols = [c for c in df.columns if c not in exclude]

    # Time split
    df = df.sort_values('settlement_start').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train[feature_cols].fillna(0)
    y_train = train['target']
    X_test = test[feature_cols].fillna(0)
    y_test = test['target']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    # Accuracies
    model_acc = accuracy_score(y_test, y_pred)
    persist_acc = accuracy_score(y_test, (test['prev_sign'] > 0).astype(int))
    baseline_acc = accuracy_score(y_test, (test['baseline_sign'] > 0).astype(int))

    print(f"\n[*] Train: {len(train)}, Test: {len(test)} periods")
    print(f"[*] Features: {len(feature_cols)}")

    print(f"\n[*] SIGN PREDICTION ACCURACY:")
    print(f"   Logistic Model:    {model_acc*100:.1f}%")
    print(f"   Baseline (-0.25*R):{baseline_acc*100:.1f}%")
    print(f"   Persistence:       {persist_acc*100:.1f}%")

    print(f"\n[*] vs Baseline: {(model_acc - baseline_acc)*100:+.1f} pp")
    print(f"[*] vs Persistence: {(model_acc - persist_acc)*100:+.1f} pp")

    # By magnitude
    print(f"\n[*] By |prev_imbalance|:")
    for thresh in [2, 5, 10]:
        small = test[test['prev_abs'] < thresh]
        large = test[test['prev_abs'] >= thresh]
        if len(small) > 50:
            small_acc = accuracy_score(small['target'], model.predict(scaler.transform(small[feature_cols].fillna(0))))
            small_base = accuracy_score(small['target'], (small['baseline_sign'] > 0).astype(int))
            print(f"   |prev| < {thresh}: Model {small_acc*100:.1f}%, Baseline {small_base*100:.1f}%, n={len(small)}")

    # Top features
    print(f"\n[*] Top features:")
    imp = pd.DataFrame({'feat': feature_cols, 'coef': model.coef_[0]})
    imp = imp.reindex(imp['coef'].abs().sort_values(ascending=False).index)
    for _, row in imp.head(8).iterrows():
        print(f"   {row['feat']:25s} {row['coef']:+.4f}")

    return model_acc, baseline_acc, persist_acc


def main():
    print("="*60)
    print("SIGN PREDICTION MODEL v2")
    print("Using regulation + load (2 years of data)")
    print("="*60)

    scada, imbalance = load_data()

    results = []
    for lead in [12, 9, 6, 3, 0]:
        df = create_features(scada, imbalance, lead)
        if len(df) < 1000:
            print(f"\nLead {lead}: Not enough data ({len(df)} periods)")
            continue
        model_acc, baseline_acc, persist_acc = train_and_evaluate(df, lead)
        results.append({
            'lead': lead,
            'model': model_acc,
            'baseline': baseline_acc,
            'persist': persist_acc,
            'vs_baseline': model_acc - baseline_acc,
            'vs_persist': model_acc - persist_acc
        })

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    res_df = pd.DataFrame(results)
    print("\nLead | Model  | Baseline | Persist | vs Base | vs Persist")
    print("-"*60)
    for _, r in res_df.iterrows():
        print(f" {r['lead']:3.0f} | {r['model']*100:5.1f}% | {r['baseline']*100:6.1f}%  | {r['persist']*100:6.1f}% | {r['vs_baseline']*100:+5.1f}pp | {r['vs_persist']*100:+6.1f}pp")

    return res_df


if __name__ == "__main__":
    results = main()
