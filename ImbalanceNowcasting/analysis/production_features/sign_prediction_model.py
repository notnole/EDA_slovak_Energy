"""
Sign Prediction Model using Logistic Regression
Uses all 3-min SCADA data with lags and trends to predict imbalance SIGN.

Focus on Lead 9 and smaller (2+ observations available).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent.parent.parent
FEATURES_DIR = ROOT / "data" / "features"
MASTER_FILE = ROOT / "data" / "master" / "master_imbalance_data.csv"


def load_all_features():
    """Load all 3-minute SCADA features."""
    print("[*] Loading 3-minute SCADA data...")

    # Regulation
    reg = pd.read_csv(FEATURES_DIR / "regulation_3min.csv")
    reg['datetime'] = pd.to_datetime(reg['datetime'])
    reg = reg.rename(columns={'regulation_mw': 'regulation'})

    # Load
    load = pd.read_csv(FEATURES_DIR / "load_3min.csv")
    load['datetime'] = pd.to_datetime(load['datetime'])
    load = load.rename(columns={'load_mw': 'load'})

    # Production
    prod = pd.read_csv(FEATURES_DIR / "production_3min.csv")
    prod['datetime'] = pd.to_datetime(prod['datetime'])
    prod = prod.rename(columns={'production_mw': 'production'})

    # Export/Import
    exp = pd.read_csv(FEATURES_DIR / "export_import_3min.csv")
    exp['datetime'] = pd.to_datetime(exp['datetime'])
    exp = exp.rename(columns={'export_import_mw': 'export_import'})

    # Merge all
    df = reg.merge(load, on='datetime', how='outer')
    df = df.merge(prod, on='datetime', how='outer')
    df = df.merge(exp, on='datetime', how='outer')

    # Sort and assign settlement periods
    df = df.sort_values('datetime').reset_index(drop=True)
    df['settlement_start'] = df['datetime'].dt.floor('15min')

    # Observation number within period (1-5)
    df['obs_num'] = df.groupby('settlement_start').cumcount() + 1

    print(f"   Loaded {len(df)} observations")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Features: regulation, load, production, export_import")

    return df


def load_imbalance():
    """Load imbalance labels."""
    print("[*] Loading imbalance labels...")
    df = pd.read_csv(MASTER_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['imbalance'] = df['System Imbalance (MWh)']
    df['sign'] = np.sign(df['imbalance'])
    # Convert to binary: 1 = positive, 0 = negative/zero
    df['sign_binary'] = (df['sign'] > 0).astype(int)
    return df[['datetime', 'imbalance', 'sign', 'sign_binary']]


def create_lead_time_features(scada, lead_time):
    """
    Create features for a specific lead time.

    Lead 12: 1 obs (obs_num=1)
    Lead 9:  2 obs (obs_num=1,2)
    Lead 6:  3 obs (obs_num=1,2,3)
    Lead 3:  4 obs (obs_num=1,2,3,4)
    Lead 0:  5 obs (obs_num=1,2,3,4,5)
    """
    n_obs = (12 - lead_time) // 3 + 1

    # Filter to relevant observations
    df = scada[scada['obs_num'] <= n_obs].copy()

    features_list = []

    for settlement, group in df.groupby('settlement_start'):
        if len(group) < n_obs:
            continue

        group = group.sort_values('obs_num')
        feat = {'settlement_start': settlement}

        for col in ['regulation', 'load', 'production', 'export_import']:
            values = group[col].values

            # Current values (by observation)
            for i, v in enumerate(values):
                feat[f'{col}_obs{i+1}'] = v

            # Mean
            feat[f'{col}_mean'] = np.nanmean(values)

            # If multiple observations: trends
            if n_obs >= 2:
                feat[f'{col}_first'] = values[0]
                feat[f'{col}_last'] = values[-1]
                feat[f'{col}_change'] = values[-1] - values[0]
                feat[f'{col}_trend'] = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0

            if n_obs >= 3:
                feat[f'{col}_std'] = np.nanstd(values)
                feat[f'{col}_range'] = np.nanmax(values) - np.nanmin(values)

        features_list.append(feat)

    return pd.DataFrame(features_list)


def add_historical_features(df, imbalance):
    """Add features from previous settlement periods."""
    df = df.sort_values('settlement_start').copy()

    # Merge with imbalance to get previous period info
    imb = imbalance.rename(columns={'datetime': 'settlement_start'})
    df = df.merge(imb[['settlement_start', 'imbalance', 'sign']], on='settlement_start', how='inner')

    # Previous period features
    df['prev_imbalance'] = df['imbalance'].shift(1)
    df['prev_sign'] = df['sign'].shift(1)
    df['prev_imbalance_abs'] = df['prev_imbalance'].abs()

    # 2 periods ago
    df['prev2_imbalance'] = df['imbalance'].shift(2)
    df['prev2_sign'] = df['sign'].shift(2)

    # Sign changes in recent history
    df['sign_change_1'] = (df['sign'] != df['prev_sign']).astype(int).shift(1)
    df['sign_change_2'] = (df['prev_sign'] != df['prev2_sign']).astype(int).shift(1)

    # Running average of recent imbalances
    df['imbalance_ma3'] = df['imbalance'].shift(1).rolling(3, min_periods=1).mean()
    df['imbalance_ma6'] = df['imbalance'].shift(1).rolling(6, min_periods=1).mean()

    # Target
    df['target'] = (df['sign'] > 0).astype(int)

    # Only drop rows missing essential columns
    essential = ['prev_imbalance', 'target']
    return df.dropna(subset=essential)


def train_sign_model(df, feature_cols, lead_time):
    """Train and evaluate sign prediction model."""
    print(f"\n{'='*60}")
    print(f"LEAD TIME {lead_time} MINUTES")
    print(f"{'='*60}")

    # Time series split (no data leakage)
    df = df.sort_values('settlement_start').reset_index(drop=True)

    # Use last 20% as test
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols].fillna(0)
    y_train = train['target']
    X_test = test[feature_cols].fillna(0)
    y_test = test['target']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n[*] Data: {len(train)} train, {len(test)} test periods")
    print(f"[*] Features used: {len(feature_cols)}")
    print(f"\n[*] Train accuracy: {train_acc*100:.1f}%")
    print(f"[*] Test accuracy:  {test_acc*100:.1f}%")

    # Baseline comparison (persistence)
    baseline_pred = test['prev_sign'].apply(lambda x: 1 if x > 0 else 0)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"[*] Baseline (persistence): {baseline_acc*100:.1f}%")

    # Baseline using regulation sign
    reg_baseline = test['regulation_mean'].apply(lambda x: 1 if x < 0 else 0)  # negative reg -> positive imbalance
    reg_acc = accuracy_score(y_test, reg_baseline)
    print(f"[*] Baseline (regulation sign): {reg_acc*100:.1f}%")

    # Improvement
    improvement = test_acc - baseline_acc
    print(f"\n[*] Improvement over persistence: {improvement*100:+.1f} pp")

    # Top features
    print(f"\n[*] Top 10 features by importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coef': model.coef_[0]
    }).sort_values('coef', key=abs, ascending=False)

    for i, row in importance.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['coef']:+.4f}")

    # Analyze errors
    test_with_pred = test.copy()
    test_with_pred['pred'] = y_pred_test
    test_with_pred['correct'] = test_with_pred['pred'] == test_with_pred['target']

    # Accuracy by previous imbalance magnitude
    print(f"\n[*] Accuracy by |prev_imbalance|:")
    for thresh in [2, 5, 10]:
        small = test_with_pred[test_with_pred['prev_imbalance_abs'] < thresh]
        large = test_with_pred[test_with_pred['prev_imbalance_abs'] >= thresh]
        if len(small) > 0 and len(large) > 0:
            print(f"   |prev| < {thresh}: {small['correct'].mean()*100:.1f}% (n={len(small)})")
            print(f"   |prev| >= {thresh}: {large['correct'].mean()*100:.1f}% (n={len(large)})")

    return model, scaler, test_acc, baseline_acc


def main():
    print("="*60)
    print("SIGN PREDICTION MODEL")
    print("Logistic Regression with SCADA features + lags + trends")
    print("="*60 + "\n")

    # Load data
    scada = load_all_features()
    imbalance = load_imbalance()

    results = []

    # Test each lead time
    for lead_time in [9, 6, 3, 0]:
        print(f"\n[*] Creating features for Lead {lead_time}...")
        features = create_lead_time_features(scada, lead_time)
        print(f"   Created {len(features)} periods with features")

        # Add historical features
        df = add_historical_features(features, imbalance)
        print(f"   After adding history: {len(df)} periods")

        if len(df) < 1000:
            print(f"   [!] Not enough data, skipping")
            continue

        # Define feature columns (exclude target and metadata)
        exclude = ['settlement_start', 'imbalance', 'sign', 'target']
        feature_cols = [c for c in df.columns if c not in exclude]

        # Train model
        model, scaler, test_acc, baseline_acc = train_sign_model(df, feature_cols, lead_time)

        results.append({
            'lead_time': lead_time,
            'test_acc': test_acc,
            'baseline_acc': baseline_acc,
            'improvement': test_acc - baseline_acc,
            'n_features': len(feature_cols),
            'n_samples': len(df)
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    print("\n[*] Key findings:")
    for _, row in results_df.iterrows():
        status = "BETTER" if row['improvement'] > 0 else "WORSE"
        print(f"   Lead {row['lead_time']}: {row['test_acc']*100:.1f}% ({row['improvement']*100:+.1f}pp vs persistence) - {status}")

    return results_df


if __name__ == "__main__":
    results = main()
