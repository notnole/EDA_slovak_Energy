"""
Import Deviation as Model Improvement
Tests if import_deviation can IMPROVE baseline/ML model, not replace it.

Key question: Can import_dev serve as a confidence indicator or tie-breaker?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.parent
RAW_DIR = ROOT / "RawData"
FEATURES_DIR = ROOT / "data" / "features"
MASTER_FILE = ROOT / "data" / "master" / "master_imbalance_data.csv"


def load_regulation():
    """Load 3-minute regulation data."""
    print("[*] Loading regulation data...")
    df = pd.read_csv(FEATURES_DIR / "regulation_3min.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"   {len(df)} rows, {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def load_import_deviation():
    """Load and compute import deviation from raw cross-border data."""
    print("[*] Loading cross-border data...")

    # Scheduled
    scheduled = pd.read_csv(
        RAW_DIR / "3MIN_ACK_REAL_BALNCE.csv",
        skiprows=1, names=['datetime', 'ms', 'scheduled', 'empty']
    )
    scheduled['datetime'] = pd.to_datetime(scheduled['datetime'], dayfirst=True)
    scheduled['scheduled'] = pd.to_numeric(scheduled['scheduled'], errors='coerce')

    # Actual
    actual = pd.read_csv(
        RAW_DIR / "3MIN_Export.csv",
        skiprows=1, names=['datetime', 'ms', 'actual', 'empty']
    )
    actual['datetime'] = pd.to_datetime(actual['datetime'], dayfirst=True)
    actual['actual'] = pd.to_numeric(actual['actual'], errors='coerce')

    # Merge
    merged = scheduled[['datetime', 'scheduled']].merge(
        actual[['datetime', 'actual']], on='datetime', how='inner'
    )
    merged['import_deviation'] = merged['actual'] - merged['scheduled']

    print(f"   {len(merged)} rows, {merged['datetime'].min()} to {merged['datetime'].max()}")
    return merged[['datetime', 'import_deviation']]


def load_imbalance():
    """Load master imbalance data."""
    print("[*] Loading imbalance data...")
    df = pd.read_csv(MASTER_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['imbalance'] = df['System Imbalance (MWh)']
    print(f"   {len(df)} rows")
    return df[['datetime', 'imbalance']]


def compute_baseline_predictions(regulation, imbalance):
    """Compute baseline predictions at Lead 12 (1 observation only)."""
    print("[*] Computing baseline predictions (Lead 12)...")

    reg = regulation.copy()

    # Assign to settlement period (regulation at time T belongs to period containing T)
    reg['settlement_start'] = reg['datetime'].dt.floor('15min')

    # For Lead 12: use only FIRST observation of the period
    # Sort and take first per period
    reg = reg.sort_values('datetime')
    first_obs = reg.groupby('settlement_start').first().reset_index()

    # Baseline formula: imbalance = -0.25 * regulation
    first_obs['baseline_pred'] = -0.25 * first_obs['regulation_mw']

    # Merge with actual imbalance
    merged = first_obs.merge(
        imbalance.rename(columns={'datetime': 'settlement_start'}),
        on='settlement_start', how='inner'
    )

    print(f"   {len(merged)} periods with baseline predictions")
    return merged


def add_import_deviation(df, import_dev):
    """Add import deviation to the dataset."""
    print("[*] Adding import deviation...")

    imp = import_dev.copy()
    imp['settlement_start'] = imp['datetime'].dt.floor('15min')

    # Aggregate to first observation (matching Lead 12)
    imp = imp.sort_values('datetime')
    imp_first = imp.groupby('settlement_start')['import_deviation'].first().reset_index()

    merged = df.merge(imp_first, on='settlement_start', how='inner')
    print(f"   {len(merged)} periods with import deviation")
    return merged


def analyze_improvement(df):
    """Analyze if import deviation improves baseline predictions."""
    print("\n" + "="*70)
    print("BASELINE MODEL PERFORMANCE")
    print("="*70)

    df = df.copy()

    # Signs
    df['actual_sign'] = np.sign(df['imbalance'])
    df['baseline_sign'] = np.sign(df['baseline_pred'])
    df['import_dev_sign'] = np.sign(df['import_deviation'])

    # Previous imbalance for context
    df = df.sort_values('settlement_start')
    df['prev_imbalance'] = df['imbalance'].shift(1)
    df['prev_sign'] = np.sign(df['prev_imbalance'])
    df['sign_flipped'] = df['prev_sign'] != df['actual_sign']
    df = df.dropna(subset=['prev_imbalance'])

    # Baseline accuracy
    baseline_correct = (df['baseline_sign'] == df['actual_sign']).mean()
    print(f"\n[*] Baseline directional accuracy: {baseline_correct*100:.1f}%")

    # Agreement analysis
    df['agree'] = df['baseline_sign'] == df['import_dev_sign']

    agree = df[df['agree']]
    disagree = df[~df['agree']]

    acc_agree = (agree['baseline_sign'] == agree['actual_sign']).mean()
    acc_disagree = (disagree['baseline_sign'] == disagree['actual_sign']).mean()

    print(f"\n[*] Baseline accuracy when agreeing with import_dev: {acc_agree*100:.1f}% (n={len(agree)})")
    print(f"[*] Baseline accuracy when disagreeing:              {acc_disagree*100:.1f}% (n={len(disagree)})")
    print(f"[*] Accuracy gap: {(acc_agree - acc_disagree)*100:.1f} percentage points")

    # On sign flips
    print("\n" + "="*70)
    print("SIGN FLIP ANALYSIS")
    print("="*70)

    flips = df[df['sign_flipped']]
    n_flips = len(flips)

    baseline_flip_acc = (flips['baseline_sign'] == flips['actual_sign']).mean()
    import_flip_acc = (flips['import_dev_sign'] == flips['actual_sign']).mean()

    print(f"\n[*] Sign flips: {n_flips} ({n_flips/len(df)*100:.1f}% of periods)")
    print(f"[*] Baseline accuracy on flips:    {baseline_flip_acc*100:.1f}%")
    print(f"[*] Import dev accuracy on flips:  {import_flip_acc*100:.1f}%")

    # Conditional strategy: use import_dev sign when baseline disagrees
    print("\n" + "="*70)
    print("CONDITIONAL STRATEGIES")
    print("="*70)

    # Strategy 1: When disagree, use import_dev
    df['strategy1'] = np.where(df['agree'], df['baseline_sign'], df['import_dev_sign'])
    strat1_acc = (df['strategy1'] == df['actual_sign']).mean()

    print(f"\n[*] Strategy 1: When disagree, use import_dev sign")
    print(f"   Accuracy: {strat1_acc*100:.1f}% (baseline: {baseline_correct*100:.1f}%)")
    print(f"   Change: {(strat1_acc - baseline_correct)*100:+.1f} pp")

    # Strategy 2: When disagree AND |prev| < threshold, use import_dev
    for thresh in [2, 3, 5]:
        small_prev = df['prev_imbalance'].abs() < thresh
        use_import = (~df['agree']) & small_prev

        df[f'strategy2_{thresh}'] = np.where(use_import, df['import_dev_sign'], df['baseline_sign'])
        acc = (df[f'strategy2_{thresh}'] == df['actual_sign']).mean()
        n_switched = use_import.sum()

        print(f"\n[*] Strategy 2 (thresh={thresh}): When disagree AND |prev|<{thresh}, use import_dev")
        print(f"   Accuracy: {acc*100:.1f}% (baseline: {baseline_correct*100:.1f}%)")
        print(f"   Change: {(acc - baseline_correct)*100:+.1f} pp")
        print(f"   Periods switched: {n_switched} ({n_switched/len(df)*100:.1f}%)")

    # Strategy 3: On flips only, use import_dev
    # We can't know flips in advance, but we can identify high-flip-risk periods
    print("\n" + "="*70)
    print("FLIP RISK PREDICTION")
    print("="*70)

    # Small |prev| indicates high flip risk
    for thresh in [2, 3, 5]:
        high_risk = df['prev_imbalance'].abs() < thresh

        flip_rate_high = df[high_risk]['sign_flipped'].mean()
        flip_rate_low = df[~high_risk]['sign_flipped'].mean()

        print(f"\n[*] |prev| < {thresh} MWh:")
        print(f"   Flip rate when high risk: {flip_rate_high*100:.1f}%")
        print(f"   Flip rate when low risk:  {flip_rate_low*100:.1f}%")

        # In high-risk periods, compare baseline vs import_dev
        high_risk_df = df[high_risk]
        baseline_hr_acc = (high_risk_df['baseline_sign'] == high_risk_df['actual_sign']).mean()
        import_hr_acc = (high_risk_df['import_dev_sign'] == high_risk_df['actual_sign']).mean()

        print(f"   Baseline accuracy (high risk): {baseline_hr_acc*100:.1f}%")
        print(f"   Import dev accuracy (high risk): {import_hr_acc*100:.1f}%")

    return df


def statistical_tests(df):
    """Statistical significance tests."""
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE")
    print("="*70)

    # McNemar test: are the two predictors significantly different?
    baseline_correct = df['baseline_sign'] == df['actual_sign']
    import_correct = df['import_dev_sign'] == df['actual_sign']

    # Contingency table
    both_right = (baseline_correct & import_correct).sum()
    both_wrong = (~baseline_correct & ~import_correct).sum()
    baseline_only = (baseline_correct & ~import_correct).sum()
    import_only = (~baseline_correct & import_correct).sum()

    print(f"\n[*] Contingency table:")
    print(f"   Both correct: {both_right}")
    print(f"   Both wrong: {both_wrong}")
    print(f"   Baseline only correct: {baseline_only}")
    print(f"   Import dev only correct: {import_only}")

    # McNemar test
    if baseline_only + import_only > 0:
        # Chi-square approximation
        chi2 = (baseline_only - import_only)**2 / (baseline_only + import_only)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        print(f"\n[*] McNemar test (baseline vs import_dev):")
        print(f"   Chi-square: {chi2:.2f}, p-value: {p_value:.4f}")

        if p_value < 0.05:
            if baseline_only > import_only:
                print(f"   Result: Baseline significantly BETTER")
            else:
                print(f"   Result: Import dev significantly BETTER")
        else:
            print(f"   Result: No significant difference")

    # Key insight: they're different, so combining them helps
    combined = both_right + baseline_only + import_only
    print(f"\n[*] Potential ceiling (if we knew which to use):")
    print(f"   Could achieve: {combined/len(df)*100:.1f}%")
    print(f"   vs baseline:   {baseline_correct.mean()*100:.1f}%")
    print(f"   vs import_dev: {import_correct.mean()*100:.1f}%")


def main():
    print("="*70)
    print("IMPORT DEVIATION AS MODEL IMPROVEMENT")
    print("Testing if import_dev can improve baseline predictions")
    print("="*70 + "\n")

    # Load data
    regulation = load_regulation()
    import_dev = load_import_deviation()
    imbalance = load_imbalance()

    # Compute baseline predictions
    df = compute_baseline_predictions(regulation, imbalance)

    # Add import deviation
    df = add_import_deviation(df, import_dev)

    # Analyze
    df = analyze_improvement(df)
    statistical_tests(df)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    return df


if __name__ == "__main__":
    df = main()
