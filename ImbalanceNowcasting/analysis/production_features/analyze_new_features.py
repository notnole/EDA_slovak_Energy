"""
Production Feature Analysis for Imbalance Nowcasting
Analyzes whether new features (import_deviation, frequency) can improve predictions.

Focus areas:
1. Error Case Analysis - When model gets sign wrong, what do other features show?
2. Sign Flip Prediction - Can new features predict sign flips?
3. Agreement Analysis - When model and new features disagree, who's right?
4. Regulation Average Prediction - Can new features predict the full-period regulation?
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "ProductionData"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_prepare_data():
    """Load all production data and merge into analysis dataset."""

    # Load SEPS data (3-minute observations)
    seps = pd.read_csv(DATA_DIR / "beam_lite_seps_data.csv")
    seps['timestamp'] = pd.to_datetime(seps['timestamp'], utc=True)

    # Derived features
    seps['import_deviation'] = seps['cross_border_measured_mw'] - seps['cross_border_scheduled_mw']
    seps['freq_deviation'] = seps['frequency_hz'] - 50.0
    seps['gen_load_balance'] = seps['generation_mw'] - seps['load_mw']

    # Convert to CET (naive) for matching
    seps['timestamp_cet'] = seps['timestamp'].dt.tz_convert(None) + pd.Timedelta(hours=1)
    seps['settlement_start'] = seps['timestamp_cet'].dt.floor('15min')

    # Load predictions
    preds = pd.read_csv(DATA_DIR / "beam_lite_predictions.csv")
    preds['timestamp'] = pd.to_datetime(preds['timestamp'])
    preds['pred_date'] = preds['timestamp'].dt.strftime('%Y-%m-%d')
    preds['settlement_start'] = pd.to_datetime(
        preds['pred_date'] + ' ' + preds['quarter_hour'].astype(str)
    ) + pd.Timedelta(hours=1)  # UTC -> CET

    # Load actuals
    actuals = pd.read_csv(DATA_DIR / "SystemImbalance_2026-01-28_2026-01-30.csv", sep=';')
    actuals['Date'] = pd.to_datetime(actuals['Date'], format='%m/%d/%Y')
    actuals['settlement_start'] = actuals['Date'] + pd.to_timedelta(
        (actuals['Settlement Term'] - 1) * 15, unit='min'
    )
    actuals['actual_imbalance'] = actuals['System Imbalance (MWh)']

    # Aggregate SEPS by settlement period
    seps_agg = seps.groupby('settlement_start').agg({
        'balancing_energy_mw': ['mean', 'first', 'last', 'std'],
        'import_deviation': ['mean', 'first', 'last'],
        'frequency_hz': ['mean', 'first', 'last', 'min', 'max'],
        'freq_deviation': ['mean', 'sum'],
        'generation_mw': ['mean', 'first', 'last'],
        'load_mw': ['mean', 'first', 'last'],
        'cross_border_measured_mw': ['mean'],
        'cross_border_scheduled_mw': ['mean'],
    })
    seps_agg.columns = ['_'.join(col) for col in seps_agg.columns]
    seps_agg = seps_agg.reset_index()

    # Get Lead 12 predictions (first observation only)
    lead12 = preds[preds['lead_time_min'] == 12].copy()

    # Get full-period regulation average from Lead 0 predictions
    # (Lead 0 has cumulative mean of all 5 observations)
    lead0 = preds[preds['lead_time_min'] == 0][['settlement_start', 'reg_cumulative_mean']].copy()
    lead0 = lead0.rename(columns={'reg_cumulative_mean': 'reg_full_period_mean'})

    # Merge everything
    merged = lead12.merge(seps_agg, on='settlement_start', how='left')
    merged = merged.merge(lead0, on='settlement_start', how='left')
    merged = merged.merge(actuals[['settlement_start', 'actual_imbalance']], on='settlement_start', how='left')
    merged = merged.dropna(subset=['actual_imbalance'])

    # Calculate derived columns
    merged['actual_sign'] = np.sign(merged['actual_imbalance'])
    merged['model_pred_sign'] = np.sign(merged['prediction_mwh'])
    merged['baseline_pred_sign'] = np.sign(merged['baseline_pred'])
    merged['import_dev_sign'] = np.sign(merged['import_deviation_first'])
    merged['freq_dev_sign'] = np.sign(merged['freq_deviation_mean'])

    merged['abs_actual'] = merged['actual_imbalance'].abs()
    merged['sign_correct'] = merged['model_pred_sign'] == merged['actual_sign']

    # Previous period values
    merged = merged.sort_values('settlement_start')
    merged['prev_actual'] = merged['actual_imbalance'].shift(1)
    merged['prev_sign'] = np.sign(merged['prev_actual'])
    merged['sign_flip'] = merged['actual_sign'] != merged['prev_sign']

    return merged


def analysis_1_error_cases(df):
    """Analyze cases where model gets sign wrong."""
    print("=" * 70)
    print("ANALYSIS 1: ERROR CASE ANALYSIS")
    print("=" * 70)
    print()

    correct = df[df['sign_correct']]
    wrong = df[~df['sign_correct']]

    print(f"Total predictions: {len(df)}")
    print(f"Sign correct: {len(correct)} ({len(correct)/len(df)*100:.1f}%)")
    print(f"Sign wrong: {len(wrong)} ({len(wrong)/len(df)*100:.1f}%)")
    print()

    # Compare feature distributions
    print("Feature comparison (Error vs Correct cases):")
    print("-" * 70)

    features_to_compare = [
        ('import_deviation_first', 'Import Dev (first obs)'),
        ('import_deviation_mean', 'Import Dev (period mean)'),
        ('freq_deviation_mean', 'Frequency Dev (mean)'),
        ('balancing_energy_mw_first', 'Regulation (first obs)'),
        ('abs_actual', 'Actual magnitude'),
    ]

    results = []
    for col, name in features_to_compare:
        if col in df.columns:
            correct_mean = correct[col].mean()
            wrong_mean = wrong[col].mean()
            correct_std = correct[col].std()
            wrong_std = wrong[col].std()

            # T-test
            if len(wrong) > 2:
                t_stat, p_val = stats.ttest_ind(correct[col].dropna(), wrong[col].dropna())
            else:
                t_stat, p_val = np.nan, np.nan

            results.append({
                'feature': name,
                'correct_mean': correct_mean,
                'wrong_mean': wrong_mean,
                'diff': wrong_mean - correct_mean,
                'p_value': p_val
            })

            sig = '*' if p_val < 0.05 else ''
            print(f"{name:30s}: Correct={correct_mean:8.2f}, Wrong={wrong_mean:8.2f}, "
                  f"Diff={wrong_mean-correct_mean:+8.2f} {sig}")

    print()

    # Key question: When model is wrong, what did import_deviation predict?
    print("When model gets sign WRONG, what did other features predict?")
    print("-" * 70)

    if len(wrong) > 0:
        import_agrees_w_model = (wrong['import_dev_sign'] == wrong['model_pred_sign']).sum()
        import_agrees_w_actual = (wrong['import_dev_sign'] == wrong['actual_sign']).sum()

        print(f"Import deviation agrees with MODEL: {import_agrees_w_model}/{len(wrong)} "
              f"({import_agrees_w_model/len(wrong)*100:.1f}%)")
        print(f"Import deviation agrees with ACTUAL: {import_agrees_w_actual}/{len(wrong)} "
              f"({import_agrees_w_actual/len(wrong)*100:.1f}%)")

        freq_agrees_w_actual = (wrong['freq_dev_sign'] == wrong['actual_sign']).sum()
        print(f"Frequency deviation agrees with ACTUAL: {freq_agrees_w_actual}/{len(wrong)} "
              f"({freq_agrees_w_actual/len(wrong)*100:.1f}%)")

    print()

    # Can we identify error cases in advance?
    print("Can we predict when model will be wrong?")
    print("-" * 70)

    # Check if disagreement between model and import_dev predicts error
    df['model_import_disagree'] = df['model_pred_sign'] != df['import_dev_sign']

    agree_cases = df[~df['model_import_disagree']]
    disagree_cases = df[df['model_import_disagree']]

    if len(agree_cases) > 0 and len(disagree_cases) > 0:
        agree_error_rate = (~agree_cases['sign_correct']).mean()
        disagree_error_rate = (~disagree_cases['sign_correct']).mean()

        print(f"Model & Import AGREE (n={len(agree_cases)}): Error rate = {agree_error_rate*100:.1f}%")
        print(f"Model & Import DISAGREE (n={len(disagree_cases)}): Error rate = {disagree_error_rate*100:.1f}%")

        if disagree_error_rate > agree_error_rate:
            print("=> Disagreement is a WARNING signal for potential error")
        else:
            print("=> Disagreement is NOT a useful warning signal")

    return pd.DataFrame(results)


def analysis_2_sign_flip_prediction(df):
    """Analyze whether new features can predict sign flips."""
    print()
    print("=" * 70)
    print("ANALYSIS 2: SIGN FLIP PREDICTION")
    print("=" * 70)
    print()

    df = df.dropna(subset=['sign_flip'])

    flips = df[df['sign_flip']]
    non_flips = df[~df['sign_flip']]

    print(f"Total transitions: {len(df)}")
    print(f"Sign flips: {len(flips)} ({len(flips)/len(df)*100:.1f}%)")
    print(f"Sign persists: {len(non_flips)} ({len(non_flips)/len(df)*100:.1f}%)")
    print()

    # Feature differences between flip and non-flip cases
    print("Feature comparison (Flip vs Non-flip):")
    print("-" * 70)

    features = [
        ('import_deviation_first', 'Import Dev (first)'),
        ('import_deviation_mean', 'Import Dev (mean)'),
        ('freq_deviation_mean', 'Freq Dev (mean)'),
        ('balancing_energy_mw_first', 'Regulation (first)'),
        ('prev_actual', 'Previous imbalance'),
    ]

    for col, name in features:
        if col in df.columns:
            flip_mean = flips[col].abs().mean()
            non_flip_mean = non_flips[col].abs().mean()

            if len(flips) > 2 and len(non_flips) > 2:
                t_stat, p_val = stats.ttest_ind(
                    flips[col].abs().dropna(),
                    non_flips[col].abs().dropna()
                )
                sig = '*' if p_val < 0.05 else ''
            else:
                sig = ''

            print(f"|{name:25s}|: Flip={flip_mean:8.2f}, NoFlip={non_flip_mean:8.2f} {sig}")

    print()

    # Can features predict flip direction?
    print("When sign FLIPS, can features predict the NEW sign?")
    print("-" * 70)

    if len(flips) > 0:
        # Import deviation predicting new sign
        import_correct = (flips['import_dev_sign'] == flips['actual_sign']).mean()
        freq_correct = (flips['freq_dev_sign'] == flips['actual_sign']).mean()
        model_correct = (flips['model_pred_sign'] == flips['actual_sign']).mean()
        persist_correct = (flips['prev_sign'] == flips['actual_sign']).mean()  # Always 0 for flips

        print(f"Model accuracy on flips:      {model_correct*100:5.1f}%")
        print(f"Import Dev accuracy on flips: {import_correct*100:5.1f}%")
        print(f"Frequency accuracy on flips:  {freq_correct*100:5.1f}%")
        print(f"Persistence (always wrong):   {persist_correct*100:5.1f}%")

    print()

    # Flip prediction model
    print("Logistic regression: Can we predict WHEN a flip will occur?")
    print("-" * 70)

    # Simple features for flip prediction
    df['abs_prev'] = df['prev_actual'].abs()
    df['abs_import_dev'] = df['import_deviation_first'].abs()

    # Correlation with flip
    for col, name in [('abs_prev', '|Previous imbalance|'),
                      ('abs_import_dev', '|Import deviation|'),
                      ('freq_deviation_mean', 'Frequency deviation')]:
        if col in df.columns:
            corr = df[[col, 'sign_flip']].dropna()
            if len(corr) > 5:
                r, p = stats.pointbiserialr(corr['sign_flip'], corr[col])
                sig = '*' if p < 0.05 else ''
                print(f"Correlation of {name:25s} with flip: r={r:+.3f} {sig}")


def analysis_3_agreement(df):
    """Analyze when model and new features agree/disagree."""
    print()
    print("=" * 70)
    print("ANALYSIS 3: AGREEMENT ANALYSIS")
    print("=" * 70)
    print()

    # Model vs Import Deviation
    df['model_import_agree'] = df['model_pred_sign'] == df['import_dev_sign']
    df['model_freq_agree'] = df['model_pred_sign'] == df['freq_dev_sign']

    print("Agreement rates:")
    print(f"  Model & Import Dev agree: {df['model_import_agree'].mean()*100:.1f}%")
    print(f"  Model & Frequency agree:  {df['model_freq_agree'].mean()*100:.1f}%")
    print()

    # When they agree vs disagree
    print("Accuracy when Model & Import Deviation AGREE:")
    agree = df[df['model_import_agree']]
    if len(agree) > 0:
        agree_acc = agree['sign_correct'].mean()
        print(f"  n={len(agree)}, Model accuracy: {agree_acc*100:.1f}%")

    print("\nAccuracy when Model & Import Deviation DISAGREE:")
    disagree = df[~df['model_import_agree']]
    if len(disagree) > 0:
        model_acc = disagree['sign_correct'].mean()
        import_acc = (disagree['import_dev_sign'] == disagree['actual_sign']).mean()
        print(f"  n={len(disagree)}")
        print(f"  Model accuracy:      {model_acc*100:.1f}%")
        print(f"  Import Dev accuracy: {import_acc*100:.1f}%")

        if import_acc > model_acc:
            print("  => Import Dev is MORE reliable when they disagree!")
        else:
            print("  => Model is MORE reliable when they disagree")

    print()

    # Voting ensemble
    print("Ensemble strategies:")
    print("-" * 70)

    # Simple majority vote (model, import, freq)
    df['vote_sum'] = (
        df['model_pred_sign'] +
        df['import_dev_sign'] +
        df['freq_dev_sign']
    )
    df['majority_sign'] = np.sign(df['vote_sum'])
    # Handle ties (vote_sum = 0) - default to model
    df.loc[df['vote_sum'] == 0, 'majority_sign'] = df.loc[df['vote_sum'] == 0, 'model_pred_sign']

    majority_acc = (df['majority_sign'] == df['actual_sign']).mean()
    model_acc = df['sign_correct'].mean()

    print(f"Model alone:            {model_acc*100:.1f}%")
    print(f"Majority vote (3 feat): {majority_acc*100:.1f}%")

    # Conditional ensemble: use import_dev only when model uncertain
    df['model_uncertain'] = df['prediction_mwh'].abs() < 3
    df['conditional_sign'] = np.where(
        df['model_uncertain'],
        df['import_dev_sign'],
        df['model_pred_sign']
    )
    conditional_acc = (df['conditional_sign'] == df['actual_sign']).mean()
    print(f"Conditional (import when |pred|<3): {conditional_acc*100:.1f}%")


def analysis_4_regulation_prediction(df):
    """Analyze whether new features can predict full-period regulation average."""
    print()
    print("=" * 70)
    print("ANALYSIS 4: REGULATION AVERAGE PREDICTION")
    print("=" * 70)
    print()

    print("Target: Full-period regulation mean (from Lead 0)")
    print("Question: Can new features at minute 3 predict final regulation average?")
    print()

    # We have reg_cumulative_mean from Lead 12 (first obs) and reg_full_period_mean (all 5 obs)
    df['reg_first'] = df['reg_cumulative_mean']  # From Lead 12 (1 obs)
    df['reg_final'] = df['reg_full_period_mean']  # From Lead 0 (5 obs mean)

    # How well does first observation predict the final mean?
    valid = df.dropna(subset=['reg_first', 'reg_final'])

    if len(valid) > 5:
        r_first, p = stats.pearsonr(valid['reg_first'], valid['reg_final'])
        print(f"First obs regulation vs Final mean: r = {r_first:.3f}")

        # Can import deviation at start predict change in regulation?
        valid['reg_change'] = valid['reg_final'] - valid['reg_first']

        print(f"\nRegulation change (final - first):")
        print(f"  Mean: {valid['reg_change'].mean():+.2f} MW")
        print(f"  Std:  {valid['reg_change'].std():.2f} MW")
        print()

        # Correlations with regulation change
        print("Can features predict regulation CHANGE during period?")
        print("-" * 70)

        features = [
            ('import_deviation_first', 'Import Dev (first)'),
            ('freq_deviation_mean', 'Freq Dev (mean)'),
            ('generation_mw_first', 'Generation (first)'),
            ('load_mw_first', 'Load (first)'),
        ]

        for col, name in features:
            if col in valid.columns:
                subset = valid[[col, 'reg_change']].dropna()
                if len(subset) > 5:
                    r, p = stats.pearsonr(subset[col], subset['reg_change'])
                    sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
                    print(f"  {name:25s} vs reg_change: r = {r:+.3f} {sig}")

        print()

        # Can we predict sign of final regulation from early features?
        print("Predicting SIGN of final regulation average:")
        print("-" * 70)

        valid['final_reg_sign'] = np.sign(valid['reg_final'])
        valid['first_reg_sign'] = np.sign(valid['reg_first'])

        first_correct = (valid['first_reg_sign'] == valid['final_reg_sign']).mean()
        import_correct = (valid['import_dev_sign'] == -valid['final_reg_sign']).mean()  # Note: opposite sign expected

        print(f"First reg obs predicts final sign: {first_correct*100:.1f}%")
        print(f"Import Dev predicts final sign:    {import_correct*100:.1f}%")


def main():
    print("=" * 70)
    print("PRODUCTION FEATURE ANALYSIS")
    print("New Features for Imbalance Direction Prediction")
    print("=" * 70)
    print()

    # Load data
    print("[*] Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"    Loaded {len(df)} settlement periods with all features")
    print()

    # Run analyses
    error_results = analysis_1_error_cases(df)
    analysis_2_sign_flip_prediction(df)
    analysis_3_agreement(df)
    analysis_4_regulation_prediction(df)

    # Save results
    df.to_csv(OUTPUT_DIR / 'analysis_dataset.csv', index=False)
    error_results.to_csv(OUTPUT_DIR / 'error_case_comparison.csv', index=False)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")
    print()

    return df


if __name__ == "__main__":
    df = main()
