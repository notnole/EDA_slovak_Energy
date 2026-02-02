"""
Alternative Baseline Model Visualizations

Try different ways to show model behavior and error patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\baseline")


def load_and_prepare():
    """Load data and compute predictions at all lead times."""
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )

    # Create QH observations
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    pivot = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot.columns[1:]]

    # Merge with labels
    df = pd.merge(label_df, pivot, on='datetime')

    # Compute predictions at each lead time
    df['pred_12'] = -0.25 * df['reg_min0']
    df['pred_9'] = -0.25 * (0.8 * df['reg_min3'] + 0.2 * df['reg_min0'])
    df['pred_6'] = -0.25 * (0.6 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
    df['pred_3'] = -0.25 * (0.4 * df['reg_min9'] + 0.2 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
    df['pred_0'] = -0.25 * (df['reg_min0'] + df['reg_min3'] + df['reg_min6'] + df['reg_min9'] + df['reg_min12']) / 5

    # Errors
    for lead in [12, 9, 6, 3, 0]:
        df[f'error_{lead}'] = df['imbalance'] - df[f'pred_{lead}']

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date

    return df.dropna()


def plot_prediction_trajectory(df):
    """
    Show how predictions evolve for specific settlement periods.
    Pick interesting cases: large positive, large negative, reversal.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Prediction Trajectory: How Baseline Evolves as Lead Time Decreases',
                 fontsize=14, fontweight='bold')

    lead_times = [12, 9, 6, 3, 0]  # Lead time = time to end of settlement

    # Find interesting cases
    # Large positive imbalance
    large_pos = df.nlargest(100, 'imbalance').sample(1, random_state=42).iloc[0]
    # Large negative imbalance
    large_neg = df.nsmallest(100, 'imbalance').sample(1, random_state=43).iloc[0]
    # Reversal case (prediction changes sign)
    df['sign_change'] = (df['pred_12'] * df['pred_0']) < 0
    reversals = df[df['sign_change']]
    if len(reversals) > 0:
        reversal = reversals.sample(1, random_state=44).iloc[0]
    else:
        reversal = df.sample(1, random_state=44).iloc[0]
    # Small imbalance
    small = df[df['imbalance'].abs() < 2].sample(1, random_state=45).iloc[0]
    # Typical
    typical = df[(df['imbalance'].abs() > 5) & (df['imbalance'].abs() < 15)].sample(1, random_state=46).iloc[0]
    # High error case
    df['max_error'] = df[[f'error_{l}' for l in lead_times]].abs().max(axis=1)
    high_error = df.nlargest(100, 'max_error').sample(1, random_state=47).iloc[0]

    cases = [
        (large_pos, 'Large Positive Imbalance', 0, 0),
        (large_neg, 'Large Negative Imbalance', 0, 1),
        (reversal, 'Prediction Reversal', 0, 2),
        (small, 'Small Imbalance', 1, 0),
        (typical, 'Typical Case', 1, 1),
        (high_error, 'High Error Case', 1, 2),
    ]

    for case, title, row, col in cases:
        ax = axes[row, col]

        # Actual imbalance (horizontal line)
        actual = case['imbalance']
        ax.axhline(actual, color='black', linewidth=2, label=f'Actual: {actual:.1f}')

        # Predictions at each lead time
        preds = [case[f'pred_{l}'] for l in lead_times]
        ax.plot(lead_times, preds, 'o-', color='tab:blue', markersize=10, linewidth=2, label='Prediction')

        # Fill area between prediction and actual
        ax.fill_between(lead_times, preds, [actual]*len(lead_times), alpha=0.3, color='tab:red')

        # Zero line
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Lead Time (min)')
        ax.set_ylabel('MWh')
        ax.set_title(f'{title}\n{case["datetime"]}')
        ax.set_xticks(lead_times)
        ax.invert_xaxis()  # Higher lead time on left
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz_prediction_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_prediction_trajectory.png")


def plot_error_heatmap(df):
    """
    Heatmap of MAE by hour and lead time.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute MAE by hour and lead time
    lead_times = [12, 9, 6, 3, 0]
    hours = range(24)

    mae_matrix = np.zeros((24, len(lead_times)))
    for i, hour in enumerate(hours):
        hour_data = df[df['hour'] == hour]
        for j, lead in enumerate(lead_times):
            mae_matrix[i, j] = hour_data[f'error_{lead}'].abs().mean()

    im = ax.imshow(mae_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f'{l} min' for l in lead_times])
    ax.set_yticks(range(24))
    ax.set_yticklabels(hours)
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('Hour of Day')
    ax.set_title('MAE Heatmap: When is the Baseline Weakest?')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE (MWh)')

    # Add text annotations
    for i in range(24):
        for j in range(len(lead_times)):
            text = ax.text(j, i, f'{mae_matrix[i, j]:.1f}',
                          ha='center', va='center', fontsize=7,
                          color='white' if mae_matrix[i, j] > 5 else 'black')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz_error_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_error_heatmap.png")


def plot_convergence_funnel(df):
    """
    Show how prediction uncertainty narrows as lead time decreases.
    Like a funnel plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    lead_times = [12, 9, 6, 3, 0]

    # Compute error percentiles at each lead time
    percentiles = [5, 25, 50, 75, 95]

    error_stats = {p: [] for p in percentiles}
    for lead in lead_times:
        errors = df[f'error_{lead}'].dropna()
        for p in percentiles:
            error_stats[p].append(np.percentile(errors, p))

    # Plot funnel
    ax.fill_between(lead_times, error_stats[5], error_stats[95], alpha=0.2, color='tab:blue', label='5-95%')
    ax.fill_between(lead_times, error_stats[25], error_stats[75], alpha=0.4, color='tab:blue', label='25-75%')
    ax.plot(lead_times, error_stats[50], 'o-', color='tab:blue', linewidth=2, markersize=8, label='Median')

    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Lead Time (min)', fontsize=12)
    ax.set_ylabel('Prediction Error (MWh)', fontsize=12)
    ax.set_title('Error Convergence Funnel: Uncertainty Narrows as Lead Time Decreases', fontsize=14, fontweight='bold')
    ax.set_xticks(lead_times)
    ax.invert_xaxis()  # Higher lead time on left
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations
    for i, lead in enumerate(lead_times):
        p95 = error_stats[95][i]
        p5 = error_stats[5][i]
        ax.annotate(f'+-{(p95-p5)/2:.1f}', xy=(lead, p95), xytext=(lead, p95+2),
                   ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz_convergence_funnel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_convergence_funnel.png")


def plot_daily_pattern(df):
    """
    Show a full day of predictions vs actuals.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Full Day View: Baseline Predictions vs Actual', fontsize=14, fontweight='bold')

    # Pick a sample day
    sample_date = pd.Timestamp('2025-11-12')
    day_data = df[df['datetime'].dt.date == sample_date.date()].sort_values('datetime')

    if len(day_data) < 10:
        sample_date = df['datetime'].dt.date.value_counts().index[0]
        day_data = df[df['datetime'].dt.date == sample_date].sort_values('datetime')

    x = day_data['datetime']

    # Plot 1: Actual vs Lead 12 prediction
    ax = axes[0]
    ax.plot(x, day_data['imbalance'], 'b-', linewidth=2, label='Actual')
    ax.plot(x, day_data['pred_12'], 'r--', linewidth=1.5, label='Pred (12 min lead)')
    ax.fill_between(x, day_data['imbalance'], day_data['pred_12'], alpha=0.3, color='red')
    ax.set_ylabel('MWh')
    ax.set_title('Lead 12 min: Early Prediction (R² = 0.36)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)

    # Plot 2: Actual vs Lead 3 prediction
    ax = axes[1]
    ax.plot(x, day_data['imbalance'], 'b-', linewidth=2, label='Actual')
    ax.plot(x, day_data['pred_3'], 'g--', linewidth=1.5, label='Pred (3 min lead)')
    ax.fill_between(x, day_data['imbalance'], day_data['pred_3'], alpha=0.3, color='green')
    ax.set_ylabel('MWh')
    ax.set_title('Lead 3 min: Near-Measurement (R² = 0.76)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)

    # Plot 3: Errors over the day
    ax = axes[2]
    ax.bar(x, day_data['error_12'], width=0.008, alpha=0.7, color='red', label='Error (12 min)')
    ax.bar(x, day_data['error_3'], width=0.005, alpha=0.7, color='green', label='Error (3 min)')
    ax.set_ylabel('Error (MWh)')
    ax.set_xlabel('Time')
    ax.set_title('Prediction Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz_daily_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_daily_pattern.png")


def plot_error_by_magnitude(df):
    """
    Show how error scales with imbalance magnitude.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Error vs Imbalance Magnitude: Where Does Baseline Struggle?',
                 fontsize=14, fontweight='bold')

    # Bin by absolute imbalance
    df['imb_abs'] = df['imbalance'].abs()
    bins = [0, 2, 5, 10, 20, 50, 200]
    labels = ['0-2', '2-5', '5-10', '10-20', '20-50', '50+']
    df['imb_bin'] = pd.cut(df['imb_abs'], bins=bins, labels=labels)

    # Plot 1: MAE by magnitude bin
    ax = axes[0]
    for lead, color in [(12, 'tab:red'), (3, 'tab:green')]:
        mae_by_bin = df.groupby('imb_bin', observed=True)[f'error_{lead}'].apply(lambda x: x.abs().mean())
        ax.plot(labels, mae_by_bin.values, 'o-', color=color, markersize=10, linewidth=2,
                label=f'Lead {lead} min')
    ax.set_xlabel('Actual |Imbalance| (MWh)')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Imbalance Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Relative error (MAPE-like)
    ax = axes[1]
    for lead, color in [(12, 'tab:red'), (3, 'tab:green')]:
        # Avoid division by zero for small imbalances
        df_nonzero = df[df['imb_abs'] > 1]
        df_nonzero['rel_error'] = df_nonzero[f'error_{lead}'].abs() / df_nonzero['imb_abs'] * 100
        rel_by_bin = df_nonzero.groupby('imb_bin', observed=True)['rel_error'].median()
        ax.plot(labels, rel_by_bin.values, 's-', color=color, markersize=10, linewidth=2,
                label=f'Lead {lead} min')
    ax.set_xlabel('Actual |Imbalance| (MWh)')
    ax.set_ylabel('Median Relative Error (%)')
    ax.set_title('Relative Error by Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viz_error_by_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: viz_error_by_magnitude.png")


def main():
    print("=" * 70)
    print("ALTERNATIVE BASELINE VISUALIZATIONS")
    print("=" * 70)

    df = load_and_prepare()
    print(f"\nLoaded {len(df):,} settlement periods")

    print("\nGenerating visualizations...")
    plot_prediction_trajectory(df)
    plot_error_heatmap(df)
    plot_convergence_funnel(df)
    plot_daily_pattern(df)
    plot_error_by_magnitude(df)

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
