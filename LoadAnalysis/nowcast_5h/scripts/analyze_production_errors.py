"""
Analyze prediction errors from the tuned single-stage models.
Generates predictions, error statistics, and diagnostic plots.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent  # ipesoft_eda_data
DATA_PATH = PROJECT_ROOT / 'features' / 'DamasLoad' / 'load_data.parquet'
MODELS_DIR = BASE_DIR / 'models_tuned'
OUTPUT_DIR = BASE_DIR / 'error_analysis'

def load_data():
    """Load the feature data."""
    print("[*] Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    # Create error column
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    # Rename columns for consistency
    df = df.rename(columns={
        'actual_load_mw': 'actual_load',
        'forecast_load_mw': 'forecast_load'
    })
    print(f"    Loaded {len(df):,} records")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df

def create_features(df, horizon, params, features_list):
    """Create features for a given horizon using the tuned parameters."""
    df = df.copy()

    # Basic error lags - use current error shifted by lag
    max_lag = params.get('max_error_lag', 24)
    for lag in range(1, max_lag + 1):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling features - shift by 1 (last known error)
    error_col = 'error'
    base_shift = 1

    if params.get('use_roll_3h', False):
        df['error_roll_mean_3h'] = df[error_col].shift(base_shift).rolling(3).mean()
        df['error_roll_std_3h'] = df[error_col].shift(base_shift).rolling(3).std()

    if params.get('use_roll_6h', False):
        df['error_roll_mean_6h'] = df[error_col].shift(base_shift).rolling(6).mean()
        df['error_roll_std_6h'] = df[error_col].shift(base_shift).rolling(6).std()

    if params.get('use_roll_12h', False):
        df['error_roll_mean_12h'] = df[error_col].shift(base_shift).rolling(12).mean()
        df['error_roll_std_12h'] = df[error_col].shift(base_shift).rolling(12).std()

    if params.get('use_roll_24h', False):
        df['error_roll_mean_24h'] = df[error_col].shift(base_shift).rolling(24).mean()
        df['error_roll_std_24h'] = df[error_col].shift(base_shift).rolling(24).std()

    # Error trends
    if params.get('use_error_trends', False):
        df['error_trend_3h'] = df[f'error_lag1'] - df[f'error_lag3'] if 'error_lag3' in df.columns else 0
        df['error_trend_6h'] = df[f'error_lag1'] - df[f'error_lag6'] if 'error_lag6' in df.columns else 0

    # Momentum
    if params.get('use_momentum', False):
        w1 = params.get('momentum_w1', 0.5)
        w2 = params.get('momentum_w2', 0.3)
        df['error_momentum'] = w1 * df.get('error_lag1', 0) + w2 * df.get('error_lag2', 0)

    # Load features - shift by 1 (last known load)
    if params.get('use_load_volatility', False):
        df['load_volatility_lag1'] = df['actual_load'].shift(1).rolling(6).std()

    if params.get('use_load_trend', False):
        df['load_trend_lag1'] = df['actual_load'].shift(1) - df['actual_load'].shift(4)

    if params.get('use_load_range', False):
        df['load_range_lag1'] = df['actual_load'].shift(1).rolling(6).max() - df['actual_load'].shift(1).rolling(6).min()

    # Regulation features (dummy - will be filled with zeros if regulation data not available)
    if 'regulation' not in df.columns:
        df['regulation'] = 0  # Placeholder
    reg_depth = params.get('reg_lag_depth', 3)
    for lag in range(1, reg_depth + 1):
        if 'regulation' in df.columns and df['regulation'].notna().any():
            df[f'reg_mean_lag{lag}'] = df['regulation'].shift(lag).rolling(3).mean()
        else:
            df[f'reg_mean_lag{lag}'] = 0

    if params.get('use_reg_std', False):
        df['reg_std_lag1'] = df['regulation'].shift(1).rolling(6).std() if df['regulation'].notna().any() else 0

    if params.get('use_reg_range', False):
        if df['regulation'].notna().any():
            df['reg_range_lag1'] = df['regulation'].shift(1).rolling(6).max() - df['regulation'].shift(1).rolling(6).min()
        else:
            df['reg_range_lag1'] = 0

    # Seasonal error - error from 24h ago
    if params.get('use_seasonal_error', False):
        df['seasonal_error'] = df[error_col].shift(24)

    # Time features - already exist in df, just ensure they exist
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour
    if 'dow' not in df.columns:
        df['dow'] = df['datetime'].dt.dayofweek

    if params.get('use_hour_cyclical', False):
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    if params.get('use_dow_cyclical', False):
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    if params.get('use_is_weekend', False):
        df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Same-hour features - error from same hour yesterday (24h ago)
    if params.get('use_same_hour_yesterday', False):
        df['error_same_hour_yesterday'] = df[error_col].shift(24)

    if params.get('use_same_hour_2d', False):
        df['error_same_hour_2d'] = df[error_col].shift(48)

    # Forecast features
    if params.get('use_forecast_level', False):
        df['forecast_load'] = df['forecast_load']

    if params.get('use_forecast_diff', False):
        df['forecast_diff_1h'] = df['forecast_load'] - df['forecast_load'].shift(1)
        df['forecast_diff_24h'] = df['forecast_load'] - df['forecast_load'].shift(24)

    # Error sign features
    if params.get('use_error_sign', False):
        df['error_lag1_sign'] = np.sign(df.get('error_lag1', 0))
        sign_series = np.sign(df.get('error_lag1', 0))
        streak = sign_series.groupby((sign_series != sign_series.shift()).cumsum()).cumcount() + 1
        df['error_sign_streak'] = streak * sign_series

    return df

def generate_predictions(df, horizon):
    """Generate predictions for a horizon using the tuned model."""
    # Load model (saved directly, not as bundle)
    model_path = MODELS_DIR / f'q0_h{horizon}_tuned.joblib'
    model = joblib.load(model_path)

    # Load params and features from separate files
    with open(MODELS_DIR / 'q0_best_params.json') as f:
        all_params = json.load(f)
    params = all_params[str(horizon)]

    with open(MODELS_DIR / 'q0_feature_lists.json') as f:
        all_features = json.load(f)
    features = all_features[str(horizon)]

    # Create features
    df_feat = create_features(df, horizon, params, features)

    # Target is the error shifted forward by horizon hours (we predict future error)
    target_col = 'target_error'
    df_feat[target_col] = df_feat['error'].shift(-horizon)

    # Filter to rows where all features and target are available
    available_features = [f for f in features if f in df_feat.columns]
    missing_features = [f for f in features if f not in df_feat.columns]
    if missing_features:
        print(f"    Warning: Missing {len(missing_features)} features: {missing_features[:5]}...")
        # Create missing features as zeros
        for f in missing_features:
            df_feat[f] = 0
        available_features = features

    valid_mask = df_feat[features + [target_col]].notna().all(axis=1)
    df_valid = df_feat[valid_mask].copy()

    print(f"    Valid rows: {len(df_valid):,} / {len(df_feat):,}")

    # Predict
    X = df_valid[features]
    y_true = df_valid[target_col]
    y_pred = model.predict(X)

    # Create results dataframe
    results = pd.DataFrame({
        'datetime': df_valid['datetime'].values,
        'actual_load': df_valid['actual_load'].values,
        'forecast_load': df_valid['forecast_load'].values,
        'actual_error': y_true.values,
        'predicted_error': y_pred,
        'prediction_error': y_true.values - y_pred,  # residual
        'abs_error': np.abs(y_true.values - y_pred),
        'hour': df_valid['hour'].values,
        'dow': df_valid['dow'].values,
        'month': df_valid['datetime'].dt.month.values,
    })

    return results

def analyze_errors(results, horizon):
    """Analyze prediction errors."""
    print(f"\n{'='*60}")
    print(f"H+{horizon} Error Analysis")
    print('='*60)

    mae = results['abs_error'].mean()
    rmse = np.sqrt((results['prediction_error']**2).mean())
    bias = results['prediction_error'].mean()

    print(f"\nOverall Statistics:")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  Bias: {bias:+.2f} MW")
    print(f"  Std:  {results['prediction_error'].std():.2f} MW")

    # Error distribution
    print(f"\nError Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(results['abs_error'], p)
        print(f"  P{p}: {val:.1f} MW")

    # By hour
    print(f"\nMAE by Hour (Top 5 worst):")
    hourly = results.groupby('hour')['abs_error'].mean().sort_values(ascending=False)
    for hour, mae_h in hourly.head(5).items():
        print(f"  Hour {hour:02d}: {mae_h:.1f} MW")

    # By day of week
    print(f"\nMAE by Day of Week:")
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = results.groupby('dow')['abs_error'].mean()
    for dow, mae_d in daily.items():
        print(f"  {dow_names[dow]}: {mae_d:.1f} MW")

    # Large errors
    large_threshold = mae * 2
    large_errors = results[results['abs_error'] > large_threshold]
    print(f"\nLarge Errors (>{large_threshold:.0f} MW): {len(large_errors)} ({100*len(large_errors)/len(results):.1f}%)")

    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'hourly_mae': hourly.to_dict(),
        'daily_mae': daily.to_dict(),
        'large_error_pct': 100*len(large_errors)/len(results)
    }

def create_plots(all_results, output_dir):
    """Create diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: MAE by horizon
    fig, ax = plt.subplots(figsize=(10, 6))
    horizons = list(all_results.keys())
    maes = [all_results[h]['stats']['mae'] for h in horizons]
    ax.bar([f'H+{h}' for h in horizons], maes, color='steelblue')
    ax.set_ylabel('MAE (MW)')
    ax.set_title('Prediction Error by Horizon')
    for i, (h, m) in enumerate(zip(horizons, maes)):
        ax.text(i, m + 1, f'{m:.1f}', ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / '01_mae_by_horizon.png', dpi=150)
    plt.close()

    # Plot 2: Error distribution by horizon
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, h in enumerate(horizons):
        ax = axes[i]
        errors = all_results[h]['predictions']['prediction_error']
        ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Prediction Error (MW)')
        ax.set_ylabel('Count')
        ax.set_title(f'H+{h} Error Distribution\nMAE={all_results[h]["stats"]["mae"]:.1f} MW')
    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / '02_error_distributions.png', dpi=150)
    plt.close()

    # Plot 3: MAE by hour (heatmap style)
    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_data = []
    for h in horizons:
        hourly_mae = all_results[h]['predictions'].groupby('hour')['abs_error'].mean()
        hourly_data.append(hourly_mae.values)
    hourly_matrix = np.array(hourly_data)

    im = ax.imshow(hourly_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'H+{h}' for h in horizons])
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Horizon')
    ax.set_title('MAE by Hour and Horizon')
    plt.colorbar(im, label='MAE (MW)')
    plt.tight_layout()
    plt.savefig(output_dir / '03_mae_by_hour_heatmap.png', dpi=150)
    plt.close()

    # Plot 4: Actual vs Predicted scatter for each horizon
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, h in enumerate(horizons):
        ax = axes[i]
        pred = all_results[h]['predictions']
        ax.scatter(pred['actual_error'], pred['predicted_error'], alpha=0.1, s=1)

        # Add perfect prediction line
        min_val = min(pred['actual_error'].min(), pred['predicted_error'].min())
        max_val = max(pred['actual_error'].max(), pred['predicted_error'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect')

        ax.set_xlabel('Actual Error (MW)')
        ax.set_ylabel('Predicted Error (MW)')
        ax.set_title(f'H+{h}: Actual vs Predicted')
        ax.legend()
    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / '04_actual_vs_predicted.png', dpi=150)
    plt.close()

    # Plot 5: Time series of errors (last month)
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    for i, h in enumerate(horizons):
        ax = axes[i]
        pred = all_results[h]['predictions'].copy()
        pred = pred[pred['datetime'] >= pred['datetime'].max() - pd.Timedelta(days=30)]
        ax.plot(pred['datetime'], pred['prediction_error'], alpha=0.7, linewidth=0.5)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(pred['datetime'], pred['prediction_error'], 0, alpha=0.3)
        ax.set_ylabel(f'H+{h} Error')
        ax.set_title(f'H+{h} Prediction Errors (Last 30 Days)')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(output_dir / '05_error_timeseries_last30d.png', dpi=150)
    plt.close()

    # Plot 6: Cumulative error distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    for h in horizons:
        errors = np.sort(all_results[h]['predictions']['abs_error'])
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=f'H+{h}')
    ax.set_xlabel('Absolute Error (MW)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Error Distribution by Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    plt.tight_layout()
    plt.savefig(output_dir / '06_cumulative_error_distribution.png', dpi=150)
    plt.close()

    print(f"\n[+] Saved 6 diagnostic plots to {output_dir}")

def main():
    print("="*60)
    print("PRODUCTION MODEL ERROR ANALYSIS")
    print("="*60)

    # Load data
    df = load_data()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Generate predictions and analyze for each horizon
    for horizon in range(1, 6):
        print(f"\n[*] Processing H+{horizon}...")
        predictions = generate_predictions(df, horizon)
        stats = analyze_errors(predictions, horizon)

        all_results[horizon] = {
            'predictions': predictions,
            'stats': stats
        }

        # Save predictions
        predictions.to_csv(OUTPUT_DIR / f'h{horizon}_predictions.csv', index=False)

    # Create plots
    create_plots(all_results, OUTPUT_DIR)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n| Horizon | MAE (MW) | RMSE (MW) | Bias (MW) | Large Err % |")
    print("|---------|----------|-----------|-----------|-------------|")
    for h in range(1, 6):
        s = all_results[h]['stats']
        print(f"| H+{h}     | {s['mae']:.2f}    | {s['rmse']:.2f}     | {s['bias']:+.2f}     | {s['large_error_pct']:.1f}%        |")

    # Save summary
    summary = {h: all_results[h]['stats'] for h in range(1, 6)}
    with open(OUTPUT_DIR / 'error_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[+] Results saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
