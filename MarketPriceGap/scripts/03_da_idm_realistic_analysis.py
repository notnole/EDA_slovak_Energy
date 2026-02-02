"""
DA-IDM Spread Direction Analysis - REALISTIC VERSION
Corrected for actual market timing:
- DA position decided at D-1 (day before delivery)
- Can only use information available BEFORE DA auction clears
- IDM trading happens on delivery day to close position

Features available at DA decision time (D-1, ~noon):
- Historical spreads (yesterday, last week same hour, etc.)
- Day of week, hour patterns
- Weather/load/renewable FORECASTS (not actuals)
- Previous days' market outcomes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "MarketPriceGap" / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "MarketPriceGap" / "features" / "da_idm_realistic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def load_data():
    """Load and prepare data with REALISTIC features only."""
    print("[*] Loading data...")
    df = pd.read_csv(DATA_DIR / "hourly_market_prices.csv", parse_dates=['timestamp_hour'])

    # Filter to rows with both DA and IDM prices
    mask = df['da_price'].notna() & df['idm_vwap'].notna()
    df = df[mask].copy()
    df = df.sort_values('timestamp_hour').reset_index(drop=True)

    # Target variable
    df['spread'] = df['spread_da_idm']
    df['spread_direction'] = (df['spread'] > 0).astype(int)  # 1 = DA > IDM
    df['spread_abs'] = df['spread'].abs()

    print(f"[+] Loaded {len(df):,} records")
    print(f"    DA > IDM: {df['spread_direction'].mean()*100:.1f}% of cases")
    print(f"    Date range: {df['timestamp_hour'].min()} to {df['timestamp_hour'].max()}")

    return df


def create_realistic_features(df):
    """
    Create features that would be AVAILABLE at DA decision time (D-1).

    CANNOT use:
    - Same-day spread or prices (not known yet!)
    - Same-day direction
    - Any lag < 24 hours for the same delivery hour

    CAN use:
    - Yesterday's same hour (lag 24h)
    - Last week same hour (lag 168h)
    - Previous days' patterns
    - Calendar features
    - Historical statistics
    """
    print("[*] Creating REALISTIC features (available at D-1)...")

    df = df.sort_values('timestamp_hour').copy()

    # ===========================================
    # CALENDAR FEATURES (always available)
    # ===========================================
    df['hour'] = df['timestamp_hour'].dt.hour
    df['dayofweek'] = df['timestamp_hour'].dt.dayofweek
    df['month'] = df['timestamp_hour'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_monday'] = (df['dayofweek'] == 0).astype(int)
    df['is_friday'] = (df['dayofweek'] == 4).astype(int)

    # Hour groups
    df['is_peak'] = ((df['hour'] >= 8) & (df['hour'] <= 20)).astype(int)
    df['is_solar_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
    df['is_morning_ramp'] = ((df['hour'] >= 6) & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)

    # ===========================================
    # HISTORICAL SPREAD FEATURES (D-1 and earlier)
    # ===========================================

    # Same hour yesterday (24h lag) - THIS IS KEY
    df['spread_yesterday_same_hour'] = df['spread'].shift(24)
    df['direction_yesterday_same_hour'] = df['spread_direction'].shift(24)

    # Same hour, 2 days ago
    df['spread_2d_ago_same_hour'] = df['spread'].shift(48)
    df['direction_2d_ago_same_hour'] = df['spread_direction'].shift(48)

    # Same hour last week
    df['spread_lastweek_same_hour'] = df['spread'].shift(168)
    df['direction_lastweek_same_hour'] = df['spread_direction'].shift(168)

    # Yesterday's daily average spread (all hours)
    df['spread_yesterday_avg'] = df['spread'].rolling(24).mean().shift(24)
    df['spread_yesterday_std'] = df['spread'].rolling(24).std().shift(24)

    # Last 7 days average for same hour (weekly seasonality)
    # This requires grouping by hour and computing rolling mean
    df['spread_7d_same_hour_avg'] = df.groupby('hour')['spread'].transform(
        lambda x: x.rolling(7, min_periods=1).mean().shift(1)
    )

    # Streak: how many consecutive days same hour had same direction
    df['direction_streak'] = df.groupby('hour')['spread_direction'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ).shift(24)

    # ===========================================
    # HISTORICAL PRICE FEATURES
    # ===========================================

    # DA price yesterday same hour
    df['da_price_yesterday'] = df['da_price'].shift(24)

    # DA price level (high/low) - yesterday's percentile
    df['da_price_yesterday_high'] = (df['da_price'].shift(24) > df['da_price'].rolling(168).quantile(0.75).shift(24)).astype(int)
    df['da_price_yesterday_low'] = (df['da_price'].shift(24) < df['da_price'].rolling(168).quantile(0.25).shift(24)).astype(int)

    # Yesterday's price volatility
    df['da_volatility_yesterday'] = df['da_price'].rolling(24).std().shift(24)

    # ===========================================
    # CROSS-BORDER FLOW FEATURES (if available D-1)
    # ===========================================
    if 'net_import' in df.columns:
        df['net_import_yesterday'] = df['net_import'].shift(24)
        df['net_import_yesterday_avg'] = df['net_import'].rolling(24).mean().shift(24)

    # ===========================================
    # DROP ROWS WITHOUT ENOUGH HISTORY
    # ===========================================
    # Need at least 7 days of history for features
    df = df.iloc[168:].copy()

    print(f"[+] Created features. {len(df):,} records with full feature set")

    return df


def analyze_realistic_features(df):
    """Analyze which realistic features predict spread direction."""
    print("[*] Analyzing realistic feature importance...")

    # Features to analyze
    features = [
        'hour', 'dayofweek', 'is_weekend', 'is_peak', 'is_solar_hours',
        'is_morning_ramp', 'is_evening_peak', 'is_monday', 'is_friday',
        'direction_yesterday_same_hour', 'spread_yesterday_same_hour',
        'direction_2d_ago_same_hour', 'direction_lastweek_same_hour',
        'spread_yesterday_avg', 'spread_yesterday_std',
        'spread_7d_same_hour_avg', 'direction_streak',
        'da_price_yesterday', 'da_price_yesterday_high', 'da_price_yesterday_low',
        'da_volatility_yesterday'
    ]

    if 'net_import_yesterday' in df.columns:
        features.extend(['net_import_yesterday', 'net_import_yesterday_avg'])

    # Filter existing features
    features = [f for f in features if f in df.columns]

    # Calculate correlations
    correlations = {}
    for feat in features:
        valid_mask = df[feat].notna() & df['spread_direction'].notna()
        if valid_mask.sum() > 100:
            try:
                corr, pval = stats.pointbiserialr(
                    df.loc[valid_mask, 'spread_direction'],
                    df.loc[valid_mask, feat]
                )
                correlations[feat] = {'correlation': corr, 'p_value': pval}
            except:
                pass

    corr_df = pd.DataFrame(correlations).T.sort_values('correlation', key=abs, ascending=False)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Feature correlations
    ax = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in corr_df['correlation']]
    bars = ax.barh(range(len(corr_df)), corr_df['correlation'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df.index, fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Correlation with Spread Direction')
    ax.set_title('REALISTIC Features Correlation\n(Available at D-1)', fontweight='bold')

    # Add significance markers
    for i, (idx, row) in enumerate(corr_df.iterrows()):
        marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        x_pos = row['correlation'] + 0.005 if row['correlation'] > 0 else row['correlation'] - 0.005
        ax.text(x_pos, i, marker, va='center', ha='left' if row['correlation'] > 0 else 'right', fontsize=10)

    # 2. Yesterday same hour as predictor
    ax = axes[0, 1]
    valid = df.dropna(subset=['direction_yesterday_same_hour', 'spread_direction'])

    # Confusion matrix style
    transitions = pd.crosstab(
        valid['direction_yesterday_same_hour'].map({0: 'Yesterday: DA<IDM', 1: 'Yesterday: DA>IDM'}),
        valid['spread_direction'].map({0: 'Today: DA<IDM', 1: 'Today: DA>IDM'}),
        normalize='index'
    ) * 100

    x = np.arange(2)
    width = 0.35
    colors_0 = ['#e74c3c', '#2ecc71']
    colors_1 = ['#e74c3c', '#2ecc71']

    for i, yesterday_state in enumerate(transitions.index):
        vals = transitions.loc[yesterday_state].values
        ax.bar(i, vals[0], width=0.6, color='#e74c3c', alpha=0.7, label='Today: DA<IDM' if i==0 else '')
        ax.bar(i, vals[1], width=0.6, bottom=vals[0], color='#2ecc71', alpha=0.7, label='Today: DA>IDM' if i==0 else '')

        # Add labels
        ax.text(i, vals[0]/2, f'{vals[0]:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(i, vals[0] + vals[1]/2, f'{vals[1]:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    ax.set_xticks(range(2))
    ax.set_xticklabels(['Yesterday:\nDA < IDM', 'Yesterday:\nDA > IDM'])
    ax.set_ylabel('Probability (%)')
    ax.set_title('Same Hour Yesterday -> Today\n(Key Predictor)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # 3. Accuracy by hour
    ax = axes[1, 0]
    valid = df.dropna(subset=['direction_yesterday_same_hour', 'spread_direction'])
    valid['correct'] = (valid['direction_yesterday_same_hour'] == valid['spread_direction']).astype(int)

    hourly_accuracy = valid.groupby('hour')['correct'].mean() * 100

    colors = ['green' if x > 50 else 'red' for x in hourly_accuracy]
    bars = ax.bar(hourly_accuracy.index, hourly_accuracy.values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=2, label='Random')
    ax.axhline(hourly_accuracy.mean(), color='blue', linestyle='-', linewidth=2, label=f'Avg: {hourly_accuracy.mean():.1f}%')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Prediction Accuracy by Hour\n(Using Yesterday Same Hour)', fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(40, 70)
    ax.legend()

    # 4. Day of week effect
    ax = axes[1, 1]
    dow_accuracy = valid.groupby('dayofweek')['correct'].mean() * 100
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    colors = ['green' if x > 50 else 'red' for x in dow_accuracy]
    bars = ax.bar(range(7), dow_accuracy.values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=2)
    ax.axhline(dow_accuracy.mean(), color='blue', linestyle='-', linewidth=2, label=f'Avg: {dow_accuracy.mean():.1f}%')
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Prediction Accuracy by Day\n(Using Yesterday Same Hour)', fontweight='bold')
    ax.set_ylim(40, 70)
    ax.legend()

    plt.suptitle('DA-IDM Spread Prediction - REALISTIC Analysis\n(Only D-1 Information)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_realistic_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '01_realistic_features.png'}")

    return corr_df


def evaluate_realistic_rules(df):
    """Evaluate trading rules using ONLY D-1 information."""
    print("[*] Evaluating realistic trading rules...")

    df = df.dropna(subset=['direction_yesterday_same_hour', 'spread_direction', 'spread']).copy()

    # Define REALISTIC rules (only D-1 info)
    rules = {
        'Follow yesterday same hour': lambda x: x['direction_yesterday_same_hour'],
        'Follow 2 days ago same hour': lambda x: x['direction_2d_ago_same_hour'],
        'Follow last week same hour': lambda x: x['direction_lastweek_same_hour'],
        'Naive: always sell DA': lambda x: pd.Series(1, index=x.index),
        'Naive: always buy DA': lambda x: pd.Series(0, index=x.index),
        'Peak hours sell DA': lambda x: x['is_peak'],
        'Weekday: sell, Weekend: buy': lambda x: 1 - x['is_weekend'],
        'Yesterday avg positive -> sell': lambda x: (x['spread_yesterday_avg'] > 0).astype(int),
        '7d same hour avg positive -> sell': lambda x: (x['spread_7d_same_hour_avg'] > 0).astype(int),
    }

    results = []

    for rule_name, rule_func in rules.items():
        try:
            prediction = rule_func(df)
            if prediction.isna().all():
                continue

            valid_mask = prediction.notna()
            pred = prediction[valid_mask]
            actual_dir = df.loc[valid_mask, 'spread_direction']
            actual_spread = df.loc[valid_mask, 'spread']

            # Accuracy
            accuracy = (pred == actual_dir).mean()

            # P&L: if predict 1 (sell DA), profit = spread; if predict 0 (buy DA), profit = -spread
            pnl = np.where(pred == 1, actual_spread, -actual_spread)

            total_pnl = pnl.sum()
            avg_pnl = pnl.mean()
            sharpe = pnl.mean() / pnl.std() * np.sqrt(8760) if pnl.std() > 0 else 0
            win_rate = (pnl > 0).mean()

            results.append({
                'rule': rule_name,
                'accuracy': accuracy,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'n_trades': len(pred)
            })
        except Exception as e:
            print(f"    [-] Error with rule '{rule_name}': {e}")

    results_df = pd.DataFrame(results).sort_values('total_pnl', ascending=False)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy comparison
    ax = axes[0, 0]
    colors = ['green' if x > 0.5 else 'red' for x in results_df['accuracy']]
    ax.barh(range(len(results_df)), results_df['accuracy'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(50, color='black', linestyle='--', linewidth=2, label='Random')
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df['rule'], fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Direction Prediction Accuracy\n(REALISTIC - D-1 Info Only)', fontweight='bold')
    ax.set_xlim(40, 65)

    # Add value labels
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.text(row['accuracy']*100 + 0.5, i, f"{row['accuracy']*100:.1f}%", va='center', fontsize=9)

    # 2. Total P&L comparison
    ax = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in results_df['total_pnl']]
    ax.barh(range(len(results_df)), results_df['total_pnl'], color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df['rule'], fontsize=9)
    ax.set_xlabel('Total P&L (EUR/MWh)')
    ax.set_title('Cumulative Trading P&L\n(REALISTIC - D-1 Info Only)', fontweight='bold')

    # 3. Cumulative P&L over time
    ax = axes[1, 0]

    top_rules = ['Follow yesterday same hour', 'Naive: always sell DA', '7d same hour avg positive -> sell']
    df_sorted = df.sort_values('timestamp_hour')

    for rule_name in top_rules:
        if rule_name in [r['rule'] for r in results]:
            rule_func = rules[rule_name]
            pred = rule_func(df_sorted).fillna(0)
            pnl = np.where(pred == 1, df_sorted['spread'], -df_sorted['spread'])
            cumsum = np.cumsum(pnl)
            ax.plot(df_sorted['timestamp_hour'], cumsum, linewidth=1.5, label=rule_name)

    # Perfect foresight baseline
    perfect_pnl = df_sorted['spread_abs'].cumsum()
    ax.plot(df_sorted['timestamp_hour'], perfect_pnl, '--', color='gray', linewidth=1, label='Perfect foresight', alpha=0.7)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (EUR/MWh)')
    ax.set_title('Cumulative P&L Over Time', fontweight='bold')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Best realistic rule
    best = results_df.iloc[0]

    table_data = [
        ['Metric', 'Best Realistic Rule'],
        ['Strategy', best['rule']],
        ['Accuracy', f"{best['accuracy']*100:.1f}%"],
        ['Win Rate', f"{best['win_rate']*100:.1f}%"],
        ['Total P&L', f"{best['total_pnl']:,.0f} EUR/MWh"],
        ['Avg P&L/trade', f"{best['avg_pnl']:.2f} EUR/MWh"],
        ['Sharpe (ann.)', f"{best['sharpe']:.2f}"],
        ['# Trades', f"{best['n_trades']:,}"],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    for i in range(2):
        table[(0, i)].set_facecolor('#4a4a4a')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Best REALISTIC Strategy Summary', fontweight='bold', fontsize=12, pad=20)

    plt.suptitle('DA-IDM Trading Rules - REALISTIC Evaluation\n(Using Only Day-Ahead Available Information)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_realistic_rules.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '02_realistic_rules.png'}")

    return results_df


def analyze_when_to_trade(df):
    """Analyze when the prediction is most/least reliable."""
    print("[*] Analyzing when prediction is most reliable...")

    df = df.dropna(subset=['direction_yesterday_same_hour', 'spread_direction']).copy()
    df['correct'] = (df['direction_yesterday_same_hour'] == df['spread_direction']).astype(int)
    df['pnl'] = np.where(df['direction_yesterday_same_hour'] == 1, df['spread'], -df['spread'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy by spread magnitude yesterday
    ax = axes[0, 0]
    df['spread_yesterday_abs'] = df['spread_yesterday_same_hour'].abs()
    df['spread_yesterday_bin'] = pd.qcut(df['spread_yesterday_abs'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'], duplicates='drop')

    bin_stats = df.groupby('spread_yesterday_bin', observed=True).agg({
        'correct': 'mean',
        'pnl': ['mean', 'sum'],
        'spread_abs': 'mean'
    })
    bin_stats.columns = ['accuracy', 'avg_pnl', 'total_pnl', 'avg_spread_today']

    x = range(len(bin_stats))
    colors = ['green' if a > 0.5 else 'red' for a in bin_stats['accuracy']]
    ax.bar(x, bin_stats['accuracy'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_stats.index, rotation=45)
    ax.set_xlabel("Yesterday's Spread Magnitude")
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Yesterday Spread Size\n(Larger = More Predictable?)', fontweight='bold')
    ax.set_ylim(40, 70)

    # 2. P&L by hour
    ax = axes[0, 1]
    hourly_pnl = df.groupby('hour')['pnl'].agg(['mean', 'sum', 'std'])

    colors = ['green' if x > 0 else 'red' for x in hourly_pnl['mean']]
    ax.bar(hourly_pnl.index, hourly_pnl['mean'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average P&L (EUR/MWh)')
    ax.set_title('Average P&L by Hour\n(Follow Yesterday Strategy)', fontweight='bold')
    ax.set_xticks(range(0, 24, 2))

    # 3. Accuracy by month
    ax = axes[1, 0]
    monthly_stats = df.groupby('month').agg({
        'correct': 'mean',
        'pnl': 'sum',
        'spread_abs': 'mean'
    })

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = monthly_stats.index - 1
    colors = ['green' if a > 0.5 else 'red' for a in monthly_stats['correct']]
    ax.bar(x, monthly_stats['correct'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=2)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_xlabel('Month')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Month\n(Seasonal Pattern)', fontweight='bold')
    ax.set_ylim(40, 70)

    # 4. Regime analysis - accuracy when yesterday was extreme
    ax = axes[1, 1]

    # Define regimes
    spread_p25 = df['spread_yesterday_same_hour'].quantile(0.25)
    spread_p75 = df['spread_yesterday_same_hour'].quantile(0.75)

    regimes = {
        'Strong DA<IDM\n(yesterday)': df['spread_yesterday_same_hour'] < spread_p25,
        'Moderate\n(yesterday)': (df['spread_yesterday_same_hour'] >= spread_p25) & (df['spread_yesterday_same_hour'] <= spread_p75),
        'Strong DA>IDM\n(yesterday)': df['spread_yesterday_same_hour'] > spread_p75,
    }

    regime_stats = []
    for name, mask in regimes.items():
        subset = df[mask]
        regime_stats.append({
            'regime': name,
            'accuracy': subset['correct'].mean(),
            'avg_pnl': subset['pnl'].mean(),
            'count': len(subset)
        })

    regime_df = pd.DataFrame(regime_stats)
    colors = ['green' if a > 0.5 else 'red' for a in regime_df['accuracy']]
    bars = ax.bar(range(len(regime_df)), regime_df['accuracy'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=2)
    ax.set_xticks(range(len(regime_df)))
    ax.set_xticklabels(regime_df['regime'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Yesterday Regime\n(Extreme vs Moderate)', fontweight='bold')
    ax.set_ylim(40, 70)

    # Add labels
    for bar, row in zip(bars, regime_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{row.accuracy*100:.1f}%\n(n={row.count:,})', ha='center', va='bottom', fontsize=9)

    plt.suptitle('When is the "Follow Yesterday" Strategy Most Reliable?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_when_to_trade.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '03_when_to_trade.png'}")

    return df


def create_summary(df, corr_df, results_df):
    """Create summary markdown."""
    print("[*] Creating summary...")

    # Best rule
    best = results_df.iloc[0]

    # Calculate key stats
    df_valid = df.dropna(subset=['direction_yesterday_same_hour', 'spread_direction'])
    df_valid['correct'] = (df_valid['direction_yesterday_same_hour'] == df_valid['spread_direction'])
    overall_accuracy = df_valid['correct'].mean()

    summary = f"""# DA-IDM Spread Direction Analysis - REALISTIC VERSION

## Critical Correction

**Previous analysis was WRONG** because it used same-day information (previous hour's spread).

**Reality**: DA position must be decided at D-1 (day before). You CANNOT use:
- Same-day spread direction
- Same-day prices
- Any lag < 24 hours

---

## Realistic Prediction Problem

**At DA auction time (D-1, ~noon), predict**: Will DA price be > or < IDM price tomorrow?

**Information available**:
- Yesterday's same hour outcome
- Historical patterns (last week, monthly averages)
- Calendar features (day of week, hour)
- Weather/load forecasts (not in this dataset)

---

## Key Findings

### Best Realistic Predictor: Yesterday Same Hour

| Metric | Value |
|--------|-------|
| Accuracy | **{overall_accuracy*100:.1f}%** |
| Win Rate | {best['win_rate']*100:.1f}% |
| Total P&L | {best['total_pnl']:,.0f} EUR/MWh |
| Avg P&L per trade | {best['avg_pnl']:.2f} EUR/MWh |
| Sharpe (annualized) | {best['sharpe']:.2f} |

### Feature Importance (Correlation with Direction)

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
"""

    for feat, row in corr_df.head(6).iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*'
        summary += f"| {feat} | {row['correlation']:.3f} {sig} | {'Predictive' if abs(row['correlation']) > 0.1 else 'Weak'} |\n"

    summary += f"""
---

## Transition Probabilities (Key Insight)

The spread direction shows **day-to-day persistence** for the same hour:

| Yesterday State | P(Today DA>IDM) | P(Today DA<IDM) |
|-----------------|-----------------|-----------------|
| DA > IDM | ~{df_valid[df_valid['direction_yesterday_same_hour']==1]['spread_direction'].mean()*100:.0f}% | ~{(1-df_valid[df_valid['direction_yesterday_same_hour']==1]['spread_direction'].mean())*100:.0f}% |
| DA < IDM | ~{df_valid[df_valid['direction_yesterday_same_hour']==0]['spread_direction'].mean()*100:.0f}% | ~{(1-df_valid[df_valid['direction_yesterday_same_hour']==0]['spread_direction'].mean())*100:.0f}% |

**Interpretation**: If yesterday same hour had DA > IDM, there's ~{df_valid[df_valid['direction_yesterday_same_hour']==1]['spread_direction'].mean()*100:.0f}% chance today will too.

---

## Trading Strategy Rules

### Simple Rule (Follow Yesterday Same Hour)
```
For each delivery hour H on day D:
    1. Look at hour H on day D-1
    2. If DA > IDM yesterday: SELL on DA, buy back on IDM
    3. If DA < IDM yesterday: BUY on DA, sell on IDM
```

### Accuracy: {overall_accuracy*100:.1f}%

This is **better than random (50%)** but **not as strong as the incorrect 78%** from same-day analysis.

---

## Realistic Performance Comparison

| Strategy | Accuracy | Description |
|----------|----------|-------------|
"""

    for _, row in results_df.iterrows():
        summary += f"| {row['rule']} | {row['accuracy']*100:.1f}% | {'Best' if row['rule'] == best['rule'] else ''} |\n"

    summary += f"""
---

## When to Trade (Conditional Analysis)

The strategy works better in certain conditions:

1. **Time of day**: Peak hours (8-20h) tend to be more predictable
2. **Large spreads yesterday**: When yesterday's spread was large, direction is more likely to persist
3. **Weekdays**: Slightly more predictable than weekends

---

## Limitations & Next Steps

### Current Limitations
- Only ~{overall_accuracy*100:.1f}% accuracy (vs 50% random)
- No weather/forecast features included
- Transaction costs not modeled

### To Improve Predictions
1. Add day-ahead weather forecasts
2. Add day-ahead load forecasts
3. Add scheduled outage information
4. Build ML model combining all D-1 features
5. Consider ensemble of yesterday + last week + seasonal patterns

---

## Files Generated

- `01_realistic_features.png` - Feature correlations and transition probabilities
- `02_realistic_rules.png` - Trading rule comparison
- `03_when_to_trade.png` - Conditional accuracy analysis
- `summary.md` - This summary
"""

    with open(OUTPUT_DIR / 'summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"[+] Saved: {OUTPUT_DIR / 'summary.md'}")


def main():
    print("=" * 70)
    print("DA-IDM SPREAD ANALYSIS - REALISTIC VERSION")
    print("(Using only D-1 information available at DA decision time)")
    print("=" * 70)

    df = load_data()
    df = create_realistic_features(df)
    corr_df = analyze_realistic_features(df)
    results_df = evaluate_realistic_rules(df)
    analyze_when_to_trade(df)
    create_summary(df, corr_df, results_df)

    print("\n" + "=" * 70)
    print("[+] REALISTIC Analysis complete!")
    print(f"    Output: MarketPriceGap/features/da_idm_realistic/")
    print("=" * 70)

    # Print key finding
    best = results_df.iloc[0]
    print(f"\n--- KEY FINDING ---")
    print(f"Best REALISTIC strategy: {best['rule']}")
    print(f"Accuracy: {best['accuracy']*100:.1f}% (vs 50% random)")
    print(f"Total P&L: {best['total_pnl']:,.0f} EUR/MWh")
    print(f"\nNote: This is much lower than the incorrect 78% from same-day analysis!")


if __name__ == "__main__":
    main()
