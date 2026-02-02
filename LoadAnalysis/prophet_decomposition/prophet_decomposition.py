"""
Prophet Decomposition for Slovakia Grid Load

Advantages over STL:
1. Multiple seasonalities (daily, weekly, yearly)
2. Built-in holiday effects - crucial for Slovakia
3. Trend changepoints detection
4. Uncertainty quantification

Slovak Public Holidays:
- Jan 1: New Year's Day / Day of the Establishment of the Slovak Republic
- Jan 6: Epiphany
- Good Friday (variable)
- Easter Monday (variable)
- May 1: Labour Day
- May 8: Victory over Fascism Day
- Jul 5: Saints Cyril and Methodius Day
- Aug 29: Slovak National Uprising Anniversary
- Sep 1: Constitution Day
- Sep 15: Our Lady of Sorrows Day
- Nov 1: All Saints' Day
- Nov 17: Struggle for Freedom and Democracy Day
- Dec 24-26: Christmas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
OUTPUT_PATH = Path(__file__).parent
OUTPUT_PATH.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def get_slovak_holidays(years: list) -> pd.DataFrame:
    """
    Generate Slovak public holidays for given years.
    Returns DataFrame with 'holiday', 'ds', 'lower_window', 'upper_window'.
    """
    from datetime import date
    import holidays as holidays_lib

    # Use holidays library for Slovakia (includes Easter calculations)
    sk_holidays = holidays_lib.Slovakia(years=years)

    holiday_list = []
    for dt, name in sorted(sk_holidays.items()):
        holiday_list.append({
            'holiday': name,
            'ds': pd.Timestamp(dt),
            'lower_window': 0,
            'upper_window': 0
        })

    # Add extended effects for major holidays
    # Christmas period (Dec 24-26 have lingering effects)
    # Easter period

    holidays_df = pd.DataFrame(holiday_list)

    # Add "bridge days" - days between holidays and weekends that often have reduced load
    # E.g., if holiday is Thursday, Friday is often a bridge day

    return holidays_df


def get_slovak_holidays_manual(years: list) -> pd.DataFrame:
    """
    Manual Slovak holidays definition (fallback if holidays lib not available).
    """
    from datetime import date, timedelta

    def easter_date(year):
        """Calculate Easter Sunday using Anonymous Gregorian algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    holidays_list = []

    for year in years:
        # Fixed holidays
        fixed = [
            (date(year, 1, 1), "New Year's Day"),
            (date(year, 1, 6), "Epiphany"),
            (date(year, 5, 1), "Labour Day"),
            (date(year, 5, 8), "Victory Day"),
            (date(year, 7, 5), "Cyril and Methodius Day"),
            (date(year, 8, 29), "Slovak National Uprising"),
            (date(year, 9, 1), "Constitution Day"),
            (date(year, 9, 15), "Our Lady of Sorrows"),
            (date(year, 11, 1), "All Saints Day"),
            (date(year, 11, 17), "Freedom and Democracy Day"),
            (date(year, 12, 24), "Christmas Eve"),
            (date(year, 12, 25), "Christmas Day"),
            (date(year, 12, 26), "St. Stephen's Day"),
        ]

        # Easter-based holidays
        easter = easter_date(year)
        easter_holidays = [
            (easter - timedelta(days=2), "Good Friday"),
            (easter + timedelta(days=1), "Easter Monday"),
        ]

        for dt, name in fixed + easter_holidays:
            holidays_list.append({
                'holiday': name,
                'ds': pd.Timestamp(dt),
                'lower_window': 0,
                'upper_window': 1,  # Effect can extend to next day
            })

    return pd.DataFrame(holidays_list)


def load_data() -> pd.DataFrame:
    """Load and prepare data for Prophet."""
    df = pd.read_parquet(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Prophet requires 'ds' and 'y' columns
    prophet_df = df[['datetime', 'actual_load_mw']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.dropna()
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

    return prophet_df, df


def fit_prophet_model(df: pd.DataFrame, holidays: pd.DataFrame) -> Prophet:
    """
    Fit Prophet model with multiple seasonalities and holidays.
    """
    print("Fitting Prophet model...")

    model = Prophet(
        # Trend
        growth='linear',
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        changepoint_range=0.9,  # Allow changepoints in first 90% of data

        # Seasonality
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,

        # Holidays
        holidays=holidays,
        holidays_prior_scale=10.0,  # Strong holiday effects

        # Uncertainty
        interval_width=0.95,
        uncertainty_samples=1000,
    )

    # Add custom seasonalities if needed
    # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(df)
    print("Model fitted successfully.")

    return model


def analyze_decomposition(model: Prophet, df: pd.DataFrame, original_df: pd.DataFrame):
    """
    Analyze and visualize Prophet decomposition components.
    """
    print("\nGenerating forecasts and components...")

    # Get predictions with components
    forecast = model.predict(df)

    # ==========================================================================
    # 1. MAIN DECOMPOSITION PLOT
    # ==========================================================================
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))

    # Observed
    axes[0].plot(forecast['ds'], df['y'].values, linewidth=0.5, alpha=0.7, color='blue')
    axes[0].set_ylabel('Load (MW)')
    axes[0].set_title('Prophet Decomposition of Slovakia Grid Load')
    axes[0].legend(['Observed'], loc='upper right')

    # Trend
    axes[1].plot(forecast['ds'], forecast['trend'], color='red', linewidth=1.5)
    axes[1].fill_between(forecast['ds'], forecast['trend_lower'], forecast['trend_upper'],
                          alpha=0.2, color='red')
    axes[1].set_ylabel('Trend (MW)')
    axes[1].set_title('Trend Component (with uncertainty)')

    # Weekly seasonality
    axes[2].plot(forecast['ds'], forecast['weekly'], color='green', linewidth=0.8)
    axes[2].set_ylabel('Weekly (MW)')
    axes[2].set_title('Weekly Seasonality')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Daily seasonality
    axes[3].plot(forecast['ds'], forecast['daily'], color='orange', linewidth=0.5)
    axes[3].set_ylabel('Daily (MW)')
    axes[3].set_title('Daily Seasonality')
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Holidays
    if 'holidays' in forecast.columns:
        axes[4].plot(forecast['ds'], forecast['holidays'], color='purple', linewidth=0.8)
        axes[4].set_ylabel('Holidays (MW)')
        axes[4].set_title('Holiday Effects')
        axes[4].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    else:
        # Sum all holiday columns
        holiday_cols = [c for c in forecast.columns if c.startswith('holiday') or
                       any(h in c.lower() for h in ['christmas', 'easter', 'new year'])]
        if holiday_cols:
            holiday_effect = forecast[holiday_cols].sum(axis=1)
            axes[4].plot(forecast['ds'], holiday_effect, color='purple', linewidth=0.8)
            axes[4].set_ylabel('Holidays (MW)')
            axes[4].set_title('Holiday Effects (Combined)')
            axes[4].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '01_prophet_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_prophet_decomposition.png")

    # ==========================================================================
    # 2. SEASONALITY PATTERNS
    # ==========================================================================
    fig = model.plot_components(forecast)
    fig.savefig(OUTPUT_PATH / '02_prophet_components.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_prophet_components.png")

    # ==========================================================================
    # 3. VARIANCE DECOMPOSITION
    # ==========================================================================
    print("\n" + "="*60)
    print("VARIANCE DECOMPOSITION")
    print("="*60)

    total_var = df['y'].var()

    components = {
        'Trend': forecast['trend'].var(),
        'Weekly': forecast['weekly'].var(),
        'Daily': forecast['daily'].var(),
    }

    # Handle yearly if present
    if 'yearly' in forecast.columns:
        components['Yearly'] = forecast['yearly'].var()

    # Calculate residual
    residual = df['y'].values - forecast['yhat'].values
    components['Residual'] = np.var(residual)

    print(f"\nTotal variance: {total_var:,.0f} MW²")
    print("\nComponent variances:")

    total_component_var = sum(components.values())
    for name, var in components.items():
        pct = var / total_var * 100
        print(f"  {name:12s}: {var:>12,.0f} MW² ({pct:>5.1f}%)")

    # Compare with STL
    print("\n" + "-"*40)
    print("Comparison with STL (period=168):")
    print("  STL Seasonal: 57.0%")
    print("  STL Trend:    35.8%")
    print("  STL Residual:  7.0%")
    print("-"*40)
    print(f"  Prophet Weekly+Daily: {(components['Weekly'] + components['Daily'])/total_var*100:.1f}%")
    print(f"  Prophet Trend:        {components['Trend']/total_var*100:.1f}%")
    print(f"  Prophet Residual:     {components['Residual']/total_var*100:.1f}%")

    # Variance decomposition bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(components.keys())
    values = [v/total_var*100 for v in components.values()]
    colors = ['red', 'green', 'orange', 'blue', 'gray'][:len(names)]

    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Prophet Variance Decomposition')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '03_variance_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: 03_variance_decomposition.png")

    # ==========================================================================
    # 4. HOLIDAY EFFECTS ANALYSIS
    # ==========================================================================
    print("\n" + "="*60)
    print("HOLIDAY EFFECTS ANALYSIS")
    print("="*60)

    # Get holiday effects from model
    holiday_effects = []
    for holiday_name in model.holidays['holiday'].unique():
        col_name = holiday_name
        if col_name in forecast.columns:
            effect = forecast[col_name].abs().max()
            holiday_effects.append({'Holiday': holiday_name, 'Max Effect (MW)': effect})

    if holiday_effects:
        holiday_df = pd.DataFrame(holiday_effects).sort_values('Max Effect (MW)', ascending=False)
        print("\nHoliday Effects (absolute max impact):")
        for _, row in holiday_df.head(15).iterrows():
            print(f"  {row['Holiday']:35s}: {row['Max Effect (MW)']:>8.1f} MW")

        # Holiday effects bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        top_holidays = holiday_df.head(15)
        ax.barh(range(len(top_holidays)), top_holidays['Max Effect (MW)'].values,
                color='purple', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_holidays)))
        ax.set_yticklabels(top_holidays['Holiday'].values)
        ax.set_xlabel('Max Load Impact (MW)')
        ax.set_title('Holiday Effects on Load (Top 15)')
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / '04_holiday_effects.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n  Saved: 04_holiday_effects.png")

    # ==========================================================================
    # 5. WEEKLY PATTERN DETAIL
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Weekly pattern
    forecast['dow'] = forecast['ds'].dt.dayofweek
    weekly_pattern = forecast.groupby('dow')['weekly'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors = ['blue']*5 + ['orange']*2

    axes[0].bar(range(7), weekly_pattern.values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(days)
    axes[0].set_ylabel('Weekly Effect (MW)')
    axes[0].set_title('Weekly Seasonality Pattern')
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Daily pattern
    forecast['hour'] = forecast['ds'].dt.hour
    daily_pattern = forecast.groupby('hour')['daily'].mean()

    axes[1].plot(daily_pattern.index, daily_pattern.values, 'o-', color='orange', linewidth=2)
    axes[1].fill_between(daily_pattern.index, 0, daily_pattern.values, alpha=0.3, color='orange')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Daily Effect (MW)')
    axes[1].set_title('Daily Seasonality Pattern')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '05_seasonality_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_seasonality_patterns.png")

    # ==========================================================================
    # 6. TREND WITH CHANGEPOINTS
    # ==========================================================================
    fig = model.plot(forecast)
    add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.title('Prophet Forecast with Trend Changepoints')
    plt.savefig(OUTPUT_PATH / '06_trend_changepoints.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_trend_changepoints.png")

    # ==========================================================================
    # 7. RESIDUAL ANALYSIS
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residual time series
    axes[0, 0].plot(forecast['ds'], residual, linewidth=0.5, alpha=0.7)
    axes[0, 0].set_ylabel('Residual (MW)')
    axes[0, 0].set_title(f'Residuals (Std: {np.std(residual):.1f} MW)')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')

    # Residual distribution
    axes[0, 1].hist(residual, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Residual (MW)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Residual Distribution')

    # Residual by hour
    forecast['residual'] = residual
    resid_by_hour = forecast.groupby('hour')['residual'].agg(['mean', 'std'])
    axes[1, 0].bar(resid_by_hour.index, resid_by_hour['mean'], yerr=resid_by_hour['std'],
                   capsize=2, alpha=0.7, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Mean Residual (MW)')
    axes[1, 0].set_title('Residual by Hour (bias check)')
    axes[1, 0].axhline(y=0, color='black', linestyle='--')

    # Residual by day of week
    resid_by_dow = forecast.groupby('dow')['residual'].agg(['mean', 'std'])
    axes[1, 1].bar(range(7), resid_by_dow['mean'], yerr=resid_by_dow['std'],
                   capsize=3, alpha=0.7, color=['blue']*5 + ['orange']*2, edgecolor='black')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(days)
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Mean Residual (MW)')
    axes[1, 1].set_title('Residual by Day of Week')
    axes[1, 1].axhline(y=0, color='black', linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '07_residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_residual_analysis.png")

    # ==========================================================================
    # 8. MODEL FIT METRICS
    # ==========================================================================
    print("\n" + "="*60)
    print("MODEL FIT METRICS")
    print("="*60)

    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    mape = np.mean(np.abs(residual / df['y'].values)) * 100

    print(f"\nIn-sample fit metrics:")
    print(f"  MAE:  {mae:.1f} MW")
    print(f"  RMSE: {rmse:.1f} MW")
    print(f"  MAPE: {mape:.2f}%")

    # Compare with DAMAS baseline
    print(f"\nComparison with DAMAS baseline:")
    print(f"  DAMAS MAE:  68.2 MW")
    print(f"  Prophet MAE: {mae:.1f} MW")
    print(f"  Improvement: {(68.2 - mae) / 68.2 * 100:.1f}%")

    return forecast, residual


def main():
    print("="*60)
    print("PROPHET DECOMPOSITION - SLOVAKIA GRID LOAD")
    print("="*60)

    # Load data
    prophet_df, original_df = load_data()
    print(f"\nData loaded: {len(prophet_df):,} records")
    print(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")

    # Get holidays
    years = prophet_df['ds'].dt.year.unique().tolist()
    print(f"\nYears in data: {years}")

    try:
        import holidays as holidays_lib
        holidays = get_slovak_holidays(years)
        print(f"Using holidays library - {len(holidays)} holidays loaded")
    except ImportError:
        holidays = get_slovak_holidays_manual(years)
        print(f"Using manual holidays - {len(holidays)} holidays loaded")

    print("\nSample holidays:")
    print(holidays.head(10).to_string(index=False))

    # Fit model
    model = fit_prophet_model(prophet_df, holidays)

    # Analyze decomposition
    forecast, residual = analyze_decomposition(model, prophet_df, original_df)

    # Save model components
    components_df = forecast[['ds', 'trend', 'weekly', 'daily', 'yhat']].copy()
    components_df['residual'] = residual
    components_df['observed'] = prophet_df['y'].values
    components_df.to_parquet(OUTPUT_PATH / 'prophet_components.parquet', index=False)
    print(f"\n  Saved: prophet_components.parquet")

    print("\n" + "="*60)
    print("DECOMPOSITION COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_PATH}")

    # Summary
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("""
1. HOLIDAY EFFECTS:
   - Prophet captures individual holiday impacts
   - Christmas period likely shows largest effect
   - Can use for improved forecasting

2. SEPARATED SEASONALITIES:
   - Daily pattern: Morning ramp, midday peak, evening decline
   - Weekly pattern: Weekday vs weekend clearly separated
   - Yearly pattern: Winter high, summer low

3. TREND CHANGEPOINTS:
   - Prophet detects when load patterns shift
   - Useful for identifying structural changes

4. RESIDUAL:
   - What's left after all components removed
   - This is what ML models need to capture
""")


if __name__ == '__main__':
    main()
