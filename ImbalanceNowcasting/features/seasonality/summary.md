# Feature Seasonality Analysis Summary

## Purpose
Analyze daily, weekly, monthly, and yearly patterns in features to understand their predictable components.

## Key Findings

### Load Seasonality
- **Daily**: Strong pattern - 2400 MW at night, peaks 3600 MW midday
- **Weekly**: ~300 MW lower on weekends
- **Monthly**: Higher in winter (Jan, Feb, Dec), lower in summer
- **Yearly**: 2026 shows higher load (but only 24 days of data)

### Regulation Seasonality
- **Daily**: Mean near zero all hours, but variance peaks midday (10-14h)
- **Weekly**: Slightly more negative on weekends (oversupply)
- **Monthly/Yearly**: No strong pattern - mostly noise

### Production (Oct 2025+ only)
- **Daily**: Two peaks - morning (8h) and afternoon (16-18h)
- **Weekly**: Lower on weekends

### Export/Import (Oct 2025+ only)
- **Daily**: Inverse of load - drops when domestic demand peaks
- **Weekly**: Higher exports on weekends (less domestic demand)

## Weekday vs Weekend
- Load: ~300 MW lower on weekends across all hours
- Regulation: More negative (oversupply) on weekends
- Production: Lower on weekends
- Export: Higher on weekends

## Model Implications

1. **Time features essential**: Hour of day and day of week capture significant variance
2. **Load is highly predictable**: Daily+weekly patterns dominate
3. **Regulation variance changes by hour**: Model uncertainty should vary by time of day
4. **Weekend flag useful**: Different dynamics on weekends
