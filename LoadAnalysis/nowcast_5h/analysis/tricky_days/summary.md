# Tricky Days Analysis

## Purpose
Identify and visualize days where DAMAS forecasts fail badly to understand model behavior.

## Top 10 Trickiest Days (2025)

| Date | Our MAE | DAMAS MAE | Improvement |
|------|---------|-----------|-------------|
| Feb 13 | 65.7 MW | 350.8 MW | 81% |
| Jul 04 | 62.2 MW | 130.8 MW | 52% |
| Oct 02 | 61.5 MW | 68.8 MW | 11% |
| Dec 30 | 53.4 MW | 94.8 MW | 44% |
| Oct 26 | 52.5 MW | 64.4 MW | 19% |

## Key Patterns

### When DAMAS Fails
- **Calendar events**: DST transitions, holidays
- **Weather extremes**: Sudden temperature changes
- **Load surprises**: Unexpected demand shifts

### When Our Model Helps Most
- Days with persistent bias (we adapt faster)
- Gradual trends (we follow the recent pattern)

### When We Still Struggle
- Sudden reversals (hour 23 worst: 48 MW MAE)
- Morning ramp uncertainty (hours 7-10)

## Error Distribution by Hour
- **Worst**: Hour 23 (48 MW), Hour 8 (40 MW)
- **Best**: Hour 2 (22 MW), Hour 3 (23 MW)

## Files
- `plot_tricky_days.py` - Generates comparison plots
- `plots/` - Daily comparison visualizations
