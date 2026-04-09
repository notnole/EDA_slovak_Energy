# Market Price Gap Analysis

## Overview

Analysis of Slovak electricity market price dynamics -- DA, IDM, and imbalance prices -- to build trading indicators and BESS scheduling strategies. All features respect D-1 timing (available before 11:00 DA gate closure).

## Data Period
- DA prices: Jan 2024 - Jan 2026
- IDM + Imbalance: Jan 2025 - Jan 2026 (shorter overlap)
- DA price forecasts: Sep 2025 - Jan 2026

## Key Results

| Application | Metric | Value |
|-------------|--------|-------|
| **BESS scheduling** | Indicator revenue | 192,322 EUR (2 cycles/day, persistent SoC) |
| | Capture rate | 78% of perfect foresight |
| | vs naive fixed-schedule | 1.9x better |
| **Speculative trading** | Total P&L | 70,161 EUR (DA-IDM spread) |
| | Best rule (R3: Large Persistence) | 68.4% accuracy, PF=1.78 |
| | Worst rule (R4: Moderate Persistence) | 47.7% accuracy -- REMOVE |

## Sub-Analyses

### Data Pipeline
| Folder | Description |
|--------|-------------|
| `scripts/` | `load_market_prices.py` -- loads DA + IDM raw data, handles mixed 15-min/60-min resolution |
| `data/processed/` | `hourly_market_prices.csv` -- canonical hourly dataset (corrected Feb 2026) |

### Label & Feature Analysis
| Folder | Description | Key Finding |
|--------|-------------|-------------|
| [label/basic_stats/](label/basic_stats/summary.md) | DA-IDM-Imbalance spread statistics | DA-IDM spread avg +8.7 EUR/MWh, weekend effect strong |
| [features/da_idm_spread/](features/da_idm_spread/summary.md) | Raw spread analysis | Spread persistence, hourly patterns |
| [features/da_idm_realistic/](features/da_idm_realistic/summary.md) | Realistic spread with timing constraints | Post-gate IDM VWAP is the correct benchmark |
| [features/da_idm_filtered/](features/da_idm_filtered/summary.md) | Filtered spread analysis | Volume-weighted, outlier-removed |
| [features/spread_indicators/](features/spread_indicators/summary.md) | Predictive indicators for spread direction | Load deviation, persistence, weekend are top 3 |
| [features/idm_imb_hourly/](features/idm_imb_hourly/summary.md) | IDM vs imbalance hourly patterns | IDM systematically higher than imbalance by ~20 EUR |
| [features/regulation_regime/](features/regulation_regime/summary.md) | Regulation regime and spread drivers | Morning regulation signal has limited predictive power |

### DA Indicator (Core)
| Folder | Description |
|--------|-------------|
| [da_indicator/](da_indicator/summary.md) | Main indicator: 5 speculative rules + BESS rank prediction |
| [da_indicator/reports/](da_indicator/reports/) | LaTeX reports for Overleaf (speculative + BESS strategies) |
| [da_indicator/scripts/bess_simulation.py](da_indicator/scripts/bess_simulation.py) | Persistent SoC BESS simulation (2 cycles/day) |

### Supporting Analysis
| Folder | Description | Key Finding |
|--------|-------------|-------------|
| [da_decision_matrix/](da_decision_matrix/summary.md) | Multi-source decision signals (load, solar, temp, nuclear) | Weekend spread 89% accurate, nuclear drives peak timing |
| [da_forecast_analysis/](da_forecast_analysis/summary.md) | DA price forecast evaluation (5 models) | SK Forecast1: r=0.80, MAE=18.1 EUR/MWh (corrected Feb 2026) |

## Known Issues (Fixed)
- **2026 data resolution**: Raw OKTE 2026 data uses 15-min periods (96/day) vs hourly (24/day) for 2024-2025. Fixed in `load_market_prices.py` with per-row resolution detection. All downstream analyses updated.
