# 03 Feature-vs-Label Relationships

## Overall Correlations with Spread Target

| Feature | Correlation | Abs Corr |
|---------|------------|----------|
| da_price | +0.0596 | 0.0596 |
| da_supply | +0.0439 | 0.0439 |
| temp_forecast_da | -0.0345 | 0.0345 |
| idm_vwap_lag | +0.0345 | 0.0345 |
| nowcast_pred_rmean4 | +0.0245 | 0.0245 |
| load_momentum | +0.0241 | 0.0241 |
| da_price_change24h | +0.0216 | 0.0216 |
| temp_national_spread | +0.0146 | 0.0146 |
| radiation_national | -0.0145 | 0.0145 |
| load_rmean16 | +0.0136 | 0.0136 |
| reg_rmean8 | +0.0130 | 0.0130 |
| proxy_dev_from_hour | -0.0128 | 0.0128 |
| nowcast_h5 | +0.0123 | 0.0123 |
| proxy_rmean16 | -0.0102 | 0.0102 |
| da_flow_cz | +0.0101 | 0.0101 |
| nowcast_momentum_h2h3 | +0.0064 | 0.0064 |
| cloudcover | +0.0006 | 0.0006 |

## Key Findings

### Strongest Relationships
- **da_price** (r=+0.060): Positive relationship
- **da_supply** (r=+0.044): Positive relationship
- **temp_forecast_da** (r=-0.034): Negative relationship

### Linearity Assessment
- **da_price**: Non-monotonic pattern across deciles (nonlinear)
- **da_supply**: Non-monotonic pattern across deciles (nonlinear)
- **temp_forecast_da**: Non-monotonic pattern across deciles (nonlinear)
- **idm_vwap_lag**: Non-monotonic pattern across deciles (nonlinear)
- **nowcast_pred_rmean4**: Non-monotonic pattern across deciles (nonlinear)
- **load_momentum**: Non-monotonic pattern across deciles (nonlinear)
- **da_price_change24h**: Non-monotonic pattern across deciles (nonlinear)
- **temp_national_spread**: Non-monotonic pattern across deciles (nonlinear)
- **radiation_national**: Non-monotonic pattern across deciles (nonlinear)
- **load_rmean16**: Non-monotonic pattern across deciles (nonlinear)
- **reg_rmean8**: Non-monotonic pattern across deciles (nonlinear)
- **proxy_dev_from_hour**: Non-monotonic pattern across deciles (nonlinear)
- **nowcast_h5**: Non-monotonic pattern across deciles (nonlinear)
- **proxy_rmean16**: Non-monotonic pattern across deciles (nonlinear)
- **da_flow_cz**: Non-monotonic pattern across deciles (nonlinear)
- **nowcast_momentum_h2h3**: Non-monotonic pattern across deciles (nonlinear)
- **cloudcover**: Non-monotonic pattern across deciles (nonlinear)

### Year-over-Year Stability
- **da_price** [Stable]: 2024: +0.037, 2025: +0.064, 2026: +0.078
- **da_supply** [Stable]: 2024: +0.049, 2025: +0.030, 2026: +0.060
- **temp_forecast_da** [Stable]: 2024: -0.060, 2025: -0.007, 2026: -0.036
- **idm_vwap_lag** [Stable]: 2025: +0.033, 2026: +0.033
- **nowcast_pred_rmean4** [Stable]: 2025: +0.008, 2026: +0.047
- **load_momentum** [Stable]: 2024: +0.007, 2025: +0.029, 2026: +0.037
- **da_price_change24h** [Stable]: 2024: +0.009, 2025: +0.029, 2026: +0.028
- **temp_national_spread** [Stable]: 2024: +0.025, 2025: +0.004, 2026: +0.013
- **radiation_national** [Stable]: 2024: -0.031, 2025: -0.010, 2026: +0.011
- **load_rmean16** [Stable]: 2024: +0.000, 2025: -0.000, 2026: +0.028
- **reg_rmean8** [Stable]: 2024: -0.011, 2025: +0.008, 2026: +0.042
- **proxy_dev_from_hour** [Stable]: 2024: +0.009, 2025: -0.007, 2026: -0.041
- **nowcast_h5** [Stable]: 2025: -0.002, 2026: +0.023
- **proxy_rmean16** [Stable]: 2024: +0.013, 2025: -0.003, 2026: -0.039
- **da_flow_cz** [Stable]: 2024: -0.005, 2025: +0.008, 2026: +0.038
- **nowcast_momentum_h2h3** [Stable]: 2025: +0.001, 2026: +0.018
- **cloudcover** [Stable]: 2024: -0.010, 2025: +0.007, 2026: -0.004