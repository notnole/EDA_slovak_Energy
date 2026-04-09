# Beam Solar Database

PostgreSQL database running the live imbalance nowcasting pipeline.

## Connection

- Host: `127.0.0.1`
- Port: `5434`
- Database: `beam-solar`
- User: `beam`

## Schema: `beam` (live pipeline)

The main operational schema — data is written by the live nowcasting service.

### `beam.predictions` (~14k rows)
Live imbalance nowcast predictions, one row per (observation, lead_time).

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamptz | Observation time (when prediction was made) |
| `prediction_mwh` | float | LightGBM model prediction (MWh) |
| `lead_time_min` | int | Minutes until settlement period ends (0, 3, 6, 9, 12) |
| `baseline_pred` | float | Regulation baseline prediction (MWh) |

- Range: 2026-02-28 to present
- 5 rows per settlement period (one per lead time)

### `beam.immediate_data` (~14k rows)
3-minute SCADA snapshots fed to the model in real time.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamptz | Observation time |
| `load` | float | System load (MW) |
| `generation` | float | System generation (MW) |
| `regulation_energy` | float | Regulation energy (MW) |
| `cross_border_scheduled` | float | Scheduled cross-border flow (MW) |
| `cross_border_measured` | float | Measured cross-border flow (MW) |
| `seps_timestamp` | text | SEPS source timestamp |

### `beam.load` (~768 rows)
Hourly load values.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Date |
| `hour` | int | Hour of day |
| `value` | float | Load value (MW) |

### `beam.solar_data` (~14k rows)
Solar production data at 3-minute intervals.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamptz | Observation time |
| `signal_percent` | int | Solar signal percentage |
| `production_w` | float | Solar production (W) |

## Schema: `data` (historical features)

Historical hourly and 3-minute data used for model training and evaluation.

| Table | Rows | Description |
|-------|------|-------------|
| `immediate` | 94k | 3-min SCADA snapshots (value, avg, load, generation) |
| `immediate_ts` | 94k | Same with parsed timestamps |
| `full_prediction_dataset` | 94k | Joined feature matrix with all inputs |
| `load` / `load_h` | 4.9k | Hourly load (predicted, real, difference) |
| `generation` / `generation_h` | 4.9k | Hourly generation (predicted, real, difference) |
| `dayahead_price` / `dayahead_h` | 4.9k | Day-ahead hourly prices |

## Schema: `data2` (raw SCADA signals)

Raw time-series signals from Ipesoft EDA and external sources.

| Table | Rows | Description |
|-------|------|-------------|
| `DaE.OH.RE_WITH_GCC` | 1.5M | Regulation energy with GCC |
| `EMS.DaE.PUB_3M.REAL_BALANCE` | 412k | 3-min published real balance |
| `EMS.DaE.PUB_3M.ACKNOWLEDGED_REAL_BALANCE` | 412k | 3-min acknowledged real balance |
| `REAL_SYSTEM_LOAD` | 343k | Real system load |
| `REAL_SYSTEM_PRODUCTION` | 343k | Real system production |
| `RE_WITH_GCC_SEP_3M_GCC` | 345k | 3-min regulation with GCC |
| `RE_WITH_GCC_SEP_ODCH` | 9k | Regulation with GCC deviations |
| `Okte.MargCena` | 79k | OKTE marginal price |
| `PICASSO.MarginalPricess.SEPS_*.Avg` | 303-371k | PICASSO balancing prices (pos/neg avg) |
| `PICASSO.MarginalPricess.SEPS_*.Weighted` | 27-28k | PICASSO balancing prices (pos/neg weighted) |
| `yr_no_bardejov` | 1.2k | Weather data (temperature, clouds, humidity, wind) |

## Schema: `predictions` (legacy)

| Table | Rows | Description |
|-------|------|-------------|
| `xgb_normal` | 94k | Historical XGBoost predictions (linear) |
| `xgb_immediate` | 0 | Unused |

## Schema: `core_logs` (operational)

Application logs, memory monitoring, and missing data tracking.
- `app_logs`: 856k rows of application log messages
- `memory_logs`: 4.7k memory usage snapshots
- `missing_data_logs`: 17 missing data incidents
