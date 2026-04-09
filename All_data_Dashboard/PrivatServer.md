# beam-solar Database Structure

#NEVER DELETE ANYTHIGN FROM THE DB

# 2. Connect to database
PGPASSWORD='s%Upt%H%5vpD2gW@9r&S' psql -h 127.0.0.1 -p 5433 -U beam -d beam-solar
```

---

## Schema: `core_logs`

| Table | Columns |
|-------|---------|
| `app_logs` | id, timestamp, level (varchar 10), message (text), details (jsonb) |
| `memory_logs` | id, timestamp, source (varchar 50), value, unit (varchar 10), process_id, details (jsonb) |
| `missing_data_logs` | id, timestamp, missing_columns (text), details (jsonb) |
| `analyzed_logs` | id, timestamp, analysis (jsonb) |
| `solar_logs` | id, timestamp, message (text) |

---

## Schema: `data`

| Table | Columns | Primary Key |
|-------|---------|-------------|
| `immediate` | date, time, value, avg, load, generation | (date, time) |
| `generation` | date, hour, predicted_value, real_value, difference | (date, hour) |
| `load` | date, hour, predicted_value, real_value, difference | (date, hour) |
| `dayahead_price` | date, hour, value | (date, hour) |

---

## Schema: `data2` (live data synced by lopatovac)

All tables have the same structure: `time` (timestamptz PK), `value` (double precision)

| Table | Description |
|-------|-------------|
| `DaE.OH.RE_WITH_GCC` | Hourly ACE with GCC |
| `EMS.DaE.PUB_3M.REAL_BALANCE` | 3-min real balance |
| `EMS.DaE.PUB_3M.ACKNOWLEDGED_REAL_BALANCE` | 3-min acknowledged real balance |
| `REAL_SYSTEM_LOAD` | Real-time system load |
| `REAL_SYSTEM_PRODUCTION` | Real-time system production |
| `RE_WITH_GCC_SEP_3M_GCC` | 3-min ACE with GCC separation |
| `RE_WITH_GCC_SEP_ODCH` | 15-min settlement deviation |
| `Okte.MargCena` | OKTE marginal price |
| `PICASSO.MarginalPricess.SEPS_POS.Avg` | PICASSO positive marginal price (avg) |
| `PICASSO.MarginalPricess.SEPS_NEG.Avg` | PICASSO negative marginal price (avg) |
| `PICASSO.MarginalPricess.SEPS_POS.Weighted` | PICASSO positive marginal price (weighted) |
| `PICASSO.MarginalPricess.SEPS_NEG.Weighted` | PICASSO negative marginal price (weighted) |
| `EQ_SK_PRICE_SPOT_15MIN_FORECAST` | time, prediction_time, value (extra column) |

---

## Schema: `predictions`

| Table | Columns | Primary Key |
|-------|---------|-------------|
| `xgb_normal` | timestamp, prediction_linear | timestamp |
| `xgb_immediate` | timestamp, prediction_binary, prediction_linear | timestamp |
