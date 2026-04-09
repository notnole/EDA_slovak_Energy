# Production Database Inventory

**Database**: `DB_EMS` on `10.4.1.66`
**Generated**: 2026-02-22
**Total Objects**: 667 (416 tables, 251 views)

---

## 📊 Executive Summary

The production database contains comprehensive market data for the Slovak electricity market (ISOT - Intraday Continuous Trading) plus supporting operational, administrative, and forecasting data. Core market data covers **15-minute delivery periods with 96 periods per day**.

### Latest Data Availability (as of 2026-02-23)
- **Hub-to-Hub Flows**: 2026-02-23 (1.28M rows)
- **ISOT Order Book**: 2026-02-23 (19.9M rows)
- **Traded Volumes**: 2026-02-23 (444K rows)
- **Bids/Orders**: 2026-02-23 (497K rows)
- **WebSocket Trades**: 2026-02-23 (16.2K rows)

---

## 🔌 Core Market Data Tables

### 1. **Hub-to-Hub Flow Data** (Cross-Border Interconnection)

#### `isot_vdt_hub2hub` ⭐ PRIMARY
**Latest Data**: 2025-11-25 → **2026-02-23** | **1.28M rows**

Cross-border electricity flows between interconnected areas (hubs). Used for capacity and flow analysis.

| Column | Type | Description |
|--------|------|-------------|
| `id_area` | int | Area identifier (FK to `isot_vdt_hub2hub_areas`) |
| `tradeday` | timestamp | Trading day |
| `periodfrom` | timestamp | Period start (15-min granularity) |
| `periodto` | timestamp | Period end |
| `available_in` | numeric | Available import capacity [MW] |
| `available_out` | numeric | Available export capacity [MW] |
| `date_in` | timestamp | Record creation timestamp |

**Collection Frequency**: Daily updates for previous trading day
**Update Pattern**: Typically available within 24 hours after delivery day

---

#### `isot_vdt_hub2hub_areas` (Reference)
Hub/area definitions with EIC codes and country mappings.

| Column | Type | Description |
|--------|------|-------------|
| `id_area` | int | Area ID (PK) |
| `area_name` | varchar | Hub name (e.g., "Slovakia HUB") |
| `eic` | varchar | EIC code (European Identifier Code) |
| `id_country` | int | Country ID |
| `ano_platny` | smallint | Active flag (1=active, 0=inactive) |

**Reference Data**: ~10-15 areas (static, rarely changes)

---

#### `isot_vdt_hub2hub_countries` (Reference)
Country codes for hub areas.

| Column | Type | Description |
|--------|------|-------------|
| `id_country` | int | Country ID |
| `countrycode` | varchar | ISO 3-letter code (SK, CZ, PL, etc.) |

**Reference Data**: ~5-10 countries

---

#### `vdt_isot_h2h_v2` (Alternative/Legacy)
**Latest Data**: 2024-10-25 → 2025-06-06 | **2.58M rows** (STALE)

Older version of hub-to-hub data with different structure (flow data at 15-min granularity with profile role details).

⚠️ **STATUS**: Legacy/stale - use `isot_vdt_hub2hub` instead

| Column | Type | Description |
|--------|------|-------------|
| `tradeday` | timestamp | Trading day |
| `deliverydur` | int | Delivery duration (15 or 60) |
| `id_areafrom` | int | Source area |
| `id_areato` | int | Destination area |
| `id_profilerole` | int | Profile role |
| `periodfrom` | int | Period index (0-96) |
| `periodto` | int | Period end index |
| `amount` | numeric | Flow amount [MWh] |
| `date_in` | timestamp | Record timestamp |

---

#### `vdt_isot_h2h_area`, `vdt_isot_h2h_profilerole` (Reference)
Supporting lookup tables for h2h_v2.

---

### 2. **ISOT Order Book** (Market Depth / Best Bid-Ask)

#### `vdt_isot_knihaobjednavok_best` ⭐ PRIMARY
**Latest Data**: 2025-11-21 → **2026-02-23** | **19.88M rows**

Best bid/ask prices at each snapshot in time. Multiple updates per second per delivery period.

| Column | Type | Description |
|--------|------|-------------|
| `tradeday` | timestamp | Trading day |
| `periodfrom` | int | Period index (0-96, 15-min intervals) |
| `periodto` | int | Period end index |
| `price` | numeric | Bid or ask price [EUR/MWh] |
| `deliverydur` | int | 15 or 60 (minutes) |
| `tradetype` | varchar | 'N'=bid (nákup), 'P'=ask (predaj) |
| `lastupdate` | timestamp | When this snapshot was recorded |
| `date_in` | timestamp | DB insertion timestamp |
| `id_depth` | int | Order book depth (1=best, 2=second-best) |
| `amount` | numeric | Quantity at this price [MWh] |

**Collection**: Real-time, typically 1-2 updates per second per period
**Period Index**: Period 0 = 00:00-00:15, Period 96 = 24:00 (next day)

---

#### `vdt_isot_knihaobjednavok` (Full Depth)
**Latest Data**: Empty in production (0 rows)

Full order book with all depth levels. Currently not populated; replaced by `_best` variant.

---

#### `vdt_isot_knihaobjednavok_best_ws` (WebSocket Archive)
**Latest Data**: Empty in production (0 rows)

Intended for WebSocket-sourced order book data (separate ingestion pipeline). Currently unused.

---

### 3. **Last Traded Prices**

#### `vdt_isot_lasttrades`
**Latest Data**: 2025-11-21 → **2026-02-23** | **444K rows**

Most recent executed trade price for each delivery period.

| Column | Type | Description |
|--------|------|-------------|
| `tradeday` | timestamp | Trading day |
| `periodfrom` | int | Period index |
| `periodto` | int | Period end |
| `price` | numeric | Last executed price [EUR/MWh] |
| `amount` | numeric | Last trade volume [MWh] |
| `total_amount` | numeric | Cumulative traded volume |
| `deliverydur` | int | 15 or 60 |
| `lastupdate` | timestamp | Trade timestamp |
| `date_in` | timestamp | DB insertion timestamp |

**Collection**: Updated whenever a trade is executed; **typically late afternoon/evening** after most trading concludes

---

### 4. **Traded Volumes** (Aggregate)

#### `vdt_isot_zobchodovane`
**Latest Data**: Empty in production (0 rows)

Intended for aggregate traded volumes. Currently not populated.

---

### 5. **Market Depth Metadata**

#### `vdt_isot_marketdepth` (Reference)
Depth level definitions (best, 2nd-best, etc.).

| Column | Type | Description |
|--------|------|-------------|
| `id_depth` | int | Depth ID |
| `market_depth` | numeric | Depth level value |
| `description` | varchar | Description |
| `ano_platny` | smallint | Active flag |
| `b_archiv` | smallint | Archive flag |

---

---

## 📝 Trades & Orders

### 6. **ISOT Trades (Executed)**

#### `isot_vdt_trade`
**Latest Data**: 2023-11-28 → **2026-02-23** | **14.7K rows**

All executed trades on ISOT market. Includes full order details: trader, delivery period, amount, price, timestamps.

| Column | Type | Description |
|--------|------|-------------|
| `msgcode` | varchar | Message code |
| `idtrade` | int | Trade ID |
| `tradeday` | timestamp | Trading day |
| `tradestage` | varchar | Stage (e.g., "O"=open, "C"=closed) |
| `tradetype` | varchar | 'B'=buy, 'S'=sell |
| `traderid` | int | Trader ID |
| `profilerole` | varchar | Profile role (e.g., "BRP", "BSP") |
| `tradeidref` | int | Referenced trade ID |
| `periodfrom` | int | Period index |
| `periodto` | int | Period end |
| `valuetime` | timestamp | Value/delivery time |
| `amount` | numeric | Traded amount [MWh] |
| `price` | numeric | Trade price [EUR/MWh] |
| `datetime` | timestamp | Trade creation timestamp |
| `datetimemodify` | timestamp | Last modification time |
| `blockorder` | varchar | Block order type |
| `deliverydur` | int | 15 or 60 |
| `indication` | varchar | Indication flag |
| `ordexpiration` | timestamp | Order expiration |
| `ano_platny` | int | Valid flag |
| `date_in`, `date_ch` | timestamp | Audit: creation/modification |
| `user_in`, `user_ch` | varchar | Audit: users |
| `tradecomment` | varchar | Trade notes |

**Collection**: Real-time during trading window
**Trading Window**: 08:00 day-1 to ~15:00 gate closure day-0 (5 min before delivery)

---

#### `isot_vdt_bid`
**Latest Data**: 2024-01-30 → **2026-02-23** | **497K rows**

Bids/orders submitted (whether filled or not). Superset of executed trades.

**Same structure as `isot_vdt_trade`** plus additional fields for orders that may not be fully or partially executed.

---

#### `isot_vdt_bid_sending_log`
**Latest Data**: Not queried

Log of bid/order submissions with status (pending, accepted, rejected, etc.).

| Column | Type | Description |
|--------|------|-------------|
| `id_isot_order_sending_log` | int | Log ID |
| `intervalstart` | timestamp | Delivery period start |
| `intervalend` | timestamp | Delivery period end |
| `expirationtime` | timestamp | Order expiration |
| `amount` | real | Order amount [MWh] |
| `price` | real | Offered price [EUR/MWh] |
| `status` | char | Order status |
| `tradetype` | char | B=buy, S=sell |
| `blockorder` | char | Block order flag |
| `blocktype` | char | Block type |
| `tradestage` | char | Trade stage |
| `deliverydur` | int | Duration |
| `indication` | char | Indication flag |
| `date_in` | timestamp | Submission time |

---

### 7. **WebSocket Trade Feeds** (Real-Time)

#### `isot_vdt_trade_websocket`
**Latest Data**: 2025-11-26 → **2026-02-23** | **4.6K rows**

Real-time trade events delivered via WebSocket feed.

| Column | Type | Description |
|--------|------|-------------|
| `id_trade` | int | Trade ID |
| `tradeday` | timestamp | Trading day |
| `deliverydur` | smallint | 15 or 60 |
| `tradetype` | varchar | Buy/Sell |
| `tradestage` | char | Trade stage |
| `valuetime` | timestamp | Delivery time |
| `price` | numeric | Price [EUR/MWh] |
| `quantity` | numeric | Total quantity [MWh] |
| `realizedquantity` | numeric | Executed quantity |
| `realizedpriceweighted` | numeric | VWAP |
| `remainingquantity` | numeric | Unfilled quantity |
| `expiration` | timestamp | Order expiration |
| `traderid` | int | Trader |
| `createdat`, `updatedat` | timestamp | Timestamps |
| `note` | varchar | Notes |
| `id_type` | smallint | Type ID |
| `id_action` | smallint | Action (FK to `isot_vdt_websocket_action`) |
| `date_in`, `date_ch` | timestamp | Audit |
| `correlationid` | varchar | Correlation ID |
| `id_participant` | int | Participant (FK to `isot_vdt_websocket_participant`) |
| `groupid` | int | Group ID |

---

#### `isot_vdt_trade_websocket_all`
**Latest Data**: 2025-12-01 → **2026-02-23** | **16.2K rows**

Extended WebSocket trade feed (likely includes order updates, partial fills, etc.).

Same structure as `isot_vdt_trade_websocket`.

---

#### `isot_vdt_trade_websocket_old`
Archived WebSocket data (not queried).

---

#### Supporting Tables
- `isot_vdt_websocket_action`: Action type definitions
- `isot_vdt_websocket_participant`: Participant types
- `isot_vdt_websocket_type`: Type definitions
- `isot_websocket_connection_log`: WebSocket connection events with timestamps and status codes

---

### 8. **Day-Ahead Market (DT) Data**

#### `isot_dt_trades`
Day-ahead market (DT) executed trades.

| Column | Type | Description |
|--------|------|-------------|
| `idtrade` | bigint | Trade ID |
| `profilerole` | varchar | Profile role |
| `periodfrom` | int | Period (0-24 for hourly DT) |
| `valuetime` | timestamp | Delivery time |
| `tradetype` | varchar | Buy/Sell |
| `amount` | numeric | Amount [MWh] |
| `price` | numeric | Price [EUR/MWh] |
| `tradecomment` | varchar | Notes |

---

#### `isot_dt_evaluation`
Day-ahead market evaluation/settlement data.

| Column | Type | Description |
|--------|------|-------------|
| `valuetime` | timestamp | Evaluation time |
| `profilerole` | varchar | Profile role |
| `periodfrom` | int | Hour index (0-23) |
| `tradetype` | varchar | Buy/Sell |
| `amount` | numeric | Amount [MWh] |
| `price` | numeric | Price [EUR/MWh] |

---

#### `isot_dt_evaluation_blocks`
Block order evaluations in day-ahead market.

---

#### `isot_dt_log`
Day-ahead market processing log with status tracking.

| Column | Type | Description |
|--------|------|-------------|
| `id_pp_isot_dt_log` | int | Log ID |
| `bt` | timestamp | Begin time |
| `et` | timestamp | End time |
| `status` | int | Processing status |
| `date_in` | timestamp | Log timestamp |
| `user_in` | varchar | User |

---

### 9. **Open Market Positions**

#### `vdt_open_market_position`
**Latest Data**: Empty in production (0 rows)

Current open buy/sell orders and positions per trader.

| Column | Type | Description |
|--------|------|-------------|
| `trade_day` | timestamp | Trading day |
| `traderid` | int | Trader ID |
| `periodfrom` | int | Period |
| `periodto` | int | Period end |
| `amount_sell` | varchar | Sell amount [MWh] |
| `price_sell` | varchar | Sell price [EUR/MWh] |
| `amount_buy` | varchar | Buy amount [MWh] |
| `price_buy` | varchar | Buy price [EUR/MWh] |
| `deliverydur` | int | Duration |
| `ordexpiration` | timestamp | Expiration |
| `tradecomment` | varchar | Notes |
| `ano_platny` | int | Active |

---

---

## 🤖 BESS Optimization Algorithm Data

### 10. **Algorithm Trade Execution Logs** ⭐

#### `algo_vdt_bess_algo_trades_log` - LIVE TRADING HISTORY
**Data Range**: 2025-11-26 → **2026-02-23** | **4,558 total trades**
**Trading Days**: 77 days | **Live Performance Data**

Complete execution log of all trades executed by the BESS Rolling Intrinsic optimization algorithm. This is the primary **operational performance record** of the strategy.

**📊 Overall Trading Performance (Since Nov 26, 2025)**:
| Metric | Value |
|--------|-------|
| **Total Trades** | 4,558 |
| **Trading Days** | 77 |
| **Total Buy Volume** | 1,911.9 MWh |
| **Total Sell Volume** | 2,593.3 MWh |
| **Total Cost (Buys)** | €210,274.27 |
| **Total Revenue (Sells)** | €434,732.93 |
| **Gross Profit** | **€224,458.66** ⭐ |
| **Profit per MWh Sold** | **€86.55** |
| **Avg Sell Price** | €167.64/MWh |
| **Avg Buy Price** | €109.98/MWh |
| **Price Spread Capture** | €57.66/MWh |

**Daily Range**:
- Best Day: 2026-02-09 → **€13,470.98 profit** (90.1 MWh sold at €180.34 avg)
- Worst Day: 2026-02-16 → **€-6,843.29 loss** (bought 120.5 MWh at €102.45 avg, sold only 37.8 MWh)
- Consistent profitability: 56 positive days, 21 negative days (73% win rate)

| Column | Type | Description |
|--------|------|-------------|
| `zacintervalu` | timestamp | Trade execution timestamp (minute level) |
| `expiration` | timestamp | Order expiration time |
| `tradetype` | varchar | **'N'** = Bid (Buy), **'P'** = Ask (Sell) |
| `deliverydur` | varchar | Delivery duration ('15' or '60') |
| `mnozstvo` | numeric | Trade volume [MWh] |
| `cena` | numeric | Execution price [EUR/MWh] |
| `retmsg` | varchar | Return message from exchange |
| `tretcode` | varchar | Treatment code |
| `comment_blok` | varchar | Block comment (e.g., 'BL4', 'BL5') |
| `date_in` | timestamp | Database insertion timestamp |
| `user_in` | varchar | System/user identifier |
| `correlationid` | int | Correlation ID |

**Trade Distribution**:
- **Sell Trades (P)**: 2,296 trades | 2,593.3 MWh | Avg €170.97/MWh
- **Buy Trades (N)**: 2,262 trades | 1,911.9 MWh | Avg €110.20/MWh

**Key Statistics**:
- Price Range: €2.00 - €629.33/MWh (high volatility periods)
- Median Trade Size: 0.5-1.0 MWh
- Most Active Hours: Afternoon/evening (typical ISOT trading hours)
- Block Orders: Tracks block order types (e.g., BL4, BL5)

**Collection**: Real-time during algorithm execution, recorded minute-by-minute

| Column | Type | Description |
|--------|------|-------------|
| `zacintervalu` | timestamp | Interval start time |
| `expiration` | timestamp | Order expiration |
| `tradetype` | varchar | 'B'=buy, 'S'=sell |
| `deliverydur` | varchar | '15' or '60' |
| `mnozstvo` | numeric | Amount [MWh] |
| `cena` | numeric | Price [EUR/MWh] |
| `retmsg` | varchar | Return message |
| `tretcode` | varchar | Treatment code |
| `comment_blok` | varchar | Block comment |
| `date_in` | timestamp | Execution timestamp |
| `user_in` | varchar | Executing user/system |
| `correlationid` | int | Correlation ID |

**Collection**: Real-time during algorithm execution (typically continuous during trading hours)

---

### 11. **Algorithm Configuration & Metadata**

#### `algo_vdt_bess_algorithms`
Algorithm versions and configurations.

| Column | Type | Description |
|--------|------|-------------|
| `version` | int | Version number |
| `description` | varchar | Algorithm description |
| `event_name` | varchar | Event name |

---

### 12. **Simulation & Backtesting Results**

#### `algo_vdt_bess_simulation_result`
**Rows**: Significant volume

Detailed simulation results with profitability metrics.

| Column | Type | Description |
|--------|------|-------------|
| `simulation_id` | int | Simulation ID |
| `spread` | numeric | Bid-ask spread [EUR/MWh] |
| `profit` | numeric | Total profit [EUR] |
| `profit_per_hour` | numeric | Hourly profit [EUR] |
| `cycles` | int | Number of charge/discharge cycles |
| `cycles_per_day` | numeric | Daily cycle count |
| `profit_per_cycle` | numeric | Profit per cycle [EUR/cycle] |
| `interval` | smallint | Time interval (minutes) |
| `version` | int | Algorithm version |
| `bt`, `et` | timestamp | Backtest period (begin/end) |
| `date_in` | timestamp | Simulation timestamp |
| `lookahead` | int | Lookahead window (hours) |
| `hours_included` | smallint | Hours in backtest |
| `tradeday_start_offset` | smallint | Offset from trade day |
| `capacity` | numeric | Battery capacity [MWh] |
| `chargepower` | numeric | Charge power [MW] |
| `dischargepower` | numeric | Discharge power [MW] |
| `soc_min`, `soc_max` | numeric | SoC bounds (0-1) |
| `softness` | numeric | Algorithm softness parameter |
| `soc_init` | numeric | Initial SoC |

---

#### `algo_vdt_bess_simulation_offers`
Offers considered during simulation.

| Column | Type | Description |
|--------|------|-------------|
| `simulation_id` | int | Simulation ID |
| `tradeday` | timestamp | Trading day |
| `tradetype` | varchar | Buy/Sell |
| `price` | numeric | Offer price [EUR/MWh] |
| `transactiontime` | timestamp | Offer time |
| `periodfrom` | smallint | Period |
| `version` | int | Algorithm version |
| `deliverydur` | int | Duration |
| `amount` | numeric | Amount [MWh] |

---

#### `algo_vdt_bess_not_traded_offers`
Offers rejected by the algorithm (below profitability threshold).

| Column | Type | Description |
|--------|------|-------------|
| `valuetime` | timestamp | Value time |
| `price` | numeric | Offer price [EUR/MWh] |
| `tradetype` | varchar | Buy/Sell |
| `deliverydur` | smallint | Duration |
| `amount` | numeric | Amount [MWh] |
| `transactiontime` | timestamp | Offer timestamp |
| `expirationtime` | timestamp | Expiration |
| `date_in` | timestamp | Record timestamp |
| `id_offer` | int | Offer ID |
| `spread` | numeric | Spread [EUR/MWh] |
| `algorithm` | int | Algorithm ID |

**Collection**: Real-time during algorithm operation; rejected offers logged for analysis

---

### 13. **Picasso Trading (Alternative Algorithm)**

#### `algo_vdt_picasso_trading`
Alternative trading algorithm (Picasso) execution log.

| Column | Type | Description |
|--------|------|-------------|
| `id_picasso_trade` | bigint | Trade ID |
| `picasso_time` | timestamp | Picasso execution time |
| `picasso_price` | numeric | Picasso trading price [EUR/MWh] |
| `wavg_time` | timestamp | VWAP calculation time |
| `wavg_price` | numeric | Volume-weighted average price |
| `value_time` | timestamp | Delivery time |
| `deliverydur` | smallint | Duration |
| `tradetype_op` | varchar | Trade type (opening) |
| `vdt_price_op` | numeric | VDT price (opening) |
| `tradetype_cl` | varchar | Trade type (closing) |
| `vdt_price_cl` | numeric | VDT price (closing) |
| `date_in`, `date_ch` | timestamp | Audit timestamps |

---

### 14. **Simulation Views & Analysis**

#### `vdt_isot_algo_simulation_best`
Best offer prices during simulation.

| Column | Type | Description |
|--------|------|-------------|
| `version` | int | Algorithm version |
| `tradeday` | timestamp | Trading day |
| `tradetype` | varchar | Buy/Sell |
| `price` | numeric | Best price [EUR/MWh] |
| `transactiontime` | timestamp | Time |
| `periodfrom` | smallint | Period |

---

#### `vdt_isot_algo_simulation_result`
Aggregated simulation results.

---

#### `algo_isot_vw_knihaobjednavok_best_statistics`
Aggregated order book statistics (view).

---

---

## 📚 Reference & Lookup Tables

### Type/Code Tables
- `isot_vdt_tradetype_types`: Trade type codes ('B'=buy, 'S'=sell)
- `isot_vdt_tradestage_types`: Trade stage codes ('O'=open, 'C'=closed)
- `isot_vdt_blockorder_types`: Block order types
- `isot_vdt_blocktype_types`: Block type definitions
- `isot_vdt_deliverydur_types`: Delivery durations (15, 60 minutes)
- `isot_vdt_indication_types`: Indication flags
- `isot_vdt_status_types`: Status codes

### Trader Data
- `isot_vdt_traders`: Trader registry with IDs, names, and flags
  - `traderid`: Trader ID
  - `trader_name`: Name
  - `b_right_to_trade`: Trading authorization flag
  - `balgotrader`: Algorithmic trader flag
  - `websocket_name`: WebSocket identifier

---

---

## 🔮 Prediction & Forecasting Tables

⚠️ **STATUS**: All prediction/forecasting tables exist in the schema but contain **NO DATA** (0 rows in production)

The infrastructure for price and demand forecasting is fully set up but currently not actively populated. These tables are ready for integration of ML-based predictions.

### Forecast Engine Tables

#### `ems_forecast` (Configuration)
**Rows**: 0

Forecast model definitions and configurations.

| Column | Type | Description |
|--------|------|-------------|
| `id_forecast` | int | Model ID (PK) |
| `code`, `name` | varchar | Model identifier and name |
| `descript` | varchar | Description |
| `id_forecast_type` | int | Type of forecast |
| `id_ems_entity` | int | EMS entity (target) |
| `vld_date_from`, `vld_date_to` | timestamp | Validity range |
| `train_vect_code` | varchar | Training vector code |
| `pred_vect_code` | varchar | Prediction vector code |
| `valid_vect_code` | varchar | Validation vector code |
| `train_history` | int | Training history (days) |
| `prediction_type` | int | Type (e.g., price, demand) |
| `prediction_horizon` | int | Forecast horizon (hours/periods) |
| `model_structure` | int | Architecture type |
| `model_complexity` | int | Complexity level |
| `freq_domain_terms` | int | Frequency domain terms |
| `step` | int | Time step |
| `user_mode` | int | User mode |
| `use_target_offset` | int | Offset flag |
| `include_calendar` | int | Calendar features flag |
| `default_value` | int | Default value |
| `moving_average` | int | MA window |
| `normalization` | int | Normalization type |
| `timezone` | int | Timezone |
| `active` | int | Active flag |

---

#### `ems_forecast_training_log`
**Rows**: 0

Training execution logs for forecast models.

| Column | Type | Description |
|--------|------|-------------|
| `id_forecast_training` | int | Training run ID |
| `id_forecast` | int | Model ID (FK) |
| `bt`, `et` | timestamp | Training period (begin/end) |
| `start_time`, `end_time` | timestamp | Execution timestamps |
| `status` | varchar | Training status |
| `rmse`, `pmad`, `mape` | numeric | Performance metrics (RMSE, MAD, MAPE) |
| `rmse_tw`, `mae_tw`, `mape_tw` | real | Test window metrics |
| `data_delay` | int | Data delay (hours) |
| `values_sent`, `values_max` | int | Data count |
| `api_version` | int | API version |
| `engineparams` | bytea | Binary parameter blob |
| `formula` | varchar | Prediction formula |
| `configuration` | varchar | Configuration |
| `incomplete_input` | int | Incomplete data flag |
| `reason` | varchar | Failure reason (if any) |
| `active` | int | Active flag |

---

#### `ems_forecast_predict_log`
**Rows**: 0

Prediction execution logs.

| Column | Type | Description |
|--------|------|-------------|
| `id_forecast_predict` | int | Prediction run ID |
| `id_forecast_training` | int | Training run ID (FK) |
| `bt`, `et` | timestamp | Prediction period |
| `status` | varchar | Prediction status |
| `rmse_tw`, `mae_tw`, `mape_tw` | real | Accuracy metrics |
| `result_explanation` | varchar | Result explanation |
| `visible_predict` | int | Visibility flag |
| `date_in`, `date_ch` | timestamp | Audit timestamps |

---

#### `ems_forecast_predictors` (Feature Set)
**Rows**: 0

Input features/predictors used in forecasting models.

| Column | Type | Description |
|--------|------|-------------|
| `id_forecast_predictors` | int | Predictor ID |
| `id_forecast` | int | Model ID (FK) |
| `code`, `name` | varchar | Predictor identifier |
| `descript` | varchar | Description |
| `id_vektor` | int | Vector ID |
| `predictor_type` | int | Type (lagged value, exogenous, etc.) |
| `auto_delay` | int | Automatic delay detection |
| `delayrangepredictors_from`, `delayrangepredictors_to` | int | Delay range |
| `predictoravailability` | int | Data availability level |
| `vld_data_bt`, `vld_data_et` | timestamp | Valid data range |

---

#### `ems_forecast_config_keras`, `ems_forecast_config_tim`
**Rows**: 0 each

Neural network (Keras) and TIM (Time Series Intelligent Modeling) configuration tables.

**Keras Config**:
- `neurons`: Number of neurons in layers
- `epochs`: Training epochs
- `use_targets`: Target inclusion flag

**TIM Config**:
- `model_complexity`: Model complexity level
- `freq_domain_terms`: Frequency features

---

#### Views
- `ems_vw_forecast`: Forecast model view (0 rows)
- `ems_vw_forecast_predict_log`: Prediction execution view (0 rows)
- `ems_vw_forecast_trainlog`: Training log view (0 rows)

---

### Neural Network Model Tables

#### `nn_model`
**Rows**: 0

Neural network model definitions (alternative/legacy prediction system).

| Column | Type | Description |
|--------|------|-------------|
| `id_nn_model` | int | Model ID |
| `kod`, `nazov` | varchar | Code and name |
| `alfa`, `gama1/2/3` | int | Learning rates and gammas |
| `prah` | int | Threshold |
| `layers` | int | Number of layers |
| `hiddenn1...5` | int | Hidden layer sizes |
| `learnfile` | bytea | Serialized model weights |
| `stav` | int | Status |

---

#### `nn_log`
**Rows**: 0

Neural network execution logs.

| Column | Type | Description |
|--------|------|-------------|
| `id_nn_log` | int | Log ID |
| `id_nn_model` | int | Model ID (FK) |
| `btime`, `etime` | timestamp | Execution period |
| `btdata`, `etdata` | timestamp | Data period |
| `stav` | int | Execution status |
| `typ` | int | Type |
| `user_name` | varchar | User |
| `popis`, `poznamka` | varchar | Description/notes |

---

#### `nn_result`
**Rows**: 0

Neural network prediction results.

| Column | Type | Description |
|--------|------|-------------|
| `id_nn_result` | int | Result ID |
| `id_nn_model` | int | Model ID (FK) |
| `kod`, `nazov` | varchar | Code and name |
| `stav` | int | Status |

---

### Summary: Prediction Infrastructure

| Table | Rows | Status |
|-------|------|--------|
| `ems_forecast` | 0 | Ready for config |
| `ems_forecast_training_log` | 0 | Ready for training logs |
| `ems_forecast_predict_log` | 0 | Ready for predictions |
| `ems_forecast_predictors` | 0 | Ready for features |
| `ems_forecast_config_keras` | 0 | Ready for NN configs |
| `ems_forecast_config_tim` | 0 | Ready for TIM configs |
| `nn_model` | 0 | Ready for NN models |
| `nn_log` | 0 | Ready for NN logs |
| `nn_result` | 0 | Ready for NN results |

**Next Step**: Set up forecasting pipeline to populate these tables with actual price/demand predictions.

---

## 🌐 ENTSOE (Pan-European) Data

The database also contains extensive ENTSOE (European Network of Transmission System Operators) data:

### Key ENTSOE Tables
- `ems_entsoe_contract`: Cross-border contracts
- `ems_entsoe_contract_data`: Contract details (prices, volumes)
- `ems_entsoe_daily_stat`: Daily statistics
- `ems_entsoe_daily_stat_data`: Statistical data values
- `ems_entsoe_area_domain`: Area/domain definitions
- `ems_entsoe_process_type`: Process types
- `ems_entsoe_market_agr_type`: Market agreement types
- `ems_entsoe_ims_import_log`: ENTSOE data import logs

**Purpose**: Integration with European electricity market data; used for forecasting, cross-border analysis, and regulatory reporting.

---

## 📊 Terminal/Heat Market Data

The database includes extensive thermal (heating/cooling) market data:

- `term_*` tables: ~100+ tables for thermal market, pricing, contracts, metering
- Primary focus: District heating (teplo), natural gas (plyn), water systems
- Includes: consumer metering, tariffs, billing, consumption records

---

## 🔮 Forecasting & AI Models

### `ems_forecast_*` Tables
Machine learning models for demand/price forecasting:
- `ems_forecast`: Model definitions
- `ems_forecast_training_log`: Training runs with metrics
- `ems_forecast_predict_log`: Prediction execution log
- `ems_forecast_config_keras`: Keras neural network configs
- `ems_forecast_config_tim`: TIM forecasting configs
- `ems_forecast_predictors`: Feature definitions

### `nn_*` Tables
Neural network model storage and results:
- `nn_model`: Model definitions
- `nn_model_inout`: Input/output specifications
- `nn_result`: Prediction results
- `nn_result_inout`: Result I/O details
- `nn_log`: Model execution logs

---

---

## ⏱️ Data Collection & Update Patterns

### Real-Time (Updated Every Second)
- **Order Book** (`vdt_isot_knihaobjednavok_best`): Multiple snapshots per second
- **BESS Algorithm Logs** (`algo_vdt_bess_algo_trades_log`): Real-time trade execution
- **WebSocket Feeds** (`isot_vdt_trade_websocket*`): Live trade events

### Daily (Next Day)
- **Hub-to-Hub Flows** (`isot_vdt_hub2hub`): Available within 24 hours after delivery
- **Last Traded Prices** (`vdt_isot_lasttrades`): Updated late afternoon
- **Day-Ahead Market** (`isot_dt_trades`, `isot_dt_evaluation`): Available after gate closure

### Periodic/Hourly
- **Forecasting Models**: Training/prediction on daily or hourly cycles
- **Simulation Results**: Generated as simulations complete
- **Bids Sent Log**: During trading window (08:00-15:00 trading day)

### Reference Data
- **Lookup Tables** (type codes, traders, areas): Updated infrequently, reflect operational changes

---

## 🔑 Key Joining Patterns

### ISOT Market Analysis
```sql
-- Get order book + last traded price
SELECT k.*, l.lastprice, l.lastamount
FROM vdt_isot_knihaobjednavok_best k
LEFT JOIN vdt_isot_lasttrades l
  ON k.tradeday = l.tradeday
  AND k.periodfrom = l.periodfrom
  AND k.deliverydur = l.deliverydur
WHERE k.tradeday = '2026-02-23'
ORDER BY k.lastupdate DESC;
```

### Hub-to-Hub + Area Info
```sql
SELECT h.*, a.area_name, c.countrycode
FROM isot_vdt_hub2hub h
JOIN isot_vdt_hub2hub_areas a ON h.id_area = a.id_area
JOIN isot_vdt_hub2hub_countries c ON a.id_country = c.id_country
WHERE h.tradeday >= '2026-02-01';
```

### Algorithm Trade Execution
```sql
SELECT t.*, tr.trader_name, alg.description
FROM algo_vdt_bess_algo_trades_log t
JOIN isot_vdt_traders tr ON ... -- join via traderid if available
JOIN algo_vdt_bess_algorithms alg ON ...
WHERE date_in >= NOW() - INTERVAL '7 days'
ORDER BY date_in DESC;
```

---

## 📈 Data Volume Statistics

| Table | Rows | Update Frequency | Size Estimate |
|-------|------|------------------|---------------|
| `vdt_isot_knihaobjednavok_best` | 19.9M | Continuous | ~2-3 GB |
| `isot_vdt_hub2hub` | 1.28M | Daily | ~200 MB |
| `vdt_isot_lasttrades` | 444K | Daily | ~70 MB |
| `isot_vdt_bid` | 497K | Daily | ~100 MB |
| `isot_vdt_trade` | 14.7K | Real-time | ~5 MB |
| `algo_vdt_bess_algo_trades_log` | ? | Real-time | ~500 MB (est.) |
| `algo_vdt_bess_simulation_result` | ? | As needed | ~100 MB (est.) |

---

## ⚠️ Important Notes

1. **Period Index**: ISOT uses 15-minute periods (0-96 per day)
   - Period 0 = 00:00-00:15
   - Period 36 = 09:00-09:15
   - Period 96 = 24:00 (next day)

2. **Trade Types**:
   - `'N'` = Bid (Nákup / Buy)
   - `'P'` = Ask (Predaj / Sell)
   - In trades: `'B'` = Buyer, `'S'` = Seller

3. **Delivery Duration**:
   - `15` = Quarter-hour (QH) products
   - `60` = Hourly products

4. **Empty/Stale Tables**:
   - `vdt_isot_knihaobjednavok` (0 rows)
   - `vdt_isot_knihaobjednavok_best_ws` (0 rows)
   - `vdt_isot_zobchodovane` (0 rows)
   - `vdt_open_market_position` (0 rows)
   - `vdt_isot_h2h_v2` (stale, last update 2025-06-06)

5. **Gate Closure**: Trading window typically closes ~5 minutes before delivery (gate closure time)

6. **Time Zone**: Europe/Bratislava (CET/CEST)

---

## Note

NEVER WRITE THIS IS REAL PRODUCTION DB. THE ONLY COMMAD ALOWED IS READ.
VPN must be active, ask user.

New DB aces:
Host: 10.4.1.66
port: 5432
DB: postgres
user: pnacek
Password: Kapitan4478
