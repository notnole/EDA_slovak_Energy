# SEPS DAE API Reference - Complete Endpoint Discovery

## Platform Overview

| Property | Value |
|----------|-------|
| Platform | Damas Energy |
| Base URL | `https://dae.sepsas.sk/SK_PROD` |
| Operator | SEPS a.s. (Slovenska elektrizacna prenosova sustava) |
| Access | Public (anonymous login available) |
| Login URL | `/DAEF-GUI/SUC/001System/UCLogin/LoginTest.aspx` |
| API Type | REST-like POST with JSON payload |

---

## API Endpoint

```
POST https://dae.sepsas.sk/SK_PROD/DAEF-GUI/DAE_MVC/DfeDamas/TsPivotTableComponent/LoadData
```

### Required Headers

| Header | Value | Notes |
|--------|-------|-------|
| Content-Type | `application/json` | |
| X-Requested-With | `XMLHttpRequest` | |
| X-XSRF-TOKEN | `<token>` | From `RequestVerification` cookie |
| DFE-Screen-Id | `TS_VIEW_SCREEN` | |
| DFE-Component-Id | `PIVOT_TABLE` | |
| DFE-Timezone-Id | `CET` | |
| Origin | `https://dae.sepsas.sk` | |
| Referer | `https://dae.sepsas.sk/SK_PROD/DAEF-GUI/DAE_MVC/UC/TS_VIEW_SCREEN` | |

### Request Payload

```json
{
  "parameters": [
    {"code": "VIEW_CODE", "value": "\\SS_PUB\\PUB\\DATAFLOW\\REAL_SYSTEM_DATA_GUI"},
    {"code": "TIME_FILTER_UNIT", "value": "DAY"},
    {"code": "INTERVAL_TILL", "value": "2026-01-24T23:00:00.000Z"},
    {"code": "INTERVAL_FROM", "value": "2026-01-23T23:00:00.000Z"},
    {"code": "USER_COND_FORMULA", "value": "1"},
    {"code": "SKIP_VIEW_CONTEXT", "value": true},
    {"code": "CHART_TYPE", "value": null},
    {"code": "VIEW_TYPE", "value": "TimeseriesView"},
    {"code": "QUERY", "value": "#?VIEW_CODE=%5CSS_PUB%5CPUB%5CDATAFLOW%5CREAL_SYSTEM_DATA_GUI"}
  ]
}
```

### Parameter Details

| Parameter | Values | Notes |
|-----------|--------|-------|
| `VIEW_CODE` | See view list below | Full path with backslashes |
| `TIME_FILTER_UNIT` | `DAY`, `HOUR`, `QUARTER_HOUR` | Resolution hint (view determines actual) |
| `INTERVAL_FROM` | ISO 8601 UTC | Start of requested range |
| `INTERVAL_TILL` | ISO 8601 UTC | End of requested range |
| `USER_COND_FORMULA` | `"1"` | Always "1" for public |
| `SKIP_VIEW_CONTEXT` | `true` | Skip view context loading |
| `VIEW_TYPE` | `"TimeseriesView"` | Always TimeseriesView |

### Response Structure

```json
{
  "gridConfig": {
    "view": {"name": "...", "code": "...", "oid": "..."},
    "sheets": [{
      "colModel": [{"cellData": {"v": "column_title"}, "title": "..."}],
      "dataModel": {
        "data": [
          [{"v": "2026-01-24"}, {"v": "00:00"}, {"v": 4521.3}, ...]
        ]
      }
    }],
    "timeSerieConfigurations": {
      "0": {"code": "TS_CODE", "timeSeriesTooltip": "...", "storageType": "..."}
    },
    "noDataFound": false
  },
  "debugMessage": ""
}
```

---

## Authentication Flow

1. **Playwright login**: Navigate to login page, click "Public access" (Verejny pristup)
2. **Cookie extraction**: Save session cookies (`.seps_cookies.json`)
3. **Requests API calls**: Use cookies + XSRF token for fast API access (~0.5s per call)

Session cookies typically last several hours. Refresh by re-running Playwright login on auth failure (HTTP 401/302).

---

## Hierarchy Structure

```
SS_PUB (Subsystem: Publishing data)
  └── PUB (Module: Publication)
        ├── [13 Views directly under PUB (market coupling, capacity, auctions)]
        ├── DATAFLOW (Component: Dataflow values)
        │     └── [Operational real-time/historical views]
        ├── PUB_LOAD_ES (Component: Load - EMFIP)
        │     └── [6 Load views]
        ├── PUB_GENERATION_ES (Component: Generation - EMFIP)
        │     └── [7 Generation views]
        ├── PUB_BALANCING (Component: Balancing - EMFIP)
        │     └── [9 Balancing views]
        ├── PUB_OUTAGES (Component: Outages - EMFIP)
        │     └── [8 Outage views]
        ├── PUB_TRANSMISSION (Component: Transmission - EMFIP)
        │     └── [11 Transmission views]
        └── PUB_CONGESTION_MNG (Component: Congestion management - EMFIP)
              └── [3 Congestion management views]
```

VIEW_CODE format: `\SS_PUB\PUB\{COMPONENT}\{VIEW_NAME}`

---

## Complete View Catalog

### 1. DATAFLOW Component (`\SS_PUB\PUB\DATAFLOW\...`)

Operational data views for the Slovak electricity system.

| VIEW_CODE | Name | Resolution | History | Description |
|-----------|------|-----------|---------|-------------|
| `SYSTEM_STATE_MAP_VIEW` | System state (real-time) | 3-min | **NONE** (live only, `isHistorized: false`) | Load, production, balance, frequency, cross-border flows, balancing energy. Real-time only - ignores date range parameters. |
| `REAL_SYSTEM_DATA_GUI` | Real system data | 15-min | **730+ days** | Load MW, production MW, system balance, regulation power/energy. 17,281 rows for 180 days in 1.3s. |
| `CROSSBOARD_FLOW_OVERVIEW_GUI` | Cross-border flow overview | 15-min | weeks+ | Per-profile cross-border flows and totals (CZ, HU, PL, UA directions). |
| `SYSTEM_PLAN_OVERVIEW_GUI` | System plan overview | Hourly | ~1 day | Day-ahead plan: load prediction, production/balance plan. |
| `HOUR_AVERAGE_OVERVIEW` | Hourly averages | Hourly | 7+ days | Load averages per hour per day. |
| `LOAD_VIEW` | Daily load peaks | Daily | **365+ days** | Max/min load, peak hour, frequency at peak. |
| `PRODUCTION_CONSUMPTION_VIEW` | Production/consumption | Daily | weeks | Daily MWh: production, consumption, balance totals. ~5 day publication delay. |
| `CROSSBORDER_FLOWS` | Cross-border flows (chart) | 15-min | weeks+ | Same data as CROSSBOARD_FLOW_OVERVIEW_GUI, chart format. |
| `SYSTEM_PRODUCTION_AND_LOAD` | System production & load (chart) | 15-min | weeks+ | Chart-formatted production and load data. |

#### REAL_SYSTEM_DATA_GUI - Column Details (Primary Data Source)

Columns from `timeSerieConfigurations`:
- `LOAD_VALUE` - System load (MW)
- `GENERATION_VALUE` - Total generation (MW)
- `SYSTEM_BALANCE` - System balance (MW)
- `REG_POWER_PLUS` - Regulation power + (MW)
- `REG_POWER_MINUS` - Regulation power - (MW)
- `REG_ENERGY_PLUS` - Regulation energy + (MWh)
- `REG_ENERGY_MINUS` - Regulation energy - (MWh)

#### SYSTEM_STATE_MAP_VIEW - Fields (Real-Time Only)

| Field | Description | Typical Range |
|-------|-------------|---------------|
| frequency_hz | System frequency | 49.95 - 50.05 Hz |
| load_mw | Total system load | 3500 - 5500 MW |
| production_mw | Total production | 3500 - 5500 MW |
| balance_mw | System balance | -200 to +200 MW |
| cross_border_cz_mw | CZ border flow | -1500 to +1500 MW |
| cross_border_hu_mw | HU border flow | -1000 to +1000 MW |
| cross_border_pl_mw | PL border flow | -1000 to +1000 MW |
| cross_border_ua_mw | UA border flow | -500 to +500 MW |
| balancing_energy_mw | Balancing energy | -500 to +500 MW |

---

### 2. PUB_LOAD_ES Component (`\SS_PUB\PUB\PUB_LOAD_ES\...`)

Load data per EMFIP regulation articles.

| VIEW_CODE | Name | EMFIP Article | Description |
|-----------|------|---------------|-------------|
| `METERED_TOTAL_LOAD_GUI` | Load ES SR - Actual | 6.1.A | Metered total load of the system |
| `PREDICTED_LOAD_GUI` | Load ES SR - Daily prediction | 16.1.B | Day-ahead load forecast |
| `MIN_MAX_WEEK_LOAD` | Load ES SR - Weekly prediction | 16.1.C | Min/max weekly load forecast |
| `MIN_MAX_MONTH_LOAD` | Load ES SR - Monthly prediction | 16.1.D | Min/max monthly load forecast |
| `MIN_MAX_YEAR_LOAD` | Load ES SR - Yearly prediction | 16.1.E | Min/max yearly load forecast |
| `YEARLY_FORECAST_BALANCE_VIEW` | Balance ES SR - Yearly prediction | 8.1 | Yearly balance forecast (gen - load) |

---

### 3. PUB_GENERATION_ES Component (`\SS_PUB\PUB\PUB_GENERATION_ES\...`)

Generation data per EMFIP regulation articles.

| VIEW_CODE | Name | EMFIP Article | Description |
|-----------|------|---------------|-------------|
| `VIEW_YEARLY_AG_P_EMFIP` | Installed power ES SR | 14.1.A | Installed generation capacity aggregated by type |
| `INSTALLED_GEN_CAPACITY_UNIT` | Installed power per unit >100MW | 14.1.B | Installed capacity of each unit over 100MW |
| `PREDICTED_GENERATION_GUI` | Generation - daily prediction | 14.1.C | Day-ahead generation forecast |
| `PREDICTION_SOLAR_WIND_GUI` | Wind and Solar forecast | 14.1.D | Solar/wind forecast: day-ahead, intraday-forecast, intraday-current |
| `GENERATION_PER_UNIT_GUI` | Generation per unit >100MW - actual | 16.1.A | **Actual generation per individual unit (>100MW)** |
| `AGGREGATED_GENERATION_PER_TYPE_GUI` | Generation per type - actual | 16.1.B, 16.1.C | **Actual generation by fuel type (13 types). Hourly, 730+ days history.** |
| `FILLING_WATER_RESERVOIRS_VIEW` | Water reservoirs filling rate | 16.1.D | Aggregated filling rate of water reservoirs |

#### AGGREGATED_GENERATION_PER_TYPE_GUI - Fuel Types (13 columns)

| Column | Typical MW | Description |
|--------|-----------|-------------|
| Jadrov'a | ~2465 | Nuclear |
| Hned'e uhlie | ~200-400 | Brown coal (lignite) |
| Cierne uhlie | 0-50 | Black coal |
| Zemn'y plyn | ~550-600 | Natural gas |
| Vodn'a (prietocn'a) | ~100-300 | Run-of-river hydro |
| Vodn'a (prec.) | variable | Pumped-storage hydro |
| Slnecn'a | 0-157 (daily cycle) | Solar PV |
| Vetern'a | 0-5 | Wind |
| Biomasa | ~100-200 | Biomass |
| Bioplyn | ~50-100 | Biogas |
| Komun'alny odpad | ~20-40 | Municipal waste |
| Ostatn'e OZE | ~5-20 | Other RES |
| In'e | variable | Other |

**Performance**: 365 days = 8,761 rows in 0.67 seconds.

---

### 4. PUB_BALANCING Component (`\SS_PUB\PUB\PUB_BALANCING\...`)

Balancing and ancillary services data per EMFIP regulation.

| VIEW_CODE | Name | EMFIP Article | Description |
|-----------|------|---------------|-------------|
| `ANS_SERVICES_CONTRACTS_YEARLY_VIEW` | Contracts ANS - yearly | 17.1.B, 17.1.C | Yearly ancillary services contracts |
| `ANS_SERVICES_CONTRACTS_MONTHLY_VIEW` | Contracts ANS - monthly | 17.1.B, 17.1.C | Monthly ancillary services contracts |
| `ANS_SERVICES_CONTRACTS_DAILY_VIEW_GUI` | Contracts ANS - daily | 17.1.B, 17.1.C | Daily ancillary services contracts |
| `BALANCING_RESERVE_AGGREGATE_POWER_VIEW` | Reserve power from daily OPS | 17.1.D | Accepted aggregated power from daily operation scheduling |
| `DAILY_REGULATION_ENERGY_VIEW_GUI` | Balancing energy (daily) | 17.1.E, 17.1.F | Daily balancing/regulation energy |
| `OVERAL_MONTHLY_RE_COSTS` | Monthly balancing costs | 17.1.I | Monthly total costs and income from balancing |
| `CB_RESERVE_ACTIVE_EMFIP_VIEW` | Cross-border reserve & BE | 17.1.J | Cross-border reserve and activated cross-border balancing energy |
| `PROCURED_ANS_DAY_VIEW` | Daily ANS procurement | - | Daily procured ancillary services |
| `BAL_RESERVE_BIDS_VIEW` | BE bids from daily OPS | 12.3.B | Anonymized balancing energy bids (volume + price) |

---

### 5. PUB_OUTAGES Component (`\SS_PUB\PUB\PUB_OUTAGES\...`)

Outage information per EMFIP regulation (planned and unplanned outages for generation, production, consumption units and transmission assets).

| VIEW_CODE | Name | EMFIP Article | Description |
|-----------|------|---------------|-------------|
| `OUTAGE_PLAN_CON` | Plan of consumption unit outage | 7.1.A | Planned outages of consumption units |
| `GU_FAILURE_PUBLIC_VIEW_CONS` | Consumption units outage | 10.1.B | Unplanned consumption unit outages |
| `OUTAGE_PLAN_TRANSMISSION_ASSETS` | Plan of transmission assets | 10.1.A | Planned outages of transmission infrastructure (lines, transformers) |
| `FAILURES_TRANSMISSION_ASSETS` | Transmission assets outage | 10.1.B | Unplanned transmission asset outages |
| `OUTAGE_PLAN_GEN` | Plan of generation unit outage | 15.1.A | Planned outages of generation units |
| `GU_FAILURE_PUBLIC_VIEW_GEN` | Generation units outage | 15.1.B | Unplanned generation unit outages |
| `OUTAGE_PLAN_PROD` | Plan of production unit outage | 15.1.C | Planned outages of production units (power plants) |
| `GU_FAILURE_PUBLIC_VIEW_PROD` | Production units outage | 15.1.D | Unplanned production unit outages |

---

### 6. PUB_TRANSMISSION Component (`\SS_PUB\PUB\PUB_TRANSMISSION\...`)

Transmission network data per EMFIP regulation (NTC, capacity allocation, market prices, cross-border flows).

| VIEW_CODE | Name | EMFIP Article | Description |
|-----------|------|---------------|-------------|
| `EXPANSION_DISMANTLING_PROJECT_VIEW` | Change of transmission asset (yearly plan) | 9.1 | Yearly plan for expanding and dismantling transmission assets |
| `ANNUAL_NTC_OVERVIEW` | NTC - Annual | 11.1 | Annual Net Transfer Capacity values per border direction |
| `MONTHLY_NTC_OVERVIEW` | NTC - Monthly | - | Monthly NTC/ATC values per border direction |
| `EXPL_NOMIN_CAPACITY_VIEW` | Nominated capacity (explicit auction) | 12.1.B | Nominated capacity from explicit auctions per border |
| `DAILY_PRICES_SK` | Daily market (price) | 12.1.D | Daily market prices for domestic control area (SK) |
| `IMLICIT_AUCTION_RESULTS_EMFIP` | Implicit auction - results | 12.1.E | Implicit auction results (market coupling) |
| `CROSS_BORDER_PLANS_VIEW` | Total scheduled commercial exchanges | 12.1.F | Total scheduled commercial exchanges from explicit+implicit allocations (daily and intraday). Split: post-20.2.2019 (per horizon) and pre-20.2.2019 (legacy). |
| `PHYSICAL_FLOW_VIEW_GUI` | Metered crossborder flow | 12.1.G | Actual metered physical cross-border flows |
| `VIEW_AAC_DAILY_THIRD_COUNTRIES` | AAC - daily (UA) | 12.1.H | Daily Already Allocated Capacity for third countries (Ukraine) |
| `VIEW_AAC_MONTHLY_THIRD_COUNTRIES` | AAC - monthly (UA) | 12.1.H | Monthly AAC for third countries (Ukraine) |
| `VIEW_AAC_YEARLY_THIRD_COUNTRIES` | AAC - yearly (UA) | 12.1.H | Yearly AAC for third countries (Ukraine) |

---

### 7. PUB_CONGESTION_MNG Component (`\SS_PUB\PUB\PUB_CONGESTION_MNG\...`)

Congestion management data per EMFIP regulation (redispatching, countertrading, costs).

| VIEW_CODE | Name | EMFIP Article | Description |
|-----------|------|---------------|-------------|
| `REDISPATCHING_VIEW` | Redispatching | 13.1.A | Redispatching actions taken for congestion management |
| `VIEW_COUNTERTRADE` | Countertrading | 13.1.B | Countertrading actions for congestion management |
| `CNG_MNG_COSTS` | Costs of congestion management | 13.1.C | Total costs of congestion management measures |

---

### 8. Direct PUB Views (`\SS_PUB\PUB\...`)

Views directly under the PUB module (not in a sub-component).

| VIEW_CODE | Name | Description |
|-----------|------|-------------|
| `DAILY_OC_FROM_EA_GUI` | Daily offered capacity (entsoe.net) | Daily offered capacity from explicit auctions sent to ENTSO-E |
| `MONTHLY_OC_FROM_EA` | Monthly offered capacity | Monthly offered capacity from explicit auctions |
| `YEARLY_OC_FROM_EA` | Yearly offered capacity | Yearly offered capacity from explicit auctions |
| `SUM_DA_RESULTS_TO_ETSOVISTA_OVERVIEW_GUI` | Daily auction results | Summary results of daily explicit auction |
| `SUM_MA_RESULTS_TO_ETSOVISTA_OVERVIEW` | Monthly auction results | Summary results of monthly explicit auction |
| `SUM_DA_RESULTS_TO_ENTOSENET_OVERVIEW` | Yearly auction results | Summary results of yearly explicit auction |
| `USED_CAPACITY_ALLOCATED_OVERVIEW_GUI` | Capacity allocation overview | Capacity allocation and utilization by control area and type |
| `AGGREGATE_PRODUCED_BE_OVERVIEW` | Activated RE by production type | Balancing energy (FCR, aFRR, mFRR) by fuel type. Sub-1 MW values. |
| `EVALUATION_GUI` | Monthly AnS evaluation | Monthly overview of ancillary services evaluation (for URSO) |
| `SLOVAKIA_WATER_RESERVOIS` | Water reservoir filling rates | Weekly reservoir levels (%). Published since August 2010. |
| `MC_STATISTICS_DAILY` | Market Coupling Statistics | Daily market coupling results (SDAC/SIDC) |
| `IA_OVERVIEW` | Monthly Market Coupling Statistics | Monthly aggregated market coupling stats |
| `GU_FAILURE_PUBLIC_VIEW` | Generator outages >100MW | Unplanned unavailability of generation units over 100MW |

---

## Data Availability Summary

### Historical Data Depth (Confirmed by Testing)

| View | Resolution | Max History | Rows/Year | Speed |
|------|-----------|-------------|-----------|-------|
| REAL_SYSTEM_DATA_GUI | 15 min | 730+ days | ~35,000 | 17k rows in 1.3s |
| AGGREGATED_GENERATION_PER_TYPE_GUI | 1 hour | 730+ days | ~8,760 | 8.7k rows in 0.7s |
| LOAD_VIEW | Daily | 365+ days | ~365 | Fast |
| CROSSBOARD_FLOW_OVERVIEW_GUI | 15 min | weeks+ | ~672/week | Fast |
| HOUR_AVERAGE_OVERVIEW | 1 hour | 7+ days | ~168/week | Fast |
| SYSTEM_STATE_MAP_VIEW | 3 min (live) | **0 (real-time only)** | N/A | Fast |

### Real-Time vs Historical

| Category | Real-Time (3-min) | Historical (15-min/hourly) |
|----------|-------------------|---------------------------|
| System load | SYSTEM_STATE_MAP_VIEW | REAL_SYSTEM_DATA_GUI, METERED_TOTAL_LOAD_GUI |
| Production | SYSTEM_STATE_MAP_VIEW | REAL_SYSTEM_DATA_GUI, AGGREGATED_GENERATION_PER_TYPE_GUI |
| Balance | SYSTEM_STATE_MAP_VIEW | REAL_SYSTEM_DATA_GUI |
| Frequency | SYSTEM_STATE_MAP_VIEW | *(not available historically)* |
| Cross-border | SYSTEM_STATE_MAP_VIEW | CROSSBOARD_FLOW_OVERVIEW_GUI |
| By fuel type | *(not available)* | AGGREGATED_GENERATION_PER_TYPE_GUI |
| Per unit (>100MW) | *(not available)* | GENERATION_PER_UNIT_GUI |
| Solar/wind forecast | *(not available)* | PREDICTION_SOLAR_WIND_GUI |

---

## Key Findings

1. **No 3-minute historical data exists.** The `SYSTEM_STATE_MAP_VIEW` has `isHistorized: false` and ignores date range parameters entirely. It always returns the current state.

2. **15-minute historical data goes back 2+ years.** `REAL_SYSTEM_DATA_GUI` provides load, production, and balance at 15-minute resolution with 730+ days of history.

3. **Hourly generation per fuel type** available via `AGGREGATED_GENERATION_PER_TYPE_GUI` with 2+ years history and 13 fuel type columns.

4. **Per-unit generation (>100MW)** available via `GENERATION_PER_UNIT_GUI` - provides actual output of each individual generation unit exceeding 100MW capacity.

5. **Solar/wind forecasts** available via `PREDICTION_SOLAR_WIND_GUI` with day-ahead, intraday-forecast, and intraday-current horizons.

6. **Water reservoir levels** available weekly since 2010 via `SLOVAKIA_WATER_RESERVOIS` and `FILLING_WATER_RESERVOIRS_VIEW`.

7. **Large batch requests work.** 180+ days of 15-min data (17,000+ rows) can be fetched in a single request in ~1.3 seconds.

8. **All data is publicly accessible** via anonymous login ("Public access" / "Verejny pristup").

---

## Usage Examples

### Fetch 1 Day of Generation Per Type

```python
import requests, json
from datetime import datetime, timedelta, timezone

session = requests.Session()
# Load cookies from .seps_cookies.json
with open(".seps_cookies.json") as f:
    for c in json.load(f):
        session.cookies.set(c["name"], c["value"], domain=c["domain"], path=c["path"])

# Get XSRF token
xsrf = next(c.value for c in session.cookies if "RequestVerification" in c.name)

now = datetime.now(timezone.utc)
payload = {
    "parameters": [
        {"code": "VIEW_CODE", "value": "\\SS_PUB\\PUB\\PUB_GENERATION_ES\\AGGREGATED_GENERATION_PER_TYPE_GUI"},
        {"code": "TIME_FILTER_UNIT", "value": "DAY"},
        {"code": "INTERVAL_TILL", "value": now.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
        {"code": "INTERVAL_FROM", "value": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")},
        {"code": "USER_COND_FORMULA", "value": "1"},
        {"code": "SKIP_VIEW_CONTEXT", "value": True},
        {"code": "CHART_TYPE", "value": None},
        {"code": "VIEW_TYPE", "value": "TimeseriesView"},
        {"code": "QUERY", "value": "#?VIEW_CODE=%5CSS_PUB%5CPUB%5CPUB_GENERATION_ES%5CAGGREGATED_GENERATION_PER_TYPE_GUI"},
    ]
}

r = session.post(
    "https://dae.sepsas.sk/SK_PROD/DAEF-GUI/DAE_MVC/DfeDamas/TsPivotTableComponent/LoadData",
    json=payload,
    headers={
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "X-XSRF-TOKEN": xsrf,
        "DFE-Screen-Id": "TS_VIEW_SCREEN",
        "DFE-Component-Id": "PIVOT_TABLE",
        "DFE-Timezone-Id": "CET",
    },
    verify=False,
    timeout=30,
)

data = r.json()
rows = data["gridConfig"]["sheets"][0]["dataModel"]["data"]
print(f"Got {len(rows)} hourly records")
for row in rows[:5]:
    cells = [c.get("v", "") if isinstance(c, dict) else c for c in row]
    print(cells[:6])
```

### Fetch Historical Load (15-min, 30 days)

```python
payload["parameters"][0]["value"] = "\\SS_PUB\\PUB\\DATAFLOW\\REAL_SYSTEM_DATA_GUI"
payload["parameters"][3]["value"] = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
payload["parameters"][8]["value"] = "#?VIEW_CODE=%5CSS_PUB%5CPUB%5CDATAFLOW%5CREAL_SYSTEM_DATA_GUI"
# ... same POST call
```

---

## Browser URL Format

Views can be opened directly in a browser after login:

```
https://dae.sepsas.sk/SK_PROD/DAEF-GUI/DAE_MVC/UC/TS_VIEW_SCREEN#?VIEW_CODE=%5CSS_PUB%5CPUB%5CPUB_GENERATION_ES%5CAGGREGATED_GENERATION_PER_TYPE_GUI&FROM=2026-0-23-23-0
```

URL encoding: `\` -> `%5C`

---

## EMFIP Data Item Reference

The views follow EU Regulation 543/2013 (EMFIP - Electricity Market Fundamental Information Platform):

| Article | Data Item | View |
|---------|-----------|------|
| 6.1.A | Actual total load | METERED_TOTAL_LOAD_GUI |
| 8.1 | Yearly balance forecast | YEARLY_FORECAST_BALANCE_VIEW |
| 14.1.A | Installed capacity aggregated | VIEW_YEARLY_AG_P_EMFIP |
| 14.1.B | Installed capacity per unit >100MW | INSTALLED_GEN_CAPACITY_UNIT |
| 14.1.C | Day-ahead generation forecast | PREDICTED_GENERATION_GUI |
| 14.1.D | Wind/solar generation forecast | PREDICTION_SOLAR_WIND_GUI |
| 16.1.A | Actual generation per unit >100MW | GENERATION_PER_UNIT_GUI |
| 16.1.B | Actual generation per type | AGGREGATED_GENERATION_PER_TYPE_GUI |
| 16.1.C | Actual generation per type (RES) | AGGREGATED_GENERATION_PER_TYPE_GUI |
| 16.1.D | Water reservoir filling rates | FILLING_WATER_RESERVOIRS_VIEW |
| 16.1.B (load) | Day-ahead load forecast | PREDICTED_LOAD_GUI |
| 16.1.C (load) | Weekly load forecast | MIN_MAX_WEEK_LOAD |
| 16.1.D (load) | Monthly load forecast | MIN_MAX_MONTH_LOAD |
| 16.1.E (load) | Yearly load forecast | MIN_MAX_YEAR_LOAD |
| 17.1.B | AnS contracts | ANS_SERVICES_CONTRACTS_*_VIEW |
| 17.1.C | AnS contracts | ANS_SERVICES_CONTRACTS_*_VIEW |
| 17.1.D | Reserve power from OPS | BALANCING_RESERVE_AGGREGATE_POWER_VIEW |
| 17.1.E | Balancing energy | DAILY_REGULATION_ENERGY_VIEW_GUI |
| 17.1.F | Balancing energy | DAILY_REGULATION_ENERGY_VIEW_GUI |
| 17.1.I | Monthly balancing costs | OVERAL_MONTHLY_RE_COSTS |
| 17.1.J | Cross-border reserve/BE | CB_RESERVE_ACTIVE_EMFIP_VIEW |
| **Outages** | | |
| 7.1.A | Plan of consumption unit outage | OUTAGE_PLAN_CON |
| 10.1.A | Plan of transmission assets outage | OUTAGE_PLAN_TRANSMISSION_ASSETS |
| 10.1.B | Unplanned transmission/consumption outage | FAILURES_TRANSMISSION_ASSETS, GU_FAILURE_PUBLIC_VIEW_CONS |
| 15.1.A | Plan of generation unit outage | OUTAGE_PLAN_GEN |
| 15.1.B | Unplanned generation unit outage | GU_FAILURE_PUBLIC_VIEW_GEN |
| 15.1.C | Plan of production unit outage | OUTAGE_PLAN_PROD |
| 15.1.D | Unplanned production unit outage | GU_FAILURE_PUBLIC_VIEW_PROD |
| **Transmission** | | |
| 9.1 | Expansion/dismantling of transmission assets | EXPANSION_DISMANTLING_PROJECT_VIEW |
| 11.1 | Annual NTC values | ANNUAL_NTC_OVERVIEW |
| 12.1.B | Nominated capacity (explicit auction) | EXPL_NOMIN_CAPACITY_VIEW |
| 12.1.D | Daily market price | DAILY_PRICES_SK |
| 12.1.E | Implicit auction results | IMLICIT_AUCTION_RESULTS_EMFIP |
| 12.1.F | Total scheduled commercial exchanges | CROSS_BORDER_PLANS_VIEW |
| 12.1.G | Metered crossborder flow | PHYSICAL_FLOW_VIEW_GUI |
| 12.1.H | AAC third countries (UA) | VIEW_AAC_DAILY/MONTHLY/YEARLY_THIRD_COUNTRIES |
| **Congestion** | | |
| 13.1.A | Redispatching | REDISPATCHING_VIEW |
| 13.1.B | Countertrading | VIEW_COUNTERTRADE |
| 13.1.C | Costs of congestion management | CNG_MNG_COSTS |

---

## Discovery Method

This reference was compiled by:
1. Playwright-based authenticated navigation of the DAE system hierarchy (all 7 components fully enumerated)
2. Systematic API probing with date range variations
3. Response analysis (timeSerieConfigurations, column headers, data rows)
4. Testing maximum historical depth per view (up to 730 days)

**Total views discovered: 57** (13 PUB + 9 DATAFLOW + 6 PUB_LOAD_ES + 7 PUB_GENERATION_ES + 9 PUB_BALANCING + 8 PUB_OUTAGES + 11 PUB_TRANSMISSION + 3 PUB_CONGESTION_MNG — some views appear in multiple components, unique count ~50+)

All views listed have been confirmed accessible from the LoadData endpoint with the public access session. All components are marked as "Public object" in the access rights table.

---

*Last updated: 2026-01-24*
*Discovery performed via Playwright + requests hybrid approach*
