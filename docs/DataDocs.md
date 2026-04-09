# Legacy Table: `db_ems.vdt_isot_knihaobjednavok_best`

## Database Access

**Connection Details**:
- **Database**: `DB_EMS`
- **Host**: `localhost`
- **Port**: `5432`
- **User**: `postgres`
- **Password**: `Kapitan1`
- **Schema**: `db_ems`
- **Table**: `vdt_isot_knihaobjednavok_best`

**Connection String**:
```
postgresql://postgres:Kapitan1@localhost:5432/DB_EMS
```

**psql Command**:
```bash
psql -U postgres -d DB_EMS -h localhost -p 5432
```

---

## Table Overview

Database view containing Slovak intraday continuous trading (ISOT) order book data with best bid/ask prices from OKTE (Slovak electricity market operator).

**Key Characteristics**:
- Bid and Ask data stored in **SEPARATE rows**
- Best prices only (depth level 1)
- Both 60-minute and 15-minute delivery products
- Timezone: Europe/Bratislava

---

## Table Structure

### Column Definitions

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `tradeday` | TIMESTAMP | Trading day | `2025-09-01 00:00:00` |
| `periodfrom` | INTEGER | Start period index (15-min intervals from midnight) | `36` (09:00) |
| `periodto` | INTEGER | End period index | `39` (09:45) |
| `deliverydur` | INTEGER | Delivery duration in minutes | `60` or `15` |
| `tradetype` | VARCHAR(1) | Trade type: 'N' = bid (Nákup), 'P' = ask (Predaj) | `'N'` |
| `price` | NUMERIC | Price in EUR/MWh | `85.50` |
| `lastupdate` | TIMESTAMP | When this data was recorded | `2025-09-01 08:30:15` |
| `id_depth` | INTEGER | Order book depth level (1 = best) | `1` |

### Period Index Calculation

Period indexes represent 15-minute intervals starting from midnight:
- Period 0 = 00:00-00:15
- Period 1 = 00:15-00:30
- Period 36 = 09:00-09:15
- Period 96 = 24:00 (next day midnight)

**Formula**: `period_index = (hour * 4) + (minute / 15)`

---

## Data Characteristics

### Trade Types

- **'N' (Nákup)**: Bid prices - buyer's offer
- **'P' (Predaj)**: Ask prices - seller's offer

**Important**: For each delivery period, there are **TWO rows**:
- One row with `tradetype='N'` containing bid price
- One row with `tradetype='P'` containing ask price

### Delivery Durations

1. **60-minute products** (`deliverydur=60`):
   - Four 15-minute periods grouped together
   - `periodfrom` and `periodto` span 4 periods
   - Example: Period 36-39 = 09:00-10:00

2. **15-minute products** (`deliverydur=15`):
   - Single 15-minute period
   - `periodfrom` and `periodto` differ by 1
   - Example: Period 36-37 = 09:00-09:15

### Depth Levels

- `id_depth=1`: Best bid/best ask (most competitive prices)
- Higher values: Less competitive prices

---

## Example Queries

### Get Bid/Ask for Specific Period

```sql
-- Combine bid and ask into single row
SELECT
    lastupdate,
    periodfrom,
    periodto,
    MAX(CASE WHEN tradetype = 'N' THEN price END) as bid_price,
    MAX(CASE WHEN tradetype = 'P' THEN price END) as ask_price
FROM db_ems.vdt_isot_knihaobjednavok_best
WHERE tradeday::DATE = '2025-09-01'
  AND periodfrom = 36
  AND deliverydur = 60
  AND id_depth = 1
GROUP BY lastupdate, periodfrom, periodto;
```

### Get All Snapshots for a Day

```sql
SELECT
    lastupdate,
    periodfrom,
    periodto,
    MAX(CASE WHEN tradetype = 'N' THEN price END) as bid_price,
    MAX(CASE WHEN tradetype = 'P' THEN price END) as ask_price
FROM db_ems.vdt_isot_knihaobjednavok_best
WHERE tradeday::DATE = '2025-09-01'
  AND deliverydur = 60
  AND id_depth = 1
GROUP BY lastupdate, periodfrom, periodto
ORDER BY lastupdate, periodfrom;
```

### Calculate Delivery Start Time

```sql
SELECT
    tradeday::DATE + (periodfrom * INTERVAL '15 minutes') as delivery_start,
    periodfrom,
    periodto,
    tradetype,
    price,
    lastupdate
FROM db_ems.vdt_isot_knihaobjednavok_best
WHERE tradeday::DATE = '2025-09-01'
  AND deliverydur = 60
  AND id_depth = 1
ORDER BY lastupdate, periodfrom;
```

---

## Database Indexes

The table has the following indexes to optimize query performance:

### 1. `idx_isot_tradeday_deliverydur_depth`

```sql
CREATE INDEX idx_isot_tradeday_deliverydur_depth
ON db_ems.vdt_isot_knihaobjednavok_best
USING btree (tradeday, deliverydur, id_depth);
```

**Purpose**: Optimizes the main data loading query used by `data_loader.py` which filters on `tradeday`, `deliverydur`, and `id_depth=1`.

### 2. `vdt_isot_knihaobjednavok_best_idx`

```sql
CREATE INDEX vdt_isot_knihaobjednavok_best_idx
ON db_ems.vdt_isot_knihaobjednavok_best
USING btree (tradeday, deliverydur, tradetype);
```

**Purpose**: Original index for queries filtering by trade type (bid/ask).

### Creating the Index

If the index doesn't exist, create it with:

```sql
CREATE INDEX idx_isot_tradeday_deliverydur_depth
ON db_ems.vdt_isot_knihaobjednavok_best (tradeday, deliverydur, id_depth);
```

This significantly improves query performance for loading market data (from ~10s to ~4s for a day's data).
