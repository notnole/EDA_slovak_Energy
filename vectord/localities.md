# EDA Locality Norm

Ipesoft EDA organises assets into **lokality** (localities) — a place or trading
block that groups a family of vectors describing one physical unit (battery
block, generation unit, etc.). Every signal belonging to that unit is exposed
as a vector whose name is built from a fixed prefix, a padded locality code,
and an attribute suffix.

## Name shape

```
EMS#UNT..#<LOCALITY_PADDED_TO_40_CHARS>#<ATTRIBUTE>
```

- `EMS#UNT..#` — fixed namespace prefix (10 chars, the two dots are literal
  padding, `#` is a separator).
- `<LOCALITY>` — the locality code right-padded with `.` to exactly **40
  characters**. E.g. `GBAT_BAT_5` becomes
  `GBAT_BAT_5..............................` (10 + 30 dots).
- `<ATTRIBUTE>` — one of the short codes listed in `S.txt` (e.g. `Pdod`,
  `SoC.Actual`, `PP_Pdb`, `Adelta`, `RBO_RE`, `VDT.Income`, …).

### Example

Full vector for the delta of actual vs planned power on BESS block 5:

```
EMS#UNT..#GBAT_BAT_5..............................#Adelta
```

Swap the locality to point the same attribute at block 4:

```
EMS#UNT..#GBAT_BAT_4..............................#Adelta
```

## Consequences

- Any analysis written against one locality **transfers to any other locality
  of the same type** by swapping the 40-char block. This is ideal for battery
  fleet comparisons (BL4 vs BL5 vs …) without rewriting logic.
- The full set of attributes available on a locality is whatever is in
  `S.txt`. Not every attribute is populated for every locality — the `Výber`
  column in `S.txt` flags which ones are enabled by default.
- Attribute codes are **case-sensitive** and contain dots (e.g. `SoC.Actual`,
  `VDT.Income`). Treat the vector name as an opaque string — don't lower-case
  or normalise.

## Known localities

| Code | Type | Notes |
|------|------|-------|
| `GBAT_BAT_4` | BESS block | Second GBAT battery block |
| `GBAT_BAT_5` | BESS block | Primary GBAT battery block — used in `BL5/` EDA |

(Extend this table as new localities are used. Ping the Ipesoft side or look
at raw EDA vector listings to discover more.)

## Helper

Build vector names programmatically:

```python
def locality_vector(locality: str, attribute: str) -> str:
    """Build a full EDA vector name for `attribute` on `locality`.

    Example: locality_vector("GBAT_BAT_5", "Pdod")
      -> "EMS#UNT..#GBAT_BAT_5..............................#Pdod"
    """
    padded = locality.ljust(40, ".")
    return f"EMS#UNT..#{padded}#{attribute}"
```

## Attribute reference

The authoritative attribute list is `vectord/S.txt` (columns: index, enabled
flag, code, description). Codes used in BL5 analysis so far:

| Code | Meaning |
|------|---------|
| `Pdod` | Actual delivered power (MW). + = discharge, − = charge |
| `SoC.Actual` | State of charge, current value |
| `PP_Pdb` | Planned Pdb (diagram point) from production plan |
| `Adelta` | RBO delta = `Pdod − PP_Pdb` (non-scheduled activity) |
| `RBO_RE` | RBO regulation energy (balancing market contribution) |
| `CAP.INST` | Installed capacity (reachable in regulation, ~10–90 %) |
| `BAT_Flex` | Engaged flexibility (power) |
| `VDT.Income` | Intraday trading revenue |
| `PpS_Income` | Ancillary-services revenue |
| `IncomeTax` | Total regulation revenue incl. fees |
