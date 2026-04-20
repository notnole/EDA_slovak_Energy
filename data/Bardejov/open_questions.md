# TEHO Bardejov — Open Questions

Questions about how TEHO Bardejov actually operates the plant and participates
in the electricity market. Answers will refine our understanding and sharpen
optimization recommendations.

## Market participation

### 1. Do they only bid day-ahead based on predicted heat load?

**Answer (Jan 2025):** Yes — the committed plan is 100% heat-driven, with zero
price sensitivity. Verified against `kalkulácie TeHo BJ 2025 01.XLSX`:

| Test | Result |
| --- | --- |
| `PLAN == CEIL(MW_EE + Marič)` for all hours | 744 / 744 match |
| Regression R² (heat + DA price) | 0.896 |
| Regression R² (heat only) | 0.894 (price adds +0.002) |
| Heat coefficient | 0.247 (t=74) — recovers the 1/4 heat→EE ratio |
| DA-price coefficient | 0.00074 MW per €/MWh (t=3.96 but economically nil) |

A 100 €/MWh DA-price swing shifts the plan by 0.07 MW — rounding noise. The
t-stat on price is driven purely by heat/price winter correlation (r=0.33),
not by any operator decision.

**Committed step distribution (Jan 2025):**
`{3 MW: 30h, 4 MW: 112h, 5 MW: 187h, 6 MW: 410h (55%), 7 MW: 5h}`. Steps 1, 2,
8 were never planned. The 7 MW step is a formula artifact (`CEIL` of 6.x) —
not a physical plant step — so execution must snap to 6 or 8 on those 5 hours.

**Implication:** with subsidies removed in 2025, a pure heat-driven plan leaves
all price-capture value on the table. This is the 324 kEUR/yr opportunity
already flagged in the project memory.

Remaining sub-questions (not yet tested):
- Are DA bids submitted as price-taking or price-limited?
- Dissipator as implicit price buffer — see Q2.

### 2. Do they do any conditional bids?

Examples of conditional bids that may or may not be in play:

- **Price-contingent block bids**: "produce 5 MW for hours 07–20 only if average
  DA price > X EUR/MWh."
- **All-or-nothing blocks**: aligned with minimum stable operation of one or
  two boilers.
- **Linked bids**: start-up / shut-down linked across hours to respect ramp
  constraints (+0.5 / -1.0 MW per 15 min).
- **Intraday (IDM) adjustments**: the `Porovnanie` legend has a color for
  "Zobchodované na intraday" (traded on intraday) — how often and on what
  trigger?
- **Plan revisions**: legend also has "Zmena plánu oproti pôvodnému" (plan
  change vs original) — who authorizes these and what drives them (heat
  surprise, price moves, boiler trip)?

**Open question:** Do they use any structured conditional/block products on
OKTE, or is everything submitted as flat hourly quantity bids?

## Follow-on questions (to add as we learn more)

- Imbalance exposure: do they self-balance via IDM or accept imbalance settlement?
- Ancillary services: is the turbine certified for any aFRR / mFRR product today,
  or only the hypothetical plan in memory?
- Who is the balance responsible party (BRP) — TEHO itself or an aggregator?
- Fuel (wood chip) procurement cadence — does fuel cost ever constrain the
  production plan in-month?
- Heat contract: is heat delivery obligation firm (must-run on demand) or is
  there flexibility the plant could exploit?
