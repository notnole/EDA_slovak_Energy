# TEHO Bardejov — Open Questions

Questions about how TEHO Bardejov actually operates the plant and participates
in the electricity market. Answers will refine our understanding and sharpen
optimization recommendations.

---

## Section 1 — Already Investigated (from data)

### Q1. Do they only bid day-ahead based on predicted heat load?

**Answer (Jan 2025):** Yes — the committed plan is 100% heat-driven, with zero
price sensitivity. Verified against `kalkulácie TeHo BJ 2025 01.XLSX`:

| Test | Result |
| --- | --- |
| `PLAN == CEIL(MW_EE + Maric)` for all hours | 744 / 744 match |
| Regression R2 (heat + DA price) | 0.896 |
| Regression R2 (heat only) | 0.894 (price adds +0.002) |
| Heat coefficient | 0.247 (t=74) -- recovers the 1/4 heat->EE ratio |
| DA-price coefficient | 0.00074 MW per EUR/MWh (t=3.96 but economically nil) |

A 100 EUR/MWh DA-price swing shifts the plan by 0.07 MW -- rounding noise.

**Committed step distribution (Jan 2025):**
`{3 MW: 30h, 4 MW: 112h, 5 MW: 187h, 6 MW: 410h (55%), 7 MW: 5h}`. Steps 1, 2,
8 were never planned. The 7 MW step appears to be a formula artifact -- 7 MW is
not a normal plant operating point.

**Implication:** with subsidies removed in 2025, a pure heat-driven plan leaves
all price-capture value on the table. This is the ~300 kEUR/yr opportunity
from the bid-sweep analysis.

### Q2. Do they do any conditional bids?

**Status: Unverified.** Suspected answer: no, because Q1 shows price has no
influence on the plan. But the `Porovnanie` sheet legend references "intraday
trades" and "plan changes vs original" -- it is unclear how often these occur
and what triggers them.

---

## Section 2 — Uncertainties to Resolve with TEHO Engineers

These are open questions where the data alone is insufficient. They are ranked
by impact on the strategy recommendation.

---

### COST MODEL

**U1. What is the actual specific chip consumption per MWh of electricity?**

The most consequential unknown. Two models disagree:

- Linear workbook (`kalkulacie`): 1.1 ATT/MWh_EE (dry tonnes), chip at 110 EUR/ATT
  -> gross EE cost ~121 EUR/MWh, net after heat credit ~41-60 EUR/MWh
- Engineering model (`Priprava prevadzky`): ~0.9 ATT/MWh_EE
  -> marginal cost at 5-8 MW ~85 EUR/MWh

This 20% difference shifts the annual bid-strategy P&L by ~250 kEUR/yr
(engineering: +318 kEUR; linear: +62 kEUR at bid-130). Ask for the fuel
meter-read divided by the turbine MWh meter for any representative month.

**U2. Is the chip price (~110 EUR/ATT) fixed by contract or spot-indexed?**

The January sheet uses 110 EUR/ATT. The `Priprava` model uses 51 EUR/t wet
(~88 EUR/ATT dry). If chip price varies seasonally or by supply, the
break-even bid threshold should move accordingly. Ask: is there a long-term
supply contract with a fixed price, or monthly procurement at market rates?

**U3. What is the actual regulated heat tariff received from the district heating network?**

The workbook formula implies ~27.95 EUR/MWh (derived from 110/5/0.787
cross-check). A 5 EUR/MWh error in this tariff shifts the EE break-even by
~3 EUR/MWh. Ask for the current tariff notice (rozhodnutie URSO) or invoice.

**U4. Do ash disposal, own-consumption, and wear costs increase materially at
two-boiler intensive operation vs one-boiler baseload?**

The engineering model applies ~10 EUR/MWh flat for non-fuel variable costs.
Is grate wear, ash volume, or refractory degradation significantly higher at
8 MW (two boilers at max) than at 4 MW (one boiler, relaxed)?

---

### OPERATIONAL / BOILER

**U5. What is the actual warm-start and cold-start time for the second boiler
(from notification to full steam at 78 bar / 520 C)?**

Memory records 4-8 hours for cold start. But:
- What counts as "warm"? (boiler >200C drum temp?)
- What is the warm-start time specifically? (maybe 1-2 hours?)
- Is there a minimum notice requirement for the boiler operator?

This is critical: at bid-115, most high-value clusters last only 3-6 hours
(spike_clusters analysis). If warm-start is 1-2h, all 3h+ clusters are
capturable. If it is 6h, then hot standby is mandatory during summer.

**U6. What is the minimum stable EE output when running two boilers simultaneously?**

At four-MW resolution it looks like the floor is ~4 MW on one boiler and
~5 MW on two. Is there a formal minimum-load specification for the turbine
at each extraction point? This determines whether they can quickly ramp down
between spikes while keeping both boilers warm.

**U7. What is the actual fuel consumption rate during second-boiler hot standby
(drum pressurised but no steam to turbine)?**

We estimated ~3.65 EUR/h (32 kEUR/yr hot standby cost from project context).
Is this measured? Is it based on a specific operating mode (drum at 78 bar, no
firing vs slow fire to maintain pressure)? This is the key input for the
hot-standby break-even calculation in the spike-cluster analysis.

---

### MAINTENANCE / SCHEDULING

**U8. What maintenance window is mandatory for each boiler, can it be moved to
October-November, and what drives the current June-August timing?**

The maintenance window analysis shows moving from June to October saves
~40 kEUR/yr in missed price-spike value. Ask:
- Is the current summer timing driven by statute (pressure-vessel inspection
  requires annual shutdown), insurance, or simply legacy practice?
- Can statutory inspections be scheduled for October?
- Is there any grid-operator or heat-network requirement to perform outages
  in summer (when district heating demand is lowest)?

---

### MARKET / COMMERCIAL

**U9. Who is the Balance Responsible Party for TEHO's electricity -- TEHO
itself or an aggregator -- and what is the actual imbalance settlement
mechanism?**

The trip-risk analysis estimated ~16 kEUR/yr imbalance exposure using a
1.5x DA multiplier. But:
- If an aggregator is the BRP, TEHO may pay a flat fee and face no direct
  imbalance price exposure (changes the trip-risk calculation entirely).
- If TEHO is its own BRP, can they re-nominate on the intraday market after a
  trip to reduce open position?
- What is their historical imbalance cost in EUR, as reported on the OKTE
  settlement statement?

**U10. What is the current DA bid submission process -- price-taking or
price-limited, hourly or block bids?**

The plan is heat-driven, but how is it submitted to OKTE?
- Do they submit hourly quantity bids at a floor price (e.g., 0 EUR/MWh
  take-whatever-clears), or do they set a minimum price?
- Do they use block bids (quantity locked across multiple hours)? Block bids
  would make intraday adjustments harder.
- If they already set a minimum price, what is it and why?
Understanding the current bid format is a prerequisite to recommending any
change.

---

### STRATEGY VALIDATION

**U11. Why did dispatch philosophy change between 2024 (flat ~7.5 MW) and
2025 (heat-driven ~5 MW)?**

The 2024 Jun-Dec data shows average actual_MW = 7.49 MW -- essentially flat at
maximum, closely resembling the bid-at-X strategy we are recommending. In 2025
the plant adopted the `kalkulacie` planning tool and production dropped.

- Was 2024's flat-max operation driven by subsidies that expired in 2025?
- Was it a turbine run-in period (avoiding load cycling on a new machine)?
- Was it a deliberate commercial choice by the operator at the time?
- If subsidies ended, what was the rationale for the heat-only formula rather
  than continuing to run near-max?

This is important: if 2024 is proof-of-concept that the plant CAN run at
8 MW, we need to understand why it stopped.

**U12. Is there a minimum or maximum heat delivery obligation to the district
heating network, by hour or by season?**

Our analysis assumed the dissipator absorbs all surplus heat, so EE dispatch
is unconstrained from above. But:
- Is there a minimum supply temperature or minimum heat flow rate guaranteed
  to the network (e.g., 60 C return temp)?
- In peak winter demand (>25 MW thermal), does the plant HAVE to maintain 8 MW
  EE just to meet heat -- making the price signal irrelevant in those hours?
- Does the heat contract specify a minimum annual MWh delivery, which could
  constrain summer shutdowns?

---

## Priority for Next Meeting

| Priority | Question | Why It Matters |
| --- | --- | --- |
| 1 | U1 -- chip consumption per MWh | +/-250 kEUR/yr depending on answer |
| 2 | U5 -- warm/cold start time | Determines whether hot standby is needed at bid-115 |
| 3 | U11 -- 2024 vs 2025 dispatch change | Validates that 8 MW operation is physically realistic |
| 4 | U9 -- BRP and imbalance mechanism | Changes the trip-risk cost entirely |
| 5 | U8 -- maintenance window flexibility | 40 kEUR/yr if can move to October |
| 6 | U12 -- heat delivery obligations | May constrain EE dispatch in winter or prevent summer shutdown |
| 7 | U2 -- chip price contract | Affects whether bid threshold should be fixed or seasonal |
| 8 | U10 -- current bid format | Prerequisite for implementing any change |
| 9 | U6 -- minimum 2-boiler load | Determines whether partial ramp-down between spikes is feasible |
| 10 | U3 -- heat tariff | Smaller EUR impact but needed for accurate cost model |
