# 1. The Baseline Predictor

## Overview
The Baseline Predictor is a deterministic, linear model based on the physical definition of energy. It treats the System Imbalance (MWh) as the time-integral of the Regulation Electricity (MW) over the 15-minute settlement period.

As the settlement period progresses, this model transitions from a heuristic estimation to a precise measurement.

## Mathematical Formulation
The prediction $\hat{y}$ is a weighted average of the available observations within the current Quarter-Hour (QH), scaled to represent energy (MWh).

$$\text{Prediction (MWh)} = 0.25 \times \sum (w_i \times \text{Power}_i)$$

Where $0.25$ converts MW (power) to MWh (energy) for a 15-minute block.

## Formulas by Lead Time
The weights $w_i$ change based on how much of the quarter-hour has passed.

| Lead Time | Time in QH | Formula | Logic |
| :--- | :--- | :--- | :--- |
| **15 min** | :00 | `0.25 × curent` | Heuristic (Last known value) |
| **12 min** | :03 | `0.25 × curent` | Heuristic (First observation) |
| **9 min** | :06 | `0.25 × (0.8x curent + 0.2×lag1)` | Early Weighted Average |
| **6 min** | :09 | `0.25 × (0.6×curent + 0.2×lag1 + 0.2×lag2)` | Mid-Weighted Average |
| **3 min** | :12 | `0.25 × (0.4×curent + 0.2×lag1 + 0.2×lag2 + 0.2×lag3)` | **Physical Integration** |


## Performance Profile
* **Strengths:** Unbeatable accuracy at **3 minutes** lead time (MAE ~1.0 - 1.4 MWh). It effectively "measures" the imbalance rather than predicting it.
* **Weaknesses:** High error at **15-9 minutes** lead time (MAE > 4.0 MWh). It cannot anticipate reversals or trends, leading to poor performance at the start of the interval.