# Entropy-Ranked Stability Index (ERSI)

This directory contains the implementations for the Entropy-Ranked Stability Index (ERSI). ERSI was designed to act as a heuristic tool for signal stability analysis. Standard entropy measures often suffer from high variance or extreme peaks during recording artifacts. ERSI suppresses these unstable windows by weighting the entropy values inversely proportional to their relative rank.

## ERSI Variants & Mathematical Formulations

To measure the stability of a signal, our codebase contains different variations of the ERSI algorithm depending on what you are trying to aggregate or fuse. Below is a breakdown for our collaborators that clarifies the dimensionality and function of each ERSI type.

---

### 1. `ERSI_computation` (Intra-Measure Time Ranking)
This is the foundational block of ERSI. It operates on a **single entropy measure** (e.g., Shannon entropy) across a timeseries of sliding windows.

- **Concept**: Rank the entropy values from lowest to highest over time. Assign a weight of $1 / \text{Rank}$. Multiply the raw entropy by this weight.
- **Dimensionality**: $N \text{ windows} \times 1 \text{ measure} \rightarrow N \text{ windows} \times 1 \text{ stabilized measure}$
- **Equation**: 
  $$\text{ERSI}_{i} = E_{i} \times \frac{1}{R_{i}^{(time)}}$$
- **When to use**: When you want to penalize extreme entropy spikes (artifacts) for a specific measure over out across a single patient's recording.

---

### 2. `ERSI_timeseries` (Additive Horizontal Fusion)
This method fuses **multiple entropy measures** (e.g., Shannon, Tsallis, SVD) computed from the *same* signal. 

- **Concept**: First, it applies `ERSI_computation` independently to every single entropy measure to get their time-stabilized versions. Then, for each time window $i$, it simply **sums** the stabilized ERSI parts together.
- **Dimensionality**: $N \text{ windows} \times M \text{ measures} \rightarrow N \text{ windows} \times 1 \text{ fused measure}$
- **Equation**: 
  $$T_i = \sum_{j=1}^{M} \left( E_{i,j} \times \frac{1}{R_{i,j}^{(time)}} \right) $$
- **When to use**: When you want a single, robust stability line over time that aggregates the independently stabilized evidence of multiple entropy functions.

---

### 3. `ERSI_full` (Multiplicative Dual-Ranking)
This is a more stringent, cross-measure stabilization method. It penalizes a time window not only if its entropy ranked high over time, but also if it ranked high relative to the other concurrent entropy algorithms.

- **Concept**: Computes two ranks:
  1. $R^{(time)}$: Rank of the measure over time (vertical).
  2. $R^{(cross)}$: Rank of the measure compared to the other $M-1$ measures at that exact time window (horizontal).
  It then multiplies the raw entropy by the inverse of *both* ranks, and averages the result across the algorithms.
- **Dimensionality**: $N \text{ windows} \times M \text{ measures} \rightarrow N \text{ windows} \times 1 \text{ dually-stabilized measure}$
- **Equation**: 
  $$F_i = \frac{1}{M} \sum_{j=1}^{M} \left( E_{i,j} \times \frac{1}{R_{i,j}^{(time)}} \times \frac{1}{R_{i,j}^{(cross)}} \right) $$
- **When to use**: When evaluating highly noisy physiological signals where you only want to trust moments where multiple entropy measures *agree* that the signal is stable. It massively suppresses artifacts that fool only one specific entropy algorithm.

---

### 4. `ERSI_by_region_timeseries`
This is conceptually similar to `ERSI_timeseries` but is meant for **Multi-Sensor/Multi-Region** data (e.g., signals coming simultaneously from `chest_ECG`, `arm_ECG`, and `smartwatch_PPG`).

- **Concept**: It groups available entropy feature columns by a substring (the region). It calculates the ranks over time inside that specific group block, computes weights, then sums them up to return a single stability timeseries per physical region.
- **When to use**: When your dataframe consists of entropy measures derived from multiple spatial locations on the body and you want a continuous ERSI index for each region independently.
