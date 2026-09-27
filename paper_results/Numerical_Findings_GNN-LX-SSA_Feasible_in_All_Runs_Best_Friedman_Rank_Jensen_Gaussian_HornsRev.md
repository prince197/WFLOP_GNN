# Numerical findings: GNN-LX-SSA is feasible in all runs and ranks first under the Jensen and Gaussian wakes

Paper: *GNN-LX-SSA: A repair-based, graph-surrogate-assisted Laplacian salp swarm algorithm for constrained
wind farm layout optimization* (Solanki, Dwivedi, Garg, Shukla).

This file collects the headline numbers of every campaign run for the paper. Complete tables are in
`Main_Text_Tables_1-20_Capacity_Feasibility_Ranks_Statistics_PowerCurve_Ablation.md` and
`Supplementary_Tables_S1-S32_Detailed_Results_Jensen_and_Gaussian.md`. The per-run data are in `data/`.

Units: power and wake loss in kW. The raw CSV columns `WakeLoss` and `EnergyProduction` of the benchmark
campaigns are in working units (1 wu = 1/15 kW). The Horns Rev CSVs are already in kW.

---

## 1. Campaign design (benchmark farms)

| Item | Value |
|---|---|
| Algorithms | 9: GNN-LX-SSA (proposed), LX-SSA, PF, GWO, BBO, SSA, PSO, GA, DE |
| Farm radii | 500, 750, 1000 m (circular benchmark farms) |
| Configurations | 39 (turbine counts up to 10, 14, 18 at the three radii) |
| Wind data sets | 2 |
| Runs per cell | 30 |
| Runs per data set | 10,530 (9 × 39 × 30) |
| Minimum spacing | 4D = 308 m |
| Wake models | Jensen (benchmark cone formulation) and Gaussian (Bastankhah & Porté-Agel) |
| Power curves | Benchmark linearised ramp, and the tabulated GE 1.5 MW / 77 m curve (NREL turbine-models) |
| Population / iterations | 30 / 100 |

## 2. Attainable capacity of the benchmark farms (Tables 2 and 3)

| Farm radius | Lower bound (construction) | Upper bound (packing) | Published maximum | Largest tested here |
|---|---|---|---|---|
| 500 m | ≥ 13 | 15 | 8 | 10 |
| 750 m | ≥ 26 | 30 | 12 | 14 |
| 1000 m | ≥ 43 | 49 | 15 | 18 |

The published maxima lie below the constructive lower bound by 5, 14 and 28 turbines.

## 3. Jensen wake: feasibility (Table 9)

| Algorithm | Infeasible runs (of 1,170) | % of runs | Cells with 0/30 feasible |
|---|---|---|---|
| DE | 514 | 43.9 | 14 |
| GA | 410 | 35.0 | 10 |
| PSO | 388 | 33.2 | 7 |
| PF | 127 | 10.9 | 0 |
| GWO | 90 | 7.7 | 0 |
| SSA | 78 | 6.7 | 0 |
| BBO | 72 | 6.2 | 1 |
| LX-SSA | 43 | 3.7 | 0 |
| **GNN-LX-SSA** | **0** | **0.0** | **0** |

- 1,722 of 10,530 runs (16.35%) are infeasible in each data set, and they are the same runs in both data sets.
- DE, GA and PSO account for 76% of all violations. 32 of the 351 algorithm-configuration cells have no feasible run.
- The penalty coefficient is 10^20. It exceeds the largest attainable wake loss by 7 to 23 orders of magnitude.

## 4. Jensen wake: ranking and tests (Tables 12 to 15)

| Algorithm | Exact evaluations per run | DS1 avg rank | DS2 avg rank | DS1 Wilcoxon p (Holm) | DS2 Wilcoxon p (Holm) |
|---|---|---|---|---|---|
| **GNN-LX-SSA** | **1,538 – 2,601 (mean 1,968)** | **2.26 (1st)** | **1.92 (1st)** | — | — |
| PF | 3,030 | 2.95 | 3.40 | < 0.001 | < 0.001 |
| BBO | 3,030 | 3.03 | 3.31 | 0.362 | 0.001 |
| LX-SSA | 6,030 | 4.23 | 4.45 | < 0.001 | < 0.001 |
| GWO | 3,030 | 5.04 | 4.53 | < 0.001 | < 0.001 |
| SSA | 3,030 | 5.58 | 5.71 | < 0.001 | < 0.001 |
| PSO | 3,030 | 6.37 | 6.21 | < 0.001 | < 0.001 |
| GA | 3,030 | 7.28 | 7.19 | < 0.001 | < 0.001 |
| DE | 3,030 | 8.27 | 8.29 | < 0.001 | < 0.001 |

- Friedman χ² (tie-corrected) = 193.92 (DS1) and 186.34 (DS2), p < 1e-30. Nemenyi CD = 1.92.
- GNN-LX-SSA is significantly better than 7 of 8 baselines in DS1 and 8 of 8 in DS2. The exception is BBO in DS1 (p = 0.362).
- Matched budget (common cap of 3,030 exact evaluations, best of 30 runs): GNN-LX-SSA ranks 2.09 (DS1) and 1.91 (DS2), first in both.
- GNN-LX-SSA budget B(N) = 1,470 + 64N, recorded averages 1,814 / 1,942 / 2,070 at 500 / 750 / 1000 m. Surrogate inferences per run: 9,000.
- The seven fixed-budget baselines spend 1.54× and LX-SSA 3.06× the mean GNN-LX-SSA budget.

## 5. Ablation: repair map against surrogate (Tables 17, 18 and 20)

Three arms at the largest configuration of each farm size (30 runs each):

| Configuration | LX-SSA feasible | LX-SSA+R feasible | GNN-LX-SSA feasible | LX-SSA+R evals | GNN-LX-SSA evals |
|---|---|---|---|---|---|
| DS1, 500 m, 10 turbines | 6/30 | 30/30 | 30/30 | 6,030 | 2,072 |
| DS1, 750 m, 14 turbines | 23/30 | 30/30 | 30/30 | 6,030 | 2,326 |
| DS1, 1000 m, 18 turbines | 25/30 | 30/30 | 30/30 | 6,030 | 2,583 |
| DS2, 500 m, 10 turbines | 6/30 | 30/30 | 30/30 | 6,030 | 2,071 |
| DS2, 750 m, 14 turbines | 23/30 | 30/30 | 30/30 | 6,030 | 2,326 |
| DS2, 1000 m, 18 turbines | 25/30 | 30/30 | 30/30 | 6,030 | 2,582 |

GNN-LX-SSA against LX-SSA+R over all 39 configurations:

| Wake | Data set | Comparison | Median ratio (GNN / no surrogate) | p |
|---|---|---|---|---|
| Jensen | DS1 | own budgets | 1.069 | < 0.001 |
| Jensen | DS1 | equal cost | 0.988 | 0.85 |
| Jensen | DS2 | own budgets | 1.033 | < 0.001 |
| Jensen | DS2 | equal cost | 0.997 | 0.73 |
| Gaussian | DS1 | own budgets | 1.188 | < 0.001 |
| Gaussian | DS1 | equal cost | 1.050 | < 0.001 |
| Gaussian | DS2 | own budgets | 1.129 | < 0.001 |
| Gaussian | DS2 | equal cost | 1.021 | 0.011 |

- The repair map lifts feasibility from 6–25 of 30 runs to 30 of 30 and supplies most of the wake-loss reduction.
- The surrogate cuts exact evaluations by about two thirds relative to the same repaired search.
- Price in wake loss: 3–7% at own budgets under Jensen, none at equal cost. Under the Gaussian wake, 13–19% at own budgets and 2–5% at equal cost.
- Summary: the repair map makes GNN-LX-SSA feasible and accurate, and the GNN surrogate makes it cheap.

## 6. Gaussian wake (Table 19, Supplementary Tables S20 to S29)

| Algorithm | Feasible runs | DS1 Jensen rank | DS1 Gaussian rank | DS1 p (Holm) | DS2 Jensen rank | DS2 Gaussian rank | DS2 p (Holm) |
|---|---|---|---|---|---|---|---|
| **GNN-LX-SSA** | **1,170 / 1,170** | 2.26 | **2.56** | — | 1.92 | **2.00** | — |
| LX-SSA | 1,117 / 1,170 | 4.23 | 3.26 | < 0.001 | 4.45 | 3.74 | < 0.001 |
| PF | 1,053 / 1,170 | 2.95 | 3.72 | < 0.001 | 3.40 | 3.82 | < 0.001 |
| GWO | 1,088 / 1,170 | 5.04 | 4.08 | < 0.001 | 4.53 | 3.28 | < 0.001 |
| BBO | 1,088 / 1,170 | 3.03 | 4.19 | 0.013 | 3.31 | 4.32 | < 0.001 |
| SSA | 1,091 / 1,170 | 5.58 | 4.79 | < 0.001 | 5.71 | 5.36 | < 0.001 |
| PSO | 774 / 1,170 | 6.37 | 6.46 | < 0.001 | 6.21 | 6.54 | < 0.001 |
| GA | 778 / 1,170 | 7.28 | 7.82 | < 0.001 | 7.19 | 7.90 | < 0.001 |
| DE | 650 / 1,170 | 8.27 | 8.12 | < 0.001 | 8.29 | 8.04 | < 0.001 |

- GNN-LX-SSA is feasible in all 2,340 runs and significantly better than all eight baselines in both data sets after Holm correction.
- 1,721 of 10,530 runs (16.34%) are infeasible per data set (1,722 under Jensen). 27 cells have no feasible run (32 under Jensen).
- Friedman χ² (tie-corrected) = 167.3 (DS1) and 185.9 (DS2), p < 1e-30.
- Matched budget (3,030 evaluations): GNN-LX-SSA ranks 1.79 (DS1) and 1.46 (DS2), first in both.
- GNN-LX-SSA budget: 1,542 – 2,595 exact evaluations per run (mean 1,968).
- Agreement with the Jensen ordering: Kendall τ = 0.83 (DS1) and 0.67 (DS2).
- Smallest penalized objective 1.4 × 10^15 against largest feasible objective 3.3 × 10^4.

## 7. Commercial power curve: GE 1.5 MW, 77 m (Table 16)

The whole Jensen campaign (9 algorithms × 39 configurations × 30 runs × 2 data sets) was re-run with the
tabulated power curve of the GE 1.5 MW turbine with a 77 m rotor (NREL turbine-models library,
DOE_GE_1.5MW_77, DOI 10.11578/dc.20210112.1). This turbine has the benchmark's rotor, rating, cut-in and
cut-out speeds and reaches rated power at 14.5 m/s. Seeds and settings are unchanged; only the power
curve differs. The expected-power integration agrees with adaptive quadrature to within 1.4 × 10^-4.

Expected power of one unwaked turbine:

| Data set | Linear ramp (benchmark) | GE 1.5 MW curve | Linear vs real |
|---|---|---|---|
| DS1 | 936.38 kW | 973.49 kW | −3.8% |
| DS2 | 487.69 kW | 540.61 kW | −9.8% |

(Against the idealised cubic law the linear ramp is +21.0% and +57.1%, so it lies between the two.)

| Algorithm | Feasible runs | DS1 linear rank | DS1 GE rank | DS1 p (Holm) | DS2 linear rank | DS2 GE rank | DS2 p (Holm) |
|---|---|---|---|---|---|---|---|
| **GNN-LX-SSA** | **1,170 / 1,170** | 2.26 | **2.33** | — | 1.92 | **1.92** | — |
| PF | 1,053 / 1,170 | 2.95 | 2.99 | < 0.001 | 3.40 | 3.71 | < 0.001 |
| BBO | 1,087 / 1,170 | 3.03 | 3.18 | 0.249 | 3.31 | 3.08 | 0.003 |
| LX-SSA | 1,117 / 1,170 | 4.23 | 4.22 | < 0.001 | 4.45 | 4.29 | < 0.001 |
| GWO | 1,088 / 1,170 | 5.04 | 4.88 | < 0.001 | 4.53 | 4.50 | < 0.001 |
| SSA | 1,091 / 1,170 | 5.58 | 5.40 | < 0.001 | 5.71 | 5.65 | < 0.001 |
| PSO | 774 / 1,170 | 6.37 | 6.47 | < 0.001 | 6.21 | 6.37 | < 0.001 |
| GA | 778 / 1,170 | 7.28 | 7.29 | < 0.001 | 7.19 | 7.24 | < 0.001 |
| DE | 650 / 1,170 | 8.27 | 8.23 | < 0.001 | 8.29 | 8.23 | < 0.001 |

- The ordering of all nine algorithms is identical under the two curves in both data sets (Kendall τ = 1.00).
- Friedman χ² (tie-corrected) = 186.7 (DS1) and 187.8 (DS2), p < 1e-30.
- Matched budget (3,030 evaluations): GNN-LX-SSA ranks 2.38 (DS1) and 1.83 (DS2), first in both.
- GNN-LX-SSA budget: 1,544 – 2,592 exact evaluations per run (mean 1,968).
- Baselines infeasible in 1,722 runs per data set (same as with the linear ramp); 27 cells with no feasible run.

## 8. Computational cost (Table 8)

| Arm | Mean amortized run time (s) | Mean exact evaluations | Amortized time per exact evaluation (ms) |
|---|---|---|---|
| Seven penalty-only baselines | 0.540 – 0.551 | 3,030 | 0.178 – 0.182 |
| LX-SSA | 0.947 | 6,030 | 0.157 |
| GNN-LX-SSA | 7.450 | 1,968 | 3.785 |

An exact evaluation would have to cost about 1.6 ms more for the surrogate-assisted arm to break even in
elapsed time on this machine. No wall-clock saving is claimed.

## 9. Horns Rev 1 real wind farm (80 turbines, in progress)

Setup: 80 Vestas V80 turbines (D = 80 m), constant C_T = 0.8, PyWake Hornsrev1 12-sector wind rose, TI = 0.1,
convex hull of the as-built layout as boundary, minimum spacing 4D = 320 m. Jensen K = 0.04, Gaussian k* = 0.04205.
Model agreement with PyWake: Jensen 635.583 vs 635.245 GWh, Gaussian 671.431 vs 671.162 GWh (within 0.05%).
Exact-evaluation budget: 3,030 per run for the baselines, 6,030 for LX-SSA, at most B(80) ≈ 6,590 for GNN-LX-SSA.

Real (as-built) layout, reference values:

| Wake model | Expected power (kW) | AEP (GWh/yr) | Wake loss (%) | Min spacing (m) |
|---|---|---|---|---|
| Jensen | 72,555 | 635.6 | 14.56 | 559 (6.99D) |
| Gaussian | 76,647 | 671.4 | 9.74 | 559 (6.99D) |

Baselines (30 runs each, identical under both wake models because the penalty drives the search):

| Algorithm | Feasible runs | Evals | Median turbines outside site | Median pairs closer than 4D | Median min spacing (m) | Smallest penalized objective |
|---|---|---|---|---|---|---|
| BBO | 0/30 | 3,030 | 6.0 | 16.5 | 243.2 | 1.75e24 |
| LX-SSA | 0/30 | 6,030 | 7.0 | 31.5 | 205.9 | 4.96e24 |
| SSA | 0/30 | 3,030 | 9.0 | 34.5 | 190.7 | 6.83e24 |
| PF | 0/30 | 3,030 | 17.0 | 32.0 | 95.1 | 2.26e25 |
| GWO | 0/30 | 3,030 | 2.5 | 89.0 | 85.7 | 4.15e25 |
| DE | 0/30 | 3,030 | 5.5 | 46.0 | 57.5 | 5.16e25 |
| GA | 0/30 | 3,030 | 13.5 | 35.0 | 55.7 | 7.13e25 |
| PSO | 0/30 | 3,030 | 24.0 | 32.0 | 51.9 | 1.76e26 |

All eight penalty-based baselines end infeasible in 240 of 240 runs under each wake model.
GNN-LX-SSA (Jensen and Gaussian) is still running. This section will be updated with its feasibility,
AEP and wake loss against the real layout when it finishes.
