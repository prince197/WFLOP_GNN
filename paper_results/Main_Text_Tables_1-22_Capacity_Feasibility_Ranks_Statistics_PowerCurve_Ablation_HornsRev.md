# Main-text tables 1 to 22: capacity bounds, feasibility, ranks, statistical tests, power curve, ablation and Horns Rev 1

All 22 tables of the manuscript, converted from the LaTeX source with their printed numbers and captions. Power and wake-loss values are in kW, energy in GWh per year. Colour highlighting of the PDF is not reproduced. Regenerate with `scripts/tex_tables_to_md.py`.

### Table 1

Nomenclature.

| Symbol | Description | Symbol | Description |
|---|---|---|---|
| ρ, σ | Cartesian coordinates of a turbine | J(x) | Penalized objective |
| r | Wind farm boundary radius | Π | Repair (projection) map onto the feasible set |
| R | Rotor radius of a turbine | Λ(t) | Triangular lattice with translation t |
| d_min | Minimum inter-turbine distance, d_min = 8R | N_p | Population size |
| θ | Wind direction | T | Number of generations |
| ζ(θ), ψ(θ) | Weibull shape and scale parameters | H | Food source (best solution) in LX-SSA |
| ω_l-1 | Blowing probability, (l-1)-th direction bin | r_1, r_2, r_3 | LX-SSA control and random parameters |
| s | Wind speed | γ(z,φ,χ) | Laplace-distributed random variate |
| s_ci, s_r, s_co | Cut-in, rated and cut-out wind speeds | φ, χ | Laplace location and scale parameters |
| P_rated | Rated power output of a turbine | G = (V,E) | Directed wake-interaction graph |
| λ', η | Slope and intercept of the linear power segment | w_ij | Learnable wake-interaction coefficient |
| C_T | Turbine thrust coefficient | h_i^(k) | Learned wake state of turbine i at step k |
| K | Wake spreading constant | Γ_θ | Trained GNWM surrogate |
| α | Wake half-cone expansion angle | g_i | Learned guidance vector for turbine i |
| β_ij | Wake-cone angle between turbines i, j | τ | Guidance coefficient (annealed) |
| d_ij | Downstream distance between turbines i, j | μ | Fraction of layouts exactly evaluated |
| δs_ij, δs_i | Pairwise and aggregate velocity deficit | F | GNWM embedding width |
| E(P_i) | Expected power output of turbine i | L_g | Number of message-passing layers |
| N_θ, N_s | Direction and speed bin counts | B | Exact evaluation budget per run |
| N | Number of turbines; d = 2N dimensionality | C_of | Exact objective evaluation cost |
| F_N(r) | Feasible set for N turbines at radius r | P_ideal | Ideal farm power without wake losses |
| N*(r) | Attainable capacity of the site | λ_g | Guidance-head loss weight |
| v_ij, N_v | Spacing violation; number of violated constraints | F_f | Friedman test statistic |
| κ | Penalty coefficient | D_avg, S | Average node degree and graph sparsity |

### Table 2

Bounds on the attainable capacity of the three benchmark farms, compared with the published maxima and the largest configurations tested here.

| Farm radius | Lower bound (Section 4.1) | Upper bound (Section 4.2) | Reported in the literature | Largest tested here | Spare vs. tested |
|---|---|---|---|---|---|
| 500 m | ≥ 13 | 15 | 8 | 10 | +3 |
| 750 m | ≥ 26 | 30 | 12 | 14 | +12 |
| 1000 m | ≥ 43 | 49 | 15 | 18 | +25 |

### Table 3

Attainable capacity N*(r) bracketed at four spacing rules, with the constructive lower bound and the upper bound of (20) in each cell. The published maxima are 8, 12 and 15 turbines at 500, 750 and 1000 m.

| Spacing | d_min | 500 m | 750 m | 1000 m | Published maxima exceeded? |
|---|---|---|---|---|---|
| 3D | 231 m | 21 – 24 | 43 – 49 | 74 – 82 | yes, at all three radii |
| 4D (used here) | 308 m | 13 – 15 | 26 – 30 | 43 – 49 | yes, at all three radii |
| 5D | 385 m | 8 – 11 | 19 – 20 | 29 – 33 | yes at 750 m and 1000 m |
| 6D | 462 m | 7 – 8 | 13 – 15 | 21 – 24 | yes at 750 m and 1000 m |

### Table 4

Wind Data Sets 1 and 2 with direction intervals, Weibull shape (zeta) and scale (psi) parameters, and blowing probabilities.

| Int. | Start (deg) | End (deg) | DS1 zeta | DS1 psi | DS1 Prob. | DS2 zeta | DS2 psi | DS2 Prob. |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 15 | 2 | 13 | 0.000 | 2 | 7.0 | 0.0002 |
| 2 | 15 | 30 | 2 | 13 | 0.010 | 2 | 5.0 | 0.0080 |
| 3 | 30 | 45 | 2 | 13 | 0.010 | 2 | 5.0 | 0.0227 |
| 4 | 45 | 60 | 2 | 13 | 0.010 | 2 | 5.0 | 0.0242 |
| 5 | 60 | 75 | 2 | 13 | 0.010 | 2 | 5.0 | 0.0225 |
| 6 | 75 | 90 | 2 | 13 | 0.200 | 2 | 4.0 | 0.0339 |
| 7 | 90 | 105 | 2 | 13 | 0.600 | 2 | 5.0 | 0.0423 |
| 8 | 105 | 120 | 2 | 13 | 0.010 | 2 | 6.0 | 0.0290 |
| 9 | 120 | 135 | 2 | 13 | 0.010 | 2 | 7.0 | 0.0617 |
| 10 | 135 | 150 | 2 | 13 | 0.010 | 2 | 7.0 | 0.0813 |
| 11 | 150 | 165 | 2 | 13 | 0.010 | 2 | 8.0 | 0.0994 |
| 12 | 165 | 180 | 2 | 13 | 0.010 | 2 | 9.5 | 0.1394 |
| 13 | 180 | 195 | 2 | 13 | 0.010 | 2 | 10.0 | 0.1839 |
| 14 | 195 | 210 | 2 | 13 | 0.010 | 2 | 8.5 | 0.1115 |
| 15 | 210 | 225 | 2 | 13 | 0.010 | 2 | 8.5 | 0.0765 |
| 16 | 225 | 240 | 2 | 13 | 0.010 | 2 | 6.5 | 0.0080 |
| 17 | 240 | 255 | 2 | 13 | 0.010 | 2 | 4.6 | 0.0051 |
| 18 | 255 | 270 | 2 | 13 | 0.010 | 2 | 2.6 | 0.0019 |
| 19 | 270 | 285 | 2 | 13 | 0.010 | 2 | 2.8 | 0.0012 |
| 20 | 285 | 300 | 2 | 13 | 0.010 | 2 | 5.0 | 0.0010 |
| 21 | 300 | 315 | 2 | 13 | 0.010 | 2 | 6.4 | 0.0017 |
| 22 | 315 | 330 | 2 | 13 | 0.010 | 2 | 5.2 | 0.0031 |
| 23 | 330 | 345 | 2 | 13 | 0.010 | 2 | 4.5 | 0.0097 |
| 24 | 345 | 360 | 2 | 13 | 0.000 | 2 | 3.9 | 0.0317 |

### Table 5

Turbine, wake and discretization parameters, common to every algorithm and both wind data sets.

| Parameter | Symbol | Value |
|---|---|---|
| Rotor radius | R | 38.5 m |
| Minimum inter-turbine spacing | d_min = 8R | 308 m |
| Cut-in wind speed | s_cut-in | 3.5 m/s |
| Rated wind speed | s_rated | 14 m/s |
| Rated power output | P_rated | 1500 kW |
| Linear power model parameters | λ^′, η | 140.86, -500 |
| Thrust coefficient | C_T | 0.8 |
| Wake spreading constant | K | 0.075 |
| Wind-speed intervals | N_s | 21 (width 0.5 m/s, from s_cut-in = 3.5 to s_rated = 14 m/s) |
| Wind-direction intervals | N_θ | 24 (width 15^°, covering 360^°) |
| Farm radii | r | 500 m, 750 m, 1000 m |
| Turbine counts | N | 2–10 (500 m), 2–14 (750 m), 2–18 (1000 m) — 39 configurations |

### Table 6

Common experimental settings for all nine algorithms.

| Setting | Value |
|---|---|
| Population size | N_p = 30, fixed for every algorithm and every configuration |
| Maximum generations / iterations | 100 |
| Total independent runs | 30 per algorithm per configuration |
| Decision-variable dimension | d = 2N (Cartesian coordinates of the N turbines) |
| Stopping rule | Fixed iteration count; identical for all algorithms |
| Constraint handling, baselines | Penalty transformation of Section 3.8 with box clipping |
| Constraint handling, GNN-LX-SSA | Repair map Π (boundary projection and spacing relaxation, Section 5.5), with the penalty of Section 3.8 on any residual violation |

### Table 7

GNWM training configuration (left) and architecture (right).

| Training (per run) | Value | Architecture | Type | Width |
|---|---|---|---|---|
| Initial training set | 30 + 40 layouts | Node encoder | Linear + ReLU | 64 |
| Direction labels | up to 16 (4N evals each) | Edge encoder | Linear + ReLU | 64 |
| Warm-up epochs | 25 | MP layer 1 | MLP φ_e / φ_n + ReLU | 64 |
| Fine-tuning | 2 epochs every 5 iterations | MP layer 2 | MLP φ_e / φ_n + ReLU | 64 |
| Replay buffer | all exact evaluations of the run | MP layer 3 | MLP φ_e / φ_n + ReLU | 64 |
| Optimizer / learning rate | Adam / 0.001 | Graph readout | Sum pooling | 64 |
| Batch size | 16 | Power head ψ_P | 2-layer MLP | 1 per node |
| Guidance weight λ_g | 0.5 | Guidance head ψ_g | 2-layer MLP | 2 per node |
| Loss | Power + guidance (Section 5.3) |  |  |  |

### Table 8

Amortized wall-clock cost of all nine arms on one 4-core CPU node, averaged over the 39 configurations of Wind Data Set 1. The fourth column is an accounting ratio and not a measured latency of the exact objective.

| Algorithm group | Mean amortized run time (s) | Mean exact evaluations | Amortized total run time per exact evaluation (ms) |
|---|---|---|---|
| Seven penalty-only baselines | 0.540 – 0.551 | 3,030 | 0.178 – 0.182 |
| LX-SSA | 0.947 | 6,030 | 0.157 |
| GNN-LX-SSA | 7.450 | 1,968 | 3.785 |

### Table 9

Constraint-violation profile of each algorithm, with failure rates by farm size and the turbine counts at which failure begins. A dash marks a state never reached. The counts are identical in both wind data sets (Section 9.1).

| Algorithm | Infeasible runs (of 1,170) | % of runs | 500 m % | 750 m % | 1000 m % | Cells with 0/30 feasible | First count with any failure 500/750/1000 m | First count where all 30 fail 500/750/1000 m |
|---|---|---|---|---|---|---|---|---|
| DE | 514 | 43.9 | 46.3 | 43.1 | 43.3 | 14 | 6 / 9 / 10 | 7 / 11 / 13 |
| GA | 410 | 35.0 | 40.4 | 34.6 | 32.5 | 10 | 7 / 10 / 12 | 8 / 11 / 15 |
| PSO | 388 | 33.2 | 31.9 | 33.1 | 33.9 | 7 | 7 / 9 / 11 | 9 / 13 / 16 |
| PF | 127 | 10.9 | 15.9 | 10.3 | 8.6 | — | 7 / 10 / 13 | — / — / — |
| GWO | 90 | 7.7 | 20.7 | 6.2 | 2.0 | — | 7 / 11 / 14 | — / — / — |
| SSA | 78 | 6.7 | 13.7 | 6.7 | 2.9 | — | 8 / 9 / 16 | — / — / — |
| BBO | 72 | 6.2 | 22.2 | 2.8 | 0.2 | 1 | 8 / 13 / 18 | 10 / — / — |
| LX-SSA | 43 | 3.7 | 11.1 | 1.8 | 1.2 | — | 8 / 13 / 16 | — / — / — |
| GNN-LX-SSA | 0 | 0.0 | 0.0 | 0.0 | 0.0 | — | — / — / — | — / — / — |

### Table 10

Feasible layouts out of 30 independent runs at the 500 m farm radius. The counts are identical in the two wind data sets (Section 9.1).

| Algorithm | 8 turbines | 9 turbines | 10 turbines |
|---|---|---|---|
| GNN-LX-SSA | 30 / 30 | 30 / 30 | 30 / 30 |
| PF | 25 / 30 | 20 / 30 | 3 / 30 |
| BBO (this campaign) | 24 / 30 | 6 / 30 | 0 / 30 |
| BBO (replication campaign) | 26 / 30 | 8 / 30 | not run |

### Table 11

Mean rank of the per-configuration mean wake loss over feasible runs. The lowest value in each column marks the best algorithm for that panel.

| Algorithm | DS1 500 m | DS1 750 m | DS1 1000 m | DS2 500 m | DS2 750 m | DS2 1000 m |
|---|---|---|---|---|---|---|
| GNN-LX-SSA | 2.00 | 2.08 | 2.53 | 2.44 | 1.46 | 2.00 |
| PF | 2.50 | 3.08 | 3.09 | 2.94 | 3.62 | 3.47 |
| BBO | 4.89 | 2.62 | 2.35 | 5.33 | 3.00 | 2.47 |
| LX-SSA | 4.50 | 4.23 | 4.09 | 4.17 | 4.23 | 4.76 |
| GWO | 4.72 | 5.00 | 5.24 | 4.39 | 4.54 | 4.59 |
| SSA | 5.50 | 6.15 | 5.18 | 5.61 | 6.00 | 5.53 |
| PSO | 5.67 | 6.31 | 6.79 | 5.22 | 6.46 | 6.53 |
| GA | 7.50 | 7.12 | 7.29 | 7.17 | 7.35 | 7.09 |
| DE | 7.72 | 8.42 | 8.44 | 7.72 | 8.35 | 8.56 |

### Table 12

Objective-evaluation budget per run against overall standing. GNN-LX-SSA uses at most B(N) = 1,470 + 64N evaluations. Surrogate inferences involve no wake-model computation, so only the objective-evaluation column is comparable across algorithms.

| Algorithm | Objective evaluations per run | Surrogate inferences | Multiple of GNN-LX-SSA | DS1 position | DS1 avg rank value | DS2 position | DS2 avg rank value |
|---|---|---|---|---|---|---|---|
| GNN-LX-SSA | 1,538 – 2,601 (mean 1,968) | 9,000 | 1.00× | 1st | 2.26 | 1st | 1.92 |
| BBO | 3,030 | — | 1.54× | 3rd | 3.03 | 2nd | 3.31 |
| GA | 3,030 | — | 1.54× | 8th | 7.28 | 8th | 7.19 |
| DE | 3,030 | — | 1.54× | 9th | 8.27 | 9th | 8.29 |
| PF | 3,030 | — | 1.54× | 2nd | 2.95 | 3rd | 3.40 |
| PSO | 3,030 | — | 1.54× | 7th | 6.37 | 7th | 6.21 |
| SSA | 3,030 | — | 1.54× | 6th | 5.58 | 6th | 5.71 |
| GWO | 3,030 | — | 1.54× | 5th | 5.04 | 5th | 4.53 |
| LX-SSA | 6,030 | — | 3.06× | 4th | 4.23 | 4th | 4.45 |

### Table 13

Friedman test over the 39 farm configurations. Equal performance of all nine algorithms is rejected in both data sets.

| Quantity | Wind Data Set 1 | Wind Data Set 2 |
|---|---|---|
| Blocks (configurations), n | 39 | 39 |
| Algorithms, k | 9 | 9 |
| Friedman chi-square, uncorrected | 178.55 | 171.81 |
| Iman-Davenport F, uncorrected | 50.84 | 46.57 |
| Friedman chi-square, tie-corrected | 193.92 | 186.34 |
| Iman-Davenport F, tie-corrected | 62.41 | 56.35 |
| Nemenyi critical difference | 1.92 | 1.92 |
| p-value | < 1e-30 | < 1e-30 |
| Null hypothesis (equal ranks) | rejected | rejected |
| Per-configuration Kruskal–Wallis rejected (Holm, 5%) | 37 of 37 | 37 of 37 |

### Table 14

Wilcoxon signed-rank tests with GNN-LX-SSA as reference. Green marks a significant win of GNN-LX-SSA and amber a non-significant comparison.

| Compared | DS1 R+ | DS1 R- | DS1 p (Holm) | DS1 r | DS2 R+ | DS2 R- | DS2 p (Holm) | DS2 r |
|---|---|---|---|---|---|---|---|---|
| BBO | 413 | 290 | 0.362 | +0.17 | 558 | 145 | 0.001 | +0.59 |
| DE | 700 | 3 | < 0.001 | +0.99 | 699 | 4 | < 0.001 | +0.99 |
| GA | 697 | 6 | < 0.001 | +0.98 | 693 | 10 | < 0.001 | +0.97 |
| GWO | 681 | 22 | < 0.001 | +0.94 | 701 | 2 | < 0.001 | +0.99 |
| PF | 652 | 51 | < 0.001 | +0.85 | 680 | 23 | < 0.001 | +0.93 |
| PSO | 695 | 8 | < 0.001 | +0.98 | 694 | 9 | < 0.001 | +0.97 |
| SSA | 700 | 3 | < 0.001 | +0.99 | 700 | 3 | < 0.001 | +0.99 |
| LX-SSA | 696 | 7 | < 0.001 | +0.98 | 696 | 7 | < 0.001 | +0.98 |

### Table 15

Average Friedman ranks at each algorithm's own budget (full) and at the common cap of 3,030 exact evaluations (matched), with the configurations feasible by the cap. Both columns rank the best wake loss over the 30 runs. Lower rank is better.

| Arm | Evals | DS1 full | DS1 matched | DS1 feas. | DS2 full | DS2 matched | DS2 feas. |
|---|---|---|---|---|---|---|---|
| GNN-LX-SSA | 1,538-2,601 | 2.13 | 2.09 | 39/39 | 1.94 | 1.91 | 39/39 |
| BBO | 3,030 | 3.27 | 3.04 | 38/39 | 3.14 | 3.01 | 38/39 |
| GWO | 3,030 | 3.62 | 3.42 | 39/39 | 4.12 | 3.78 | 39/39 |
| PF | 3,030 | 4.44 | 4.04 | 39/39 | 4.40 | 3.96 | 39/39 |
| SSA | 3,030 | 4.65 | 4.32 | 39/39 | 5.00 | 4.54 | 39/39 |
| LX-SSA | 3,030 | 4.33 | 5.88 | 35/39 | 4.28 | 6.18 | 35/39 |
| PSO | 3,030 | 6.58 | 6.29 | 32/39 | 6.28 | 5.95 | 32/39 |
| EA (GA) | 3,030 | 7.60 | 7.53 | 29/39 | 7.49 | 7.41 | 29/39 |
| DE | 3,030 | 8.38 | 8.38 | 25/39 | 8.36 | 8.26 | 25/39 |

### Table 16

Friedman average ranks of the Jensen campaign under the linearised ramp and the GE 1.5 MW power curve, with the Holm-corrected Wilcoxon p-value of GNN-LX-SSA against each competitor under the commercial curve. Feasible-run counts refer to the commercial curve and are identical in both data sets.

|  |  | Wind Data Set 1 | Wind Data Set 2 |  |  |  |  |
|---|---|---|---|---|---|---|---|
| Algorithm | Feasible runs | Linear rank | GE 1.5 MW rank | p (Holm) | Linear rank | GE 1.5 MW rank | p (Holm) |
| GNN-LX-SSA | 1,170 / 1,170 | 2.26 | 2.33 | — | 1.92 | 1.92 | — |
| PF | 1,053 / 1,170 | 2.95 | 2.99 | < 0.001 | 3.40 | 3.71 | < 0.001 |
| BBO | 1,087 / 1,170 | 3.03 | 3.18 | 0.249 | 3.31 | 3.08 | 0.003 |
| LX-SSA | 1,117 / 1,170 | 4.23 | 4.22 | < 0.001 | 4.45 | 4.29 | < 0.001 |
| GWO | 1,088 / 1,170 | 5.04 | 4.88 | < 0.001 | 4.53 | 4.50 | < 0.001 |
| SSA | 1,091 / 1,170 | 5.58 | 5.40 | < 0.001 | 5.71 | 5.65 | < 0.001 |
| PSO | 774 / 1,170 | 6.37 | 6.47 | < 0.001 | 6.21 | 6.37 | < 0.001 |
| GA | 778 / 1,170 | 7.28 | 7.29 | < 0.001 | 7.19 | 7.24 | < 0.001 |
| DE | 650 / 1,170 | 8.27 | 8.23 | < 0.001 | 8.29 | 8.23 | < 0.001 |

### Table 17

Three-arm ablation over 30 runs per cell. LX-SSA penalizes constraint violations. LX-SSA+R applies the repair map of Section 5.5 and evaluates every candidate exactly. GNN-LX-SSA adds the GNWM surrogate, which reduces the exact-evaluation budget to at most B(N) = 1,470 + 64N. Bold marks the lowest value in each cell.

| Configuration | Arm | Feasible runs | Exact evaluations | Mean wake loss (kW) | Best wake loss (kW) |
|---|---|---|---|---|---|
| DS1, 500 m, 10 turbines | LX-SSA | 6 / 30 | 6,030 | 562.8 | 367.6 |
|  | LX-SSA+R | 30 / 30 | 6,030 | 179.8 | 134.7 |
|  | GNN-LX-SSA | 30 / 30 | 2,072 | 175.2 | 152.2 |
| DS1, 750 m, 14 turbines | LX-SSA | 23 / 30 | 6,030 | 716.0 | 480.3 |
|  | LX-SSA+R | 30 / 30 | 6,030 | 338.5 | 253.1 |
|  | GNN-LX-SSA | 30 / 30 | 2,326 | 362.0 | 237.5 |
| DS1, 1000 m, 18 turbines | LX-SSA | 25 / 30 | 6,030 | 887.8 | 532.8 |
|  | LX-SSA+R | 30 / 30 | 6,030 | 415.4 | 336.2 |
|  | GNN-LX-SSA | 30 / 30 | 2,583 | 463.2 | 295.2 |
| DS2, 500 m, 10 turbines | LX-SSA | 6 / 30 | 6,030 | 494.3 | 474.0 |
|  | LX-SSA+R | 30 / 30 | 6,030 | 343.0 | 307.7 |
|  | GNN-LX-SSA | 30 / 30 | 2,071 | 327.5 | 315.0 |
| DS2, 750 m, 14 turbines | LX-SSA | 23 / 30 | 6,030 | 680.8 | 568.5 |
|  | LX-SSA+R | 30 / 30 | 6,030 | 461.8 | 403.2 |
|  | GNN-LX-SSA | 30 / 30 | 2,326 | 478.5 | 429.9 |
| DS2, 1000 m, 18 turbines | LX-SSA | 25 / 30 | 6,030 | 783.1 | 603.3 |
|  | LX-SSA+R | 30 / 30 | 6,030 | 551.7 | 504.0 |
|  | GNN-LX-SSA | 30 / 30 | 2,582 | 564.1 | 510.2 |

### Table 18

GNN-LX-SSA against the same search without the surrogate (LX-SSA+R) over all 39 configurations of each wind data set. Own budgets compares the per-configuration mean wake loss over 30 runs, at 1,968 and 6,030 exact evaluations per run on average. Equal cost compares the median best-so-far wake loss over 30 runs at the exact-evaluation count of GNN-LX-SSA. Counts give the configurations in which each arm is lower. The ratio is the median of GNN-LX-SSA over LX-SSA+R, and p is from the two-sided Wilcoxon signed-rank test.

| Data set | Comparison | GNN-LX-SSA lower | LX-SSA+R lower | Ties | Median ratio | p |
|---|---|---|---|---|---|---|
| Wind Data Set 1 | Own budgets | 5 | 32 | 2 | 1.069 | < 0.001 |
|  | Equal cost | 18 | 16 | 5 | 0.988 | 0.85 |
| Wind Data Set 2 | Own budgets | 6 | 31 | 2 | 1.033 | < 0.001 |
|  | Equal cost | 19 | 16 | 4 | 0.997 | 0.73 |

### Table 19

Friedman average ranks under the Jensen (Section 9) and Gaussian wakes per wind data set, and Holm-corrected Wilcoxon signed-rank p-values of GNN-LX-SSA against each competitor under the Gaussian wake. Feasible-run counts refer to the Gaussian wake and are identical in both data sets. A lower rank is better.

|  |  | Wind Data Set 1 | Wind Data Set 2 |  |  |  |  |
|---|---|---|---|---|---|---|---|
| Algorithm | Feasible runs | Jensen rank | Gaussian rank | p (Holm) | Jensen rank | Gaussian rank | p (Holm) |
| GNN-LX-SSA | 1,170 / 1,170 | 2.26 | 2.56 | — | 1.92 | 2.00 | — |
| LX-SSA | 1,117 / 1,170 | 4.23 | 3.26 | < 0.001 | 4.45 | 3.74 | < 0.001 |
| PF | 1,053 / 1,170 | 2.95 | 3.72 | < 0.001 | 3.40 | 3.82 | < 0.001 |
| GWO | 1,088 / 1,170 | 5.04 | 4.08 | < 0.001 | 4.53 | 3.28 | < 0.001 |
| BBO | 1,088 / 1,170 | 3.03 | 4.19 | 0.013 | 3.31 | 4.32 | < 0.001 |
| SSA | 1,091 / 1,170 | 5.58 | 4.79 | < 0.001 | 5.71 | 5.36 | < 0.001 |
| PSO | 774 / 1,170 | 6.37 | 6.46 | < 0.001 | 6.21 | 6.54 | < 0.001 |
| GA | 778 / 1,170 | 7.28 | 7.82 | < 0.001 | 7.19 | 7.90 | < 0.001 |
| DE | 650 / 1,170 | 8.27 | 8.12 | < 0.001 | 8.29 | 8.04 | < 0.001 |

### Table 20

GNN-LX-SSA against the same search without the surrogate (LX-SSA+R) over 39 configurations per wind data set under the Gaussian wake, in the format of Table 18. Own budgets compares mean wake loss over 30 runs at 1,968 and 6,030 exact evaluations per run on average, and Equal cost compares median best-so-far wake loss at the evaluation count of GNN-LX-SSA. The ratio is GNN-LX-SSA over LX-SSA+R (median over configurations with values of at least 1/15 kW), and p is from the two-sided Wilcoxon signed-rank test.

| Data set | Comparison | GNN-LX-SSA lower | LX-SSA+R lower | Ties | Median ratio | p |
|---|---|---|---|---|---|---|
| Wind Data Set 1 | Own budgets | 0 | 39 | 0 | 1.188 | < 0.001 |
|  | Equal cost | 10 | 29 | 0 | 1.050 | < 0.001 |
| Wind Data Set 2 | Own budgets | 1 | 38 | 0 | 1.129 | < 0.001 |
|  | Equal cost | 13 | 26 | 0 | 1.021 | 0.011 |

### Table 21

Horns Rev 1 (80 turbines, 4D spacing, 36 wind directions). The columns give the exact evaluations per run, the feasible runs out of 30 under the Jensen (J) and the Gaussian (G) wake, and the medians over 30 runs of the turbines outside the site, the pairs closer than 4D and the minimum spacing (J / G where two values are given).

| Algorithm | Evaluations | Feasible (J / G) | Outside | Pairs < 4D | Spacing (m) |
|---|---|---|---|---|---|
| GNN-LX-SSA, random start | 6,550 | 30 / 30 | 0 | 0 | 320 |
| GNN-LX-SSA, as-built start | 6,550 | 30 / 30 | 0 | 0 | 549 / 380 |
| BBO | 3,030 | 0 / 0 | 6.0 | 16.5 | 243 |
| LX-SSA | 6,030 | 0 / 0 | 7.0 | 31.5 | 206 |
| SSA | 3,030 | 0 / 0 | 9.0 | 34.5 | 191 |
| PF | 3,030 | 0 / 0 | 17.0 | 32.0 | 95 |
| GWO | 3,030 | 0 / 0 | 2.5 | 89.0 | 86 |
| DE | 3,030 | 0 / 0 | 5.5 | 46.0 | 58 |
| GA | 3,030 | 0 / 0 | 13.5 | 35.0 | 56 |
| PSO | 3,030 | 0 / 0 | 24.0 | 32.0 | 52 |

### Table 22

Annual energy production (GWh per year) at Horns Rev 1, re-evaluated in PyWake at the 36 directions of the objective and at 1° directions. For GNN-LX-SSA the median of 30 runs is given with the best run in parentheses.

|  | Jensen wake | Gaussian wake |  |  |
|---|---|---|---|---|
| Layout | 36 directions | 1° directions | 36 directions | 1° directions |
| As-built | 654.96 | 647.96 | 689.46 | 692.11 |
| GNN-LX-SSA, random start | 637.76 (643.76) | 634.83 (640.97) | 684.78 (686.63) | 684.70 (686.44) |
| GNN-LX-SSA, as-built start | 655.01 (655.26) | 647.83 (648.26) | 691.42 (691.84) | 691.52 (691.87) |
