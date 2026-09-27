# Main-text tables 1 to 19: capacity bounds, feasibility, ranks, statistical tests, budget and ablation

All 19 tables of the manuscript, converted from the LaTeX source with their printed numbers and captions. Power and wake-loss values are in kW. Colour highlighting of the PDF is not reproduced. Regenerate with `scripts/tex_tables_to_md.py`.

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

Attainable capacity of the three benchmark farms, bracketed between the constructive lower bound and the packing-density upper bound, compared with the maxima reported in the literature and the largest configurations tested in this campaign.

| Farm radius | Lower bound (Section 4.1) | Upper bound (Section 4.2) | Reported in the literature | Largest tested here | Spare vs. tested |
|---|---|---|---|---|---|
| 500 m | ≥ 13 | 15 | 8 | 10 | +3 |
| 750 m | ≥ 26 | 30 | 12 | 14 | +12 |
| 1000 m | ≥ 43 | 49 | 15 | 18 | +25 |

### Table 3

Attainable capacity N*(r) bracketed at four spacing rules. Each cell gives the constructive lower bound and the finite-packing upper bound of (20). The published maxima are 8, 12 and 15 turbines at 500, 750 and 1000 m.

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

GNWM training configuration (left) and architecture (right). All training takes place inside each run, on the exact evaluations of that run.

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

Amortized wall-clock cost of all nine arms on the same 4-core CPU node (four worker processes, one BLAS thread each), averaged over the 39 configurations of Wind Data Set 1. The Wind Data Set 2 figure of GNN-LX-SSA differs by 2%. The 30 seeds of a configuration run together in one batched call, so the second column is the group time divided by 30 and not a sequential wall time. The fourth column is the second divided by the third, an accounting ratio over the complete optimization call and not a measured latency of the exact objective.

| Algorithm group | Mean amortized run time (s) | Mean exact evaluations | Amortized total run time per exact evaluation (ms) |
|---|---|---|---|
| Seven penalty-only baselines | 0.540 – 0.551 | 3,030 | 0.178 – 0.182 |
| LX-SSA | 0.947 | 6,030 | 0.157 |
| GNN-LX-SSA | 7.450 | 1,968 | 3.785 |

### Table 9

Constraint-violation profile of each algorithm, giving how often it fails, at which farm sizes, and the turbine count at which failure begins. A dash means that the algorithm never reached that state at any tested size. The counts in the two wind data sets are identical (see Section 9.1).

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

Feasible layouts out of 30 independent runs for the farm radius of 500 m. The counts in Wind Data Set 1 and Wind Data Set 2 are identical, and Section 9.1 explains why.

| Algorithm | 8 turbines | 9 turbines | 10 turbines |
|---|---|---|---|
| GNN-LX-SSA | 30 / 30 | 30 / 30 | 30 / 30 |
| PF | 25 / 30 | 20 / 30 | 3 / 30 |
| BBO (this campaign) | 24 / 30 | 6 / 30 | 0 / 30 |
| BBO (replication campaign) | 26 / 30 | 8 / 30 | not run |

### Table 11

Mean rank of the per-configuration mean wake loss over feasible runs. The values are averaged rank positions rather than standings, and the lowest value in each column identifies the best-performing algorithm for that panel.

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

Objective-evaluation budget per run against overall standing for all nine algorithms. Position is the place of the algorithm among the nine, and average rank value is the mean of its per-configuration ranks, which equals 1.00 only if an algorithm is best in every configuration. GNN-LX-SSA uses at most B(N) = 1,470 + 64N evaluations, with recorded averages of 1,814 at 500 m, 1,942 at 750 m and 2,070 at 1000 m. Surrogate inferences involve no wake-model computation, so only the objective-evaluation column is comparable across algorithms.

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

Friedman test over the 39 farm configurations. The hypothesis that all nine algorithms perform equally is rejected decisively in both data sets.

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

Wilcoxon signed-rank tests with GNN-LX-SSA as reference in both wind data sets. Green marks a comparison that GNN-LX-SSA wins significantly, and amber marks one that is not significant.

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

Average Friedman ranks at each algorithm's own budget (full) and at the common cap of 3,030 exact evaluations (matched), with the number of configurations in which a feasible layout was reached by the cap. Both columns rank the best wake loss over the 30 runs of each configuration, the statistic stored at the checkpoints. Lower rank is better. LX-SSA is the only arm whose input value changes between the columns, and the only arm whose rank worsens.

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

Three-arm ablation over 30 runs per cell. LX-SSA penalizes constraint violations. LX-SSA+R applies the repair map of Section 5.5 and evaluates every candidate exactly. GNN-LX-SSA adds the GNWM surrogate, which screens candidates and reduces the exact-evaluation budget to at most B(N) = 1,470 + 64N. Means and minima are over the feasible runs, and bold marks the lowest value in each cell. The GNN-LX-SSA rows are the campaign's own runs, and the other two arms use the same code, seeds and settings.

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

### Table 17

GNN-LX-SSA against the same search without the surrogate (LX-SSA+R) over all 39 configurations of each wind data set. Own budgets compares the per-configuration mean wake loss over 30 runs, with GNN-LX-SSA at 1,968 and LX-SSA+R at 6,030 exact evaluations per run on average. Equal cost compares the median over 30 runs of the best-so-far wake loss, with both arms read at the exact-evaluation count of GNN-LX-SSA in each configuration. Counts give the configurations in which each arm is lower, and ties are two- and three-turbine cases with the same optimum. The ratio is GNN-LX-SSA over LX-SSA+R, as the median over configurations with values of at least 1/15 kW, and p is from the two-sided Wilcoxon signed-rank test over configurations.

| Data set | Comparison | GNN-LX-SSA lower | LX-SSA+R lower | Ties | Median ratio | p |
|---|---|---|---|---|---|---|
| Wind Data Set 1 | Own budgets | 5 | 32 | 2 | 1.069 | < 0.001 |
|  | Equal cost | 18 | 16 | 5 | 0.988 | 0.85 |
| Wind Data Set 2 | Own budgets | 6 | 31 | 2 | 1.033 | < 0.001 |
|  | Equal cost | 19 | 16 | 4 | 0.997 | 0.73 |

### Table 18

Friedman average ranks under the Jensen wake (Section 9) and under the Gaussian wake (this section) for each wind data set, with the Holm-corrected Wilcoxon signed-rank p-value of GNN-LX-SSA against each competitor under the Gaussian wake. The feasible-run counts refer to the Gaussian wake and are identical in the two data sets. A lower rank is better.

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

### Table 19

GNN-LX-SSA against the same search without the surrogate (LX-SSA+R) over all 39 configurations of each wind data set under the Gaussian wake, in the format of Table 17. The Own budgets rows give the per-configuration mean wake loss over 30 runs, with GNN-LX-SSA at 1,968 and LX-SSA+R at 6,030 exact evaluations per run on average. The Equal cost rows give the median over 30 runs of the best-so-far wake loss, with both arms read at the exact-evaluation count of GNN-LX-SSA in each configuration. The ratio is the value of GNN-LX-SSA divided by that of LX-SSA+R, the median over configurations with values of at least 1/15 kW, and p is from the two-sided Wilcoxon signed-rank test over configurations.

| Data set | Comparison | GNN-LX-SSA lower | LX-SSA+R lower | Ties | Median ratio | p |
|---|---|---|---|---|---|---|
| Wind Data Set 1 | Own budgets | 0 | 39 | 0 | 1.188 | < 0.001 |
|  | Equal cost | 10 | 29 | 0 | 1.050 | < 0.001 |
| Wind Data Set 2 | Own budgets | 1 | 38 | 0 | 1.129 | < 0.001 |
|  | Equal cost | 13 | 26 | 0 | 1.021 | 0.011 |

