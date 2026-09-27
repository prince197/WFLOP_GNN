# Paper results

Numerical results of every campaign behind the paper *GNN-LX-SSA: A repair-based, graph-surrogate-assisted
Laplacian salp swarm algorithm for constrained wind farm layout optimization*.

| File or folder | Contents |
|---|---|
| `Numerical_Findings_GNN-LX-SSA_Feasible_in_All_Runs_Best_Friedman_Rank_Jensen_Gaussian_HornsRev.md` | Headline numbers of all campaigns in one place (start here) |
| `Main_Text_Tables_1-20_Capacity_Feasibility_Ranks_Statistics_PowerCurve_Ablation.md` | All 20 tables of the manuscript |
| `Supplementary_Tables_S1-S32_Detailed_Results_Jensen_and_Gaussian.md` | All 32 tables of the Supplementary Material |
| `data/jensen_benchmark/` | Per-run results, efficiency traces and ablation runs under the Jensen wake |
| `data/gaussian_benchmark/` | The same under the Gaussian wake |
| `data/commercial_power_curve_GE1.5MW/` | The Jensen campaign re-run with the GE 1.5 MW commercial power curve |
| `data/hornsrev1_real_farm/` | Horns Rev 1 (80 turbines) per-run results, one file per wake model and algorithm |
| `analysis_outputs/` | JSON outputs of the analysis scripts (ranks, Friedman, Wilcoxon-Holm, matched budget, ablation, Horns Rev) |
| `scripts/` | Analysis scripts, the LaTeX table converter and `file_name_map.tsv` |

File names state the result they hold. The benchmark data files carry values in working units (1 wu = 1/15 kW),
as written by the campaign code. The tables and the findings file are in kW. The Horns Rev files are in kW.

## Reproducing the analysis

The scripts read the files under their original campaign names. Recreate those names as links, then run the
scripts from `scripts/`:

```
cd paper_results/scripts
./restore_original_names.sh
python reanalyze.py new        # Jensen tables  -> reanalysis_new.json
python reanalyze.py gauss      # Gaussian tables -> reanalysis_gauss.json
python gauss_numbers.py        # extra Gaussian numbers
python gaussabl_analysis.py    # Gaussian ablation
python gauss_analysis.py       # Jensen vs Gaussian ranks and Kendall tau
python reanalyze.py ge15       # commercial power curve -> reanalysis_ge15.json
python pc_compare.py           # Table 16: linear vs GE 1.5 MW ranks
WFLOP_CODE=/path/to/hornsrev-checkout python analyse_hr.py .   # Horns Rev (needs the hornsrev branch code)
```

`reanalyze.py gauss` run this way reproduces the stored JSON byte for byte.
