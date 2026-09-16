# Module 8H — Final 10-Seed Confirmatory Benchmark

## Status

`COMPLETE_CONFIRMATORY`

Module 8H completed the locked matched-seed design for seeds `11, 22, 33, 44, 55, 66, 77, 88, 99, 110`. Training ran on CUDA with `reward_v4`, validation-2024-only checkpoint selection, and no seed replacement or selective extra training. The retrospective final benchmark 2025 was opened only after all checkpoint registries were complete.

The main table contains `DDPG`, `TD3`, `SAC`, and `SACSI-POMDP` (`4 × 10 = 40` rows). The factorial table contains `SAC Basic`, `SAC + Forecast`, `SAC + LSTM`, and `SACSI Full` (`4 × 10 = 40` rows).

## Primary endpoint

Time in Target (%) is the pre-specified primary endpoint. Water and the remaining formal metrics are secondary/descriptive.

| Main pipeline | Time in Target, mean ± SD | Irrigation, mean ± SD (mm) |
|---|---:|---:|
| SACSI-POMDP | 55.018 ± 1.927 | 372.197 ± 42.601 |
| SAC | 54.116 ± 2.082 | 394.098 ± 87.313 |
| TD3 | 41.850 ± 12.696 | 176.153 ± 186.339 |
| DDPG | 41.279 ± 12.798 | 151.509 ± 167.889 |

The low irrigation totals for DDPG/TD3 do not establish better efficiency: many of those checkpoints under-irrigated and had substantially lower target occupancy. Sixteen main-table rows and two factorial-table rows failed the validation gate; every one was retained as pre-specified.

## Main confirmatory inference

The Friedman omnibus result was `χ²(3) = 15.064`, `p = 0.001763`. The one-way repeated-measures ANOVA sensitivity result was `F(3,27) = 7.946`, `p = 0.000590`, partial `η² = 0.469`.

Primary planned contrasts used two-sided exact paired sign-flip tests with Holm correction. Paired t tests are sensitivity results, not replacements for the exact primary test.

| Planned contrast | Mean difference (pp) | 95% bootstrap CI | Cohen's dz | Exact Holm p |
|---|---:|---:|---:|---:|
| SACSI-POMDP − SAC | 0.902 | [0.336, 1.531] | 0.880 | 0.046875 |
| SACSI-POMDP − TD3 | 13.168 | [5.402, 20.854] | 0.996 | 0.046875 |
| SACSI-POMDP − DDPG | 13.740 | [5.660, 21.105] | 1.034 | 0.046875 |

All three contrasts support superiority of the **locked SACSI training pipeline** for Time in Target under this experiment. The SAC contrast is statistically supported but close to the `0.05` boundary and has a small absolute difference of `0.902` percentage points; it must not be described as a large operational gain.

## Factorial family

| Variant | Time in Target, mean ± SD | Irrigation, mean ± SD (mm) |
|---|---:|---:|
| SAC Basic | 54.116 ± 2.082 | 394.098 ± 87.313 |
| SAC + Forecast | 53.259 ± 3.338 | 383.589 ± 67.644 |
| SAC + LSTM | 54.374 ± 2.629 | 362.767 ± 52.835 |
| SACSI Full | 55.018 ± 1.927 | 372.197 ± 42.601 |

| Factorial effect | Mean effect (pp) | 95% bootstrap CI | Partial η² | Exact Holm p | Decision |
|---|---:|---:|---:|---:|---|
| Forecast | -0.107 | [-0.632, 0.422] | 0.016 | 0.964844 | Not supported |
| Memory | 1.009 | [0.579, 1.438] | 0.680 | 0.017578 | Supported for locked pipeline |
| Forecast × Memory | 1.501 | [-0.865, 4.771] | 0.091 | 0.964844 | Not supported |

Factorial pairwise exact-Holm results support SACSI Full over SAC Basic (`p = 0.046875`) and SAC + Forecast (`p = 0.046875`), but not over SAC + LSTM (`p = 0.451172`). Therefore:

- a standalone forecast benefit is not supported;
- the memory pipeline effect is supported;
- forecast-memory interaction/synergy is not supported;
- SACSI is not demonstrably better than the memory-only variant.

## Mandatory protocol limitation

The context variants use the locked SAC anchor (`6,720` interactions) followed by `6,720` context-adaptation interactions. DDPG, TD3, and SAC use `6,720` total interactions. Consequently, Module 8H confirms differences between the **locked end-to-end training pipelines**, not a pure equal-total-budget causal architecture effect.

Allowed wording:

> Under the locked warm-start training pipelines, SACSI-POMDP achieved higher matched-seed Time in Target than SAC, TD3, and DDPG, with all pre-specified exact-Holm contrasts below 0.05.

Not allowed:

> The SACSI architecture alone is universally superior under an equal training budget.

The forecast remains an `SF-20 h+1 controlled synthetic forecast proxy`, not an archived operational forecast. These results are virtual-garden evidence, not field-effectiveness evidence.

## Acceptance audit

- [x] main table has 40 unique model-seed rows;
- [x] factorial table has 40 unique variant-seed rows;
- [x] all ten matched seeds are present;
- [x] no primary endpoint is missing or non-finite;
- [x] all 60 unique checkpoints use `reward_v4` and CUDA training;
- [x] failed validation seeds are retained;
- [x] benchmark 2025 was not used for training or checkpoint selection;
- [x] Friedman, exact sign-flip, Holm, Cohen's dz, bootstrap CI, and parametric sensitivities are exported;
- [x] mass-balance maximum is below `1e-8`;
- [x] negative and null forecast/interaction results are retained;
- [x] automated suite passes (`62 passed`).

## Source artifacts

All final files are in `Results/Confirmatory_10Seed/`:

- `main_10seed_results_2025.csv`
- `sac_family_10seed_factorial.csv`
- `friedman_results.csv`
- `planned_contrasts.csv`
- `holm_adjusted_results.csv`
- `bootstrap_ci.csv`
- `factorial_inference.csv`
- `final_statistics_summary.json`
- `confirmatory_manifest.json`

## Manual verification

```powershell
cd D:\ARYA\SACSI_Dissertation
python scripts\run_confirmatory_benchmark.py --smoke
python scripts\run_confirmatory_benchmark.py --statistics-only
python -m pytest -p no:cacheprovider tests\test_confirmatory_benchmark.py -q
```

Expected final targeted result: `4 passed`.
