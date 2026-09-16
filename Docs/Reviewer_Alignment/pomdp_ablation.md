# Module 8G — Incremental POMDP Contribution and Ablation

Status: **COMPLETE — EXPLORATORY REVALIDATION**  
Training compute: **CUDA GPU**  
Reward: **locked `reward_v4`**  
Matched development seeds: **11, 22, 33**  
Evaluation: **retrospective benchmark 2025**

## Protocol synchronization

The historical Sprint 11 ablation remains available as reward-v2 evidence, but it is not the primary Module 8G result. The SAC family was synchronized to the latest reward and checkpoint-selection decisions:

- F0M0: fair SAC Basic checkpoints from Module 8F;
- F1M0: SAC + Forecast with SF20 h+1 context;
- F0M1: SAC + LSTM with causal `24 × 8` history;
- F1M1: SACSI Full with both history and SF20 context.

Every context variant starts from the matched fair SAC anchor and receives an equal adaptation budget of `20 × 336 = 6,720` interactions. Episode 0 remains a valid checkpoint candidate. Selection uses validation 2024 only: gate, highest Time in Target, lowest water, then lowest band RMSE. The 2025 data was opened only after all training and checkpoint selection were locked.

This is an incremental warm-start study, not a pure equal-total-training-budget causal experiment: context variants include the anchor budget plus adaptation. Performance effects are therefore exploratory and must be confirmed in Module 8H.

## Selected checkpoints

| Variant | Seed 11 | Seed 22 | Seed 33 |
|---|---:|---:|---:|
| SAC Basic | 20 | 20 | 20 |
| SAC + Forecast | 0 | 0 | 5 |
| SAC + LSTM | 10 | 15 | 5 |
| SACSI Full | 10 | 10 | 0 |

All 12 checkpoints passed the validation gate. Episode 0 means the adaptation did not beat the anchor under the locked selection rule; it is not treated as a successful context activation.

## Retrospective 2025 factorial results

Mean ± sample standard deviation over three matched seeds:

| Variant | Time in Target | Irrigation |
|---|---:|---:|
| SAC Basic | 55.40 ± 0.92% | 366.72 ± 24.09 mm |
| SAC + Forecast | 52.44 ± 4.85% | 351.00 ± 48.87 mm |
| SAC + LSTM | 54.02 ± 3.92% | 337.75 ± 18.03 mm |
| SACSI Full | 55.68 ± 0.66% | 365.63 ± 26.11 mm |

SACSI Full is descriptively `+0.28` percentage points above the SAC anchor, but the sample is only three seeds. Forecast-only and memory-only means are lower than the anchor. No performance superiority or statistical-significance claim is released from these aggregate values.

For Time in Target, mean factorial effects were:

- forecast main effect: `-0.65` percentage points;
- memory main effect: `+0.93` percentage points;
- interaction: `+4.62` percentage points.

The interaction is strongly influenced by seed 33, where Forecast and LSTM degraded separately while SACSI selected episode 0 and reproduced the anchor. It must not be interpreted as inferential evidence.

## Context activation diagnostics

| Seed | History norm | Forecast norm | History active | Forecast active |
|---:|---:|---:|---|---|
| 11 | 0.5920 | 0.1950 | Yes | Yes |
| 22 | 0.4839 | 0.2648 | Yes | Yes |
| 33 | 0.0000 | 0.0000 | No | No |

Mean absolute action deltas against Full were approximately:

- No History: `0.01245 mm/hour`;
- Shuffled History: `0.01184 mm/hour`;
- Reversed History: `0.01299 mm/hour`;
- Zero History: `0.01263 mm/hour`;
- No Forecast: `0.00221 mm/hour`;
- Shuffled Forecast: `0.00293 mm/hour`;
- Zero Forecast: `0.00193 mm/hour`;
- No Context/current-only: `0.01296 mm/hour`.

Thus the selected seed-11/22 branches are behaviorally active, while seed 33 is explicitly retained as inactive. Activation alone does not establish benefit.

## Forecast robustness

| Condition | Mean Time in Target | Mean irrigation |
|---|---:|---:|
| SF10 | 55.68% | 365.69 mm |
| SF20 | 55.68% | 365.63 mm |
| SF30 | 55.70% | 365.52 mm |

The three noise levels are descriptively very similar for these locked checkpoints. This supports local output stability, not a general robustness or superiority claim.

## Sequence sensitivity

| Sequence | Mean Time in Target | Mean irrigation |
|---:|---:|---:|
| k6 | 55.88% | 360.03 mm |
| k12 | 55.89% | 361.84 mm |
| k24 | 55.68% | 365.63 mm |
| k48 | 55.68% | 365.75 mm |

k6 and k12 are slightly higher than the primary k24 representation. These are inference-only window interventions on k24-trained checkpoints, so they do not justify retuning after 2025 exposure.

## Evidence outputs

- `Results/POMDP_Ablation/sac_family_factorial_results.csv`
- `Results/POMDP_Ablation/context_intervention_results.csv`
- `Results/POMDP_Ablation/forecast_robustness.csv`
- `Results/POMDP_Ablation/sequence_sensitivity.csv`
- `Results/POMDP_Ablation/factorial_effects.csv`
- `Results/POMDP_Ablation/sac_family_checkpoint_registry.csv`
- `Results/POMDP_Ablation/pomdp_ablation_manifest.json`

## Acceptance gate 8G

- [x] Complete 2×2 SAC-family matrix.
- [x] Forecast and history activation diagnostics complete.
- [x] Full/zero/shuffle forecast interventions complete.
- [x] Current-only/zero/shuffle/reverse history interventions complete.
- [x] SF10/SF20/SF30 robustness complete.
- [x] k6/k12/k24/k48 sequence sensitivity complete.
- [x] Failed/inactive context seeds retained.
- [x] No branch claimed beneficial without metric evidence.
- [x] No statistical significance claimed from aggregate means.

Module 8G is complete at the three-seed exploratory-revalidation level.
