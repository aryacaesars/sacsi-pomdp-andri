# Module 8F — Fair DDPG–TD3–SAC Benchmark

Status: **FAIRNESS AUDIT PASSED**  
Experiment class: **3-seed development benchmark**  
Training compute: **CUDA GPU**  
Inference: **deterministic CPU**  
Reward: **locked `reward_v4`**  
Benchmark classification: **retrospective final benchmark 2025**

## Why SAC was retrained

The historical SAC Basic checkpoints used `reward_v2` and a 100-episode budget. Reusing them would violate the Module 8F fairness lock because DDPG and TD3 use `reward_v4` and 20 episodes. SAC was therefore retrained on seeds 11, 22, and 33 using the same common protocol. DDPG and TD3 checkpoints from Modules 8D/8E were reused without retraining.

## Common fairness lock

| Field | Locked value |
|---|---|
| Virtual Garden | `field_capacity_0.35` |
| Observation | current-state 8-D |
| Action | 0–5 mm/hour |
| Hidden capacity | 2 × 64 units |
| Batch / warm-up | 64 / 500 steps |
| Actor / critic LR | 0.0005 / 0.0005 |
| Gamma / tau | 0.99 / 0.005 |
| Reward | `reward_v4` |
| Train / validation | 2021–2023 / 2024 |
| Seeds | 11, 22, 33 |
| Budget | 20 × 336 = 6,720 interactions/seed |
| Validation interval | 5 episodes |
| Checkpoint selection | gate → highest TiT → lowest water → lowest RMSE |
| Forecast / history | off / off |

Common fairness hash for DDPG, TD3, and SAC:

```text
a4ba42f6f0a4ccabeb9ca6bb642d07cf122a2c6b56329d3c4f94c01ea436464a
```

Algorithm-specific mechanisms were preserved: DDPG uses one critic and deterministic exploration; TD3 uses twin critics, target smoothing, and policy delay; SAC uses twin critics, a stochastic actor, and automatic entropy tuning.

## Validation 2024 summary

Mean ± sample standard deviation over three matched seeds:

| Rank | Algorithm | Time in Target | Irrigation | Band RMSE |
|---:|---|---:|---:|---:|
| 1 | SAC | 58.71 ± 0.24% | 580.51 ± 36.73 mm | 0.02706 ± 0.00026 |
| 2 | TD3 | 45.86 ± 19.08% | 367.97 ± 318.69 mm | 0.03230 ± 0.00929 |
| 3 | DDPG | 31.21 ± 12.76% | 156.30 ± 270.72 mm | 0.03787 ± 0.00894 |

All three fair SAC checkpoints passed the validation gate. TD3 retained one passing and two failed seeds; DDPG retained all three failed seeds.

## Retrospective 2025 summary

Training, fairness hashing, and validation-only checkpoint selection were completed before the Module 8F runner loaded the 2025 data. No training, tuning, or checkpoint reselection occurred after opening it.

| Rank | Algorithm | Time in Target | Irrigation | Band RMSE |
|---:|---|---:|---:|---:|
| 1 | SAC | 55.40 ± 0.92% | 366.72 ± 24.09 mm | 0.02218 ± 0.00020 |
| 2 | TD3 | 45.83 ± 13.80% | 238.47 ± 206.58 mm | 0.02521 ± 0.00522 |
| 3 | DDPG | 33.61 ± 6.44% | 55.97 ± 96.95 mm | 0.02982 ± 0.00245 |

SAC is descriptively highest and most seed-stable in this three-seed benchmark. TD3 is intermediate but retains a collapsed seed, while two DDPG seeds remain near zero irrigation. These results do not establish statistical superiority: the final claim requires the matched 10-seed confirmatory design in Module 8H.

## Evidence outputs

- Validation rows: `Results/Fair_DRL/fair_drl_results_validation.csv`
- Retrospective 2025 rows: `Results/Fair_DRL/fair_drl_results_2025.csv`
- Checkpoint registry and SHA256: `Results/Fair_DRL/fair_drl_checkpoint_registry.csv`
- Fairness evidence: `Results/Fair_DRL/fairness_audit.json`
- Validation and 2025 summaries: `Results/Fair_DRL/fair_drl_summary_*.csv`
- Fair SAC checkpoints: `Checkpoints/Fair_DRL/SAC/`

## Acceptance gate 8F

- [x] Common-field config hash is identical across DDPG/TD3/SAC.
- [x] All algorithms use matched seeds 11/22/33.
- [x] Training budget is fixed at 6,720 interactions per seed.
- [x] Checkpoint selection uses validation 2024 only.
- [x] The 2025 file has 8,760 unique continuous hourly timestamps.
- [x] Validation and 2025 metrics use one common schema.
- [x] All failed seeds remain in results and checkpoint registry.
- [x] Nine checkpoints exist and their SHA256 values match.
- [x] No retraining or checkpoint reselection occurred after the Module 8F runner opened 2025.

Module 8F is complete at the three-seed development-benchmark level.
