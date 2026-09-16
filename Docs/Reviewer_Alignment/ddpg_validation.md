# Module 8D — DDPG Continuous-Control Baseline

Status: **IMPLEMENTATION VALID**  
Experiment class: **3-seed development validation**  
Compute: **CUDA GPU**  
Reward: **locked `reward_v4`**  
Benchmark 2025 accessed: **No**

## Locked protocol

- Observation: current state only, 8 dimensions.
- Action: deterministic continuous irrigation in `0–5 mm/hour`.
- Actor: two hidden layers, 64 units per layer.
- Critic: one Q-network with the same hidden capacity.
- Core mechanisms: replay buffer, Gaussian exploration noise, target actor/critic, and soft target updates.
- Forecast: off.
- History/memory: off.
- Training split: 2021–2023.
- Checkpoint-selection split: validation 2024 only.
- Initial development seeds: 11, 22, and 33.
- Budget: 20 episodes × 336 hours = 6,720 environment interactions per seed.
- Selection priority: validation gate, highest Time in Target, lowest water use, then lowest band RMSE.

The retrospective 2025 benchmark was not used for training, tuning, or checkpoint selection.

## Development validation results

| Seed | Selected episode | Time in Target | Irrigation | Band RMSE | Gate |
|---:|---:|---:|---:|---:|---|
| 11 | 20 | 23.84% | 0.0006 mm | 0.04303 | Fail |
| 22 | 15 | 45.95% | 468.90 mm | 0.02754 | Fail |
| 33 | 20 | 23.84% | 0.0004 mm | 0.04303 | Fail |

All three runs used CUDA, retained finite actor/critic losses, respected the action bounds, and had a maximum absolute water-balance residual of approximately `2.84e-14 mm`.

Seeds 11 and 33 converged to an effectively zero-irrigation policy, while seed 22 learned an active but still gate-failing policy. This is evidence that the DDPG implementation is technically valid but unstable under the initial locked budget. Failed seeds are deliberately retained. These development results do not support a performance or superiority claim and are not the final 10-seed benchmark.

`water_use_efficiency` should not be interpreted for the near-zero-irrigation seeds because division by an almost-zero denominator produces a very large value. Time in Target, irrigation, violations, and band RMSE remain the decision metrics.

## Reproducibility outputs

- Config: `configs/ddpg_config.yaml`
- Checkpoints: `Checkpoints/DDPG/ddpg_seed{11,22,33}_best.pt`
- Validation summary: `Results/DDPG/ddpg_validation_results.csv`
- Training log: `Results/DDPG/ddpg_training_log.csv`
- Checkpoint-selection log: `Results/DDPG/ddpg_checkpoint_selection.csv`
- Per-seed metadata and trajectories: `Results/DDPG/` and `Logs/DDPG/`

## Acceptance gate 8D

- [x] Unit tests pass.
- [x] Action bounds are valid.
- [x] Actor and critic losses are finite.
- [x] Replay-buffer sampling is valid.
- [x] Soft target update is valid.
- [x] Checkpoint save/load round trip is valid.
- [x] GPU smoke test completes.
- [x] Forecast and history are not used.
- [x] Failed development seeds are retained and documented.

Module 8D is complete at the implementation-validation level. Final comparative inference remains deferred to Modules 8F and 8H.
