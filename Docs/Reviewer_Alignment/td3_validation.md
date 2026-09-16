# Module 8E — TD3 Continuous-Control Baseline

Status: **IMPLEMENTATION VALID**  
Experiment class: **3-seed development validation**  
Compute: **CUDA GPU**  
Reward: **locked `reward_v4`**  
Benchmark 2025 accessed: **No**

## Locked protocol

- Observation: current state only, 8 dimensions.
- Action: deterministic continuous irrigation in `0–5 mm/hour`.
- Actor: two hidden layers, 64 units per layer.
- Critics: two independently initialized Q-networks with the same hidden capacity.
- Clipped Double-Q target: minimum target-Q estimate.
- Target-policy smoothing: Gaussian noise standard deviation `0.2 mm`, clipped to `±0.5 mm` before the action is bounded.
- Delayed policy update: one actor/target update per two critic updates.
- Forecast: off.
- History/memory: off.
- Training split: 2021–2023.
- Checkpoint-selection split: validation 2024 only.
- Initial development seeds: 11, 22, and 33.
- Budget: 20 episodes × 336 hours = 6,720 environment interactions per seed.
- Selection priority: validation gate, highest Time in Target, lowest water use, then lowest band RMSE.

The environment, reward, observation, action bounds, data split, seed set, interaction budget, metric engine, and checkpoint-selection rule match the Module 8D DDPG development protocol. The retrospective 2025 benchmark was not used.

## Development validation results

| Seed | Selected episode | Time in Target | Irrigation | Band RMSE | Gate |
|---:|---:|---:|---:|---:|---|
| 11 | 20 | 56.33% | 555.49 mm | 0.02703 | Fail |
| 22 | 15 | 57.42% | 548.42 mm | 0.02684 | Pass |
| 33 | 20 | 23.84% | 0.0023 mm | 0.04303 | Fail |

Each run completed 6,221 critic updates and 3,110 delayed actor updates. All runs used CUDA, retained finite actor/critic losses, respected the action bounds, and had a maximum absolute water-balance residual of approximately `2.84e-14 mm`.

Seeds 11 and 22 learned active irrigation policies with similar validation results, but seed 33 converged to an effectively zero-irrigation policy. TD3 therefore improves stability relative to the initial DDPG runs on two seeds but is not seed-stable under this small development experiment. This observation is descriptive only; matched final comparison and inferential claims remain deferred to Modules 8F and 8H.

`water_use_efficiency` should not be interpreted for the near-zero-irrigation seed because division by an almost-zero denominator produces a very large value. Time in Target, irrigation, violations, and band RMSE remain the decision metrics.

## Reproducibility outputs

- Config: `configs/td3_config.yaml`
- Checkpoints: `Checkpoints/TD3/td3_seed{11,22,33}_best.pt`
- Validation summary: `Results/TD3/td3_validation_results.csv`
- Training log: `Results/TD3/td3_training_log.csv`
- Checkpoint-selection log: `Results/TD3/td3_checkpoint_selection.csv`
- Per-seed metadata and trajectories: `Results/TD3/` and `Logs/TD3/`

## Acceptance gate 8E

- [x] Twin critics are independently initialized.
- [x] Clipped Double-Q target uses the minimum target critic.
- [x] Target-action noise is clipped and the final action remains bounded.
- [x] Delayed actor update is verified at `policy_delay=2`.
- [x] Actions are finite and within `0–5 mm/hour`.
- [x] Checkpoint save/load round trip is valid.
- [x] GPU smoke test completes.
- [x] Forecast and history are not used.
- [x] Failed development seeds are retained and documented.

Module 8E is complete at the implementation-validation level.
