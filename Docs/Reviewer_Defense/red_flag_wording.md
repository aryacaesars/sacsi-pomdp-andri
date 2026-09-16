# Red-Flag Wording — Module 9C

Gunakan daftar ini saat menulis, presentasi, dan menjawab reviewer. Jika wording pada kolom pertama muncul, ganti dengan wording yang dibatasi evidence.

| Jangan katakan | Gunakan wording | Status | Evidence |
|---|---|---|---|
| Field validation | The simulator passed the locked numerical and physical-response gates. | RELEASED | `Results/Simple_Case_Validation/simple_case_results.csv|Results/Simple_Case_Validation/raw_episode_summary.csv` |
| Selection based on 2025 | reward_v4 was selected on validation 2024 by the pre-specified Pareto/stability rule. | RELEASED | `Results/Reward_Validation/reward_confirmation_decision.json` |
| Mechanisms were made identical | The three algorithms used the locked common environment, split, reward, interaction budget, seeds, and checkpoint rule. | RELEASED | `Results/Fair_DRL/fairness_audit.json` |
| Activation proves benefit | Context diagnostics show branch activation under the exploratory protocol. | RELEASED_WITH_GUARD | `Results/POMDP_Ablation/context_intervention_results.csv|Results/POMDP_Ablation/pomdp_ablation_manifest.json` |
| Forecast improves performance | The forecast main effect was not supported. | NOT_RELEASED | `Results/Confirmatory_10Seed/factorial_inference.csv` |
| Universal memory benefit | The memory main effect was supported for the locked warm-start pipelines. | RELEASED_WITH_GUARD | `Results/Confirmatory_10Seed/factorial_inference.csv` |
| Synergy is proven | The forecast-by-memory interaction was not supported. | NOT_RELEASED | `Results/Confirmatory_10Seed/factorial_inference.csv` |
| Universal equal-budget architecture superiority | The locked SACSI pipeline achieved higher matched-seed Time in Target in the retrospective 2025 simulation benchmark. | RELEASED_WITH_GUARD | `Results/Confirmatory_10Seed/main_10seed_results_2025.csv|Results/Confirmatory_10Seed/planned_contrasts.csv` |
| Statistical robustness superiority | SF10–SF30 results are exploratory robustness diagnostics. | DESCRIPTIVE_ONLY | `Results/POMDP_Ablation/forecast_robustness.csv` |
| All inputs are field measurements | Meteorology is raw/real, soil state is simulated, and forecast input is a controlled synthetic proxy. | RELEASED | `Docs/Reviewer_Alignment/scope_and_data_classification.md` |
| IoT field deployment was validated | IoT is an implementation and sensing context. | CONTEXT_ONLY | `Docs/Reviewer_Alignment/scope_and_data_classification.md` |
| Energy savings | No energy metric is reported because no pump-power model is available. | NOT_REPORTED | `Docs/Reviewer_Alignment/scope_and_data_classification.md` |
| Real-world field effectiveness | Evidence is limited to the locked Virtual Garden simulation. | NOT_RELEASED | `Docs/Reviewer_Alignment/scope_and_data_classification.md` |
| Forecast meningkatkan performa SACSI. | Standalone forecast main effect tidak didukung; hanya memory main effect yang didukung. | BLOCK | `Results/Confirmatory_10Seed/factorial_inference.csv` |
| SACSI lebih baik daripada SAC + LSTM. | Perbandingan SACSI Full dengan SAC + LSTM tidak signifikan setelah exact-Holm. | BLOCK | `Results/Confirmatory_10Seed/planned_contrasts.csv` |
| DDPG, TD3, dan SAC terbukti berbeda signifikan. | Fair comparison tersedia, tetapi hipotesis inferensial langsung khusus tiga algoritma belum konklusif. | BLOCK | `Results/Dissertation_Evidence/hypothesis_decision_table.csv` |
| Air paling sedikit berarti controller paling efisien. | Water harus dibaca bersama Time in Target, violation, deficit, dan Pareto trade-off. | BLOCK | `Results/Confirmatory_10Seed/main_10seed_results_2025.csv` |
| Semua seed berhasil melewati validation gate. | Failed validation seeds dipertahankan sesuai protokol. | BLOCK | `Results/Confirmatory_10Seed/final_statistics_summary.json` |

## Aturan lisan cepat

1. Awali klaim performa dengan **“under the locked warm-start training pipelines”**.
2. Sebut evaluasi sebagai **retrospective 2025 Virtual Garden simulation benchmark**.
3. Sebut forecast sebagai **SF-20 h+1 controlled synthetic forecast proxy**.
4. Pisahkan **framework validity**, **context activation**, **performance benefit**, dan **statistical superiority**.
5. Jika ditanya deployment, jawab **belum diuji di lapangan**.
