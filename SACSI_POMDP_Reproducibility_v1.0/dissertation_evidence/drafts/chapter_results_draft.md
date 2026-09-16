# Draft Bab 5 — Hasil

Status evidence: **FROZEN / READY FOR EDITORIAL INTEGRATION**

Primary endpoint: **Time in Target (%)**

Naskah ini dihasilkan dari artefak beku Modul 8A–9A. Penyuntingan gaya bahasa diperbolehkan, tetapi angka dan batas klaim harus tetap mengikuti `result_insertion_matrix.csv` dan `claim_matrix.csv`.

## 5.1 Data Audit & Provenance

Eksperimen menggunakan forcing meteorologi observasional untuk simulator. Kelembapan tanah, runoff, drainage, deficit, dan surplus adalah keluaran Virtual Garden, bukan pengukuran tanah lapangan. Input forecast controller adalah **SF-20 h+1 controlled synthetic forecast proxy**, bukan archived as-issued operational forecast. [Source: `Docs/Reviewer_Alignment/scope_and_data_classification.md`]

Training menggunakan 2021–2023, pemilihan reward dan checkpoint hanya menggunakan 2024, sedangkan 2025 dibuka setelah registry checkpoint terkunci sebagai retrospective final benchmark. Tidak dilakukan retraining atau checkpoint reselection setelah pembukaan benchmark. [Source: `Results/Fair_DRL/fairness_audit.json`] [Source: `Results/Confirmatory_10Seed/final_statistics_summary.json`]

## 5.2 Virtual Garden Validation

Virtual Garden menggunakan target kelembapan 0.22–0.32 m³/m³ dan batas aksi irigasi 0–5 mm/hour. Seluruh 6 dari 6 simple cases memenuhi respons yang diharapkan, aksi terbatas, keluaran finite, dan pemeriksaan neraca massa. Residual neraca massa maksimum pada episode raw-data adalah 2.842e-14 mm. Hasil ini mendukung konsistensi numerik simulator, bukan validasi lapangan. [Source: `Docs/Reviewer_Alignment/research_question_objective_map.csv`] [Source: `Results/Fair_DRL/fairness_audit.json`] [Source: `Results/Simple_Case_Validation/simple_case_results.csv`] [Source: `Results/Simple_Case_Validation/raw_episode_summary.csv`]

## 5.3 Reward Validation

Reward final dikunci sebagai **reward_v4** melalui validasi 2024 tanpa mengakses benchmark 2025. Kandidat terpilih bersifat Pareto non-dominated dan mencapai Time in Target 56.828 ± 5.152% dengan irigasi 613.445 ± 146.976 mm pada 10 seed. Keputusan ini adalah trade-off multi-objective, bukan pemaksimalan cumulative reward semata. [Source: `Results/Reward_Validation/reward_confirmation_decision.json`]

## 5.4 Simple-Case & Raw-Data Validation

Enam simple cases mencakup pengeringan, pulse hujan, pulse irigasi, kondisi dekat batas atas, pemulihan dari kondisi di bawah target, dan respons anticipatory terhadap hujan. Episode DRY, WET, dan MIXED menggunakan forcing meteorologi 2024 yang sama untuk controller referensi tanpa retuning per episode. Respons soil moisture tetap merupakan simulasi. [Source: `Results/Simple_Case_Validation/simple_case_results.csv`] [Source: `Results/Simple_Case_Validation/raw_episode_summary.csv`]

## 5.5 Fair DDPG–TD3–SAC Benchmark

Audit fairness berstatus **PASS**: DDPG, TD3, dan SAC memakai environment, observation, action bounds, reward, split, interaction budget, seed, metric engine, dan checkpoint rule yang sama, sementara mekanisme algoritmik masing-masing dipertahankan. [Source: `Results/Fair_DRL/fairness_audit.json`]

Pada benchmark pengembangan tiga-seed, Time in Target deskriptif adalah SAC 55.400 ± 0.917%, TD3 45.833 ± 13.804%, dan DDPG 33.615 ± 6.439%. Angka ini merupakan evidence pengembangan deskriptif; keputusan final menggunakan desain sepuluh-seed. [Source: `Results/Fair_DRL/fair_drl_results_2025.csv`]

## 5.6 Incremental POMDP Contribution

Eksperimen eksploratori tiga-seed membentuk desain 2×2 SAC Basic, SAC + Forecast, SAC + LSTM, dan SACSI Full serta intervensi terhadap history dan forecast. Diagnostik menunjukkan bahwa branch context dapat diaktifkan, tetapi aktivasi tidak disamakan dengan manfaat performa atau superioritas statistik. [Source: `Results/POMDP_Ablation/pomdp_ablation_manifest.json`] [Source: `Results/POMDP_Ablation/context_intervention_results.csv`]

## 5.7 10-Seed Confirmatory Benchmark

Tabel final berisi 40 baris untuk empat pipeline dan 10 matched seeds. Seluruh seed yang gagal validation gate tetap dipertahankan. [Source: `Results/Confirmatory_10Seed/main_10seed_results_2025.csv`] [Source: `Results/Confirmatory_10Seed/final_statistics_summary.json`]

| Pipeline | Time in Target mean ± SD (%) | Irrigation mean ± SD (mm) |
|---|---|---|
| SACSI-POMDP | 55.018 ± 1.927 | 372.197 ± 42.601 |
| SAC | 54.116 ± 2.082 | 394.098 ± 87.313 |
| TD3 | 41.850 ± 12.696 | 176.153 ± 186.339 |
| DDPG | 41.279 ± 12.798 | 151.509 ± 167.889 |

Friedman omnibus menunjukkan perbedaan kondisi, χ²(3) = 15.064, p = 0.001763. [Source: `Results/Confirmatory_10Seed/friedman_results.csv`]

| Planned contrast | Difference (pp) | 95% bootstrap CI | Cohen's dz | Exact-Holm p |
|---|---|---|---|---|
| SACSI-POMDP - SAC | 0.902 | [0.336, 1.531] | 0.880 | 0.046875 |
| SACSI-POMDP - TD3 | 13.168 | [5.402, 20.854] | 0.996 | 0.046875 |
| SACSI-POMDP - DDPG | 13.740 | [5.660, 21.105] | 1.034 | 0.046875 |

Ketiga planned contrast mendukung Time in Target yang lebih tinggi untuk **locked SACSI warm-start training pipeline** pada benchmark simulasi retrospektif 2025. Selisih terhadap SAC hanya 0.902 percentage points sehingga tidak ditafsirkan sebagai peningkatan operasional besar. [Source: `Results/Confirmatory_10Seed/planned_contrasts.csv`]

## 5.8 Robustness & Diagnostics

Tabel factorial berisi 40 baris untuk empat varian dan sepuluh matched seeds. [Source: `Results/Confirmatory_10Seed/sac_family_10seed_factorial.csv`]

| Effect | Mean (pp) | 95% bootstrap CI | Exact-Holm p | Decision |
|---|---|---|---|---|
| Forecast main effect | -0.107 | [-0.632, 0.422] | 0.964844 | not supported |
| Memory main effect | 1.009 | [0.579, 1.438] | 0.017578 | supported |
| Forecast x Memory interaction | 1.501 | [-0.865, 4.771] | 0.964844 | not supported |

Hanya memory main effect yang didukung. Standalone forecast effect dan forecast × memory interaction tidak didukung. Eksperimen SF10, SF20, dan SF30, sequence window 6, 12, 24, dan 48 jam, serta 9 kondisi intervensi diperlakukan sebagai diagnostik eksploratori, bukan uji superiority baru. [Source: `Results/Confirmatory_10Seed/factorial_inference.csv`] [Source: `Results/POMDP_Ablation/forecast_robustness.csv`] [Source: `Results/POMDP_Ablation/sequence_sensitivity.csv`] [Source: `Results/POMDP_Ablation/context_intervention_results.csv`]

## 5.9 Jawaban RM1–RM4

- **RM1/H1:** gate rekayasa Virtual Garden dan reward didukung, dengan batas bahwa evidence berasal dari simulasi.
- **RM2/H2:** fairness dan hasil deskriptif DDPG–TD3–SAC tersedia, tetapi hipotesis inferensial khusus tiga algoritma belum konklusif karena omnibus beku mencakup SACSI sebagai kondisi keempat.
- **RM3/H3:** null gabungan ditolak karena memory main effect didukung; forecast dan interaction tetap null.
- **RM4/H4:** null ditolak untuk ruang lingkup locked warm-start pipelines pada benchmark simulasi retrospektif; klaim equal-total-budget architecture dan field effectiveness tidak dibuka.

Keputusan lengkap dan evidence per hipotesis tersedia di `Results/Dissertation_Evidence/hypothesis_decision_table.csv`.
