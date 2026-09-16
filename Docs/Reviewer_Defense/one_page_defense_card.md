# One-Page Defense Card — SACSI-POMDP

## Satu kalimat kontribusi

SACSI-POMDP adalah kerangka evaluasi modular untuk continuous smart-irrigation control yang menggabungkan current observation, temporal memory, dan controlled forecast context dalam Virtual Garden, dengan ablation terkontrol dan confirmatory matched-seed inference.

## Desain yang harus diingat

- Scope: hortikultura generik dalam Virtual Garden; belum ada field validation.
- Data: real/raw meteorology → simulated soil state → SF-20 h+1 controlled synthetic forecast proxy.
- Split: training 2021–2023; reward/checkpoint selection 2024; retrospective benchmark 2025.
- Primary endpoint: Time in Target (%).
- Main comparison: DDPG, TD3, SAC, dan SACSI-POMDP pada 10 matched seeds.
- Factorial: SAC Basic, SAC + Forecast, SAC + LSTM, SACSI Full.

## Angka inti

- Mean Time in Target: SACSI 55.018%; SAC 54.116%; TD3 41.850%; DDPG 41.279%.
- SACSI differences: SAC +0.902 pp; TD3 +13.168 pp; DDPG +13.740 pp; seluruh exact-Holm p = 0.046875.
- Memory main effect: +1.009 pp, 95% bootstrap CI [0.579, 1.438], exact-Holm p = 0.017578.
- Forecast main effect dan forecast × memory interaction: tidak didukung.

## Keputusan H1–H4

- H1: supported engineering gates; bukan field validation.
- H2: direct three-algorithm inference inconclusive; laporkan hasil deskriptif saja.
- H3: reject null dengan component guard; memory didukung, forecast dan interaction null.
- H4: reject null untuk locked pipeline scope; bukan equal-total-budget architecture claim.

## Jawaban aman saat ditekan reviewer

- “Superior?” → **Ya untuk locked pipeline dan Time in Target pada retrospective simulation benchmark; bukan universal atau field claim.**
- “Forecast berguna?” → **Branch aktif, tetapi standalone performance benefit tidak didukung.**
- “Mengapa tetap novel?” → **Novelty berada pada integrasi modular, controlled ablation, dan evidence hierarchy; novelty tidak mensyaratkan semua efek positif.**
- “Fair budget?” → **Main DDPG/TD3/SAC fair; SACSI context pipeline memiliki tambahan adaptation budget, sehingga klaim arsitektur equal-budget ditahan.**
- “Siap deploy?” → **Belum; diperlukan calibration, hardware-in-the-loop, archived forecasts, dan field trials.**

## Source cepat

`Results/Confirmatory_10Seed/main_10seed_results_2025.csv` · `planned_contrasts.csv` · `factorial_inference.csv` · `final_statistics_summary.json` · `Results/Reviewer_Defense/claim_to_evidence_matrix.csv`
