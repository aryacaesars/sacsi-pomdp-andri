# Draft Bab 6 — Pembahasan

## 6.1 Reward trade-off

Pemilihan reward_v4 memperlihatkan bahwa objective irigasi tidak cukup dinilai dari cumulative reward. Time in Target, penggunaan air, violation, deficit, kestabilan antar-seed, dan Pareto non-dominance harus dibaca bersama. Karena reward dipilih hanya pada validasi 2024, hasil 2025 tetap berfungsi sebagai evaluasi retrospektif, bukan sumber tuning. [Source: `Results/Reward_Validation/reward_confirmation_decision.json`]

## 6.2 DDPG versus TD3 versus SAC

DDPG dan TD3 menggunakan air lebih sedikit pada sejumlah seed, tetapi pencapaian targetnya juga lebih rendah dan lebih bervariasi. Hal tersebut menunjukkan bahwa irigasi minimum tidak otomatis berarti efisien; controller dapat terlihat hemat karena under-irrigation. SAC lebih stabil pada primary endpoint dalam evidence ini. Perbandingan mekanisme algoritmik tetap dibatasi pada protokol Virtual Garden yang dikunci. [Source: `Results/Confirmatory_10Seed/main_10seed_results_2025.csv`] [Source: `Results/Fair_DRL/fairness_audit.json`]

## 6.3 Partial observability

Formulasi POMDP relevan karena controller tidak mengamati seluruh state dan proses laten tanah serta tidak mengetahui forcing mendatang secara sempurna. History dan forecast menyediakan context tambahan, tetapi alasan formulasi ini hanya mendukung validitas kerangka; manfaat performanya tetap harus dibuktikan melalui ablation dan inferensi. [Source: `Docs/Reviewer_Alignment/scope_and_data_classification.md`] [Source: `Results/POMDP_Ablation/context_intervention_results.csv`]

## 6.4 Forecast contribution

Standalone forecast main effect sebesar -0.107 percentage points tidak didukung setelah koreksi exact-Holm. Oleh sebab itu, penelitian ini tidak mengklaim bahwa forecast proxy secara mandiri meningkatkan performa. Hasil robustness SF10–SF30 tetap berguna sebagai diagnostik sensitivitas terhadap error proxy. [Source: `Results/Confirmatory_10Seed/factorial_inference.csv`] [Source: `Results/POMDP_Ablation/forecast_robustness.csv`]

## 6.5 Memory contribution

Memory main effect sebesar 1.009 percentage points dengan 95% bootstrap CI [0.579, 1.438] dan exact-Holm p = 0.017578 didukung pada locked warm-start pipelines. Temuan ini konsisten dengan peran history dalam merangkum dinamika laten, tetapi belum membuktikan manfaat universal di luar simulator dan protokol ini. [Source: `Results/Confirmatory_10Seed/factorial_inference.csv`]

## 6.6 SACSI integration

SACSI mengintegrasikan current observation, representasi history, dan controlled synthetic forecast proxy. Dibanding SAC, peningkatan Time in Target adalah 0.902 percentage points dengan 95% bootstrap CI [0.336, 1.531] dan exact-Holm p = 0.046875. Hasil ini mendukung locked end-to-end pipeline, tetapi tidak memisahkan efek arsitektur dari tambahan adaptation budget. [Source: `Results/Confirmatory_10Seed/planned_contrasts.csv`] [Source: `Results/Confirmatory_10Seed/final_statistics_summary.json`]

## 6.7 Precision-farming implication

Kerangka ini menyediakan cara terstruktur untuk menggabungkan sensing saat ini, history, dan informasi prediktif dalam continuous irrigation control. Implikasinya adalah potensi desain controller yang lebih sadar context, bukan bukti kesiapan deployment. Integrasi sensor, aktuator, komunikasi, keselamatan, dan kalibrasi lokasi masih memerlukan validasi tersendiri. [Source: `Docs/Reviewer_Alignment/scope_and_data_classification.md`]

## 6.8 Limitations

Penelitian dibatasi oleh Virtual Garden untuk hortikultura generik, soil state hasil simulasi, satu setting meteorologi retrospektif, dan forecast terkontrol yang bukan archived as-issued operational forecast. Tidak ada field trial, pengukuran hasil panen, atau model daya pompa sehingga klaim efektivitas lapangan, yield, dan penghematan energi tidak dibuat. Selain itu, context variants menggunakan SAC anchor 6,720 interactions ditambah 6,720 adaptation interactions, sedangkan non-context algorithms menggunakan total 6,720 interactions. [Source: `Docs/Reviewer_Alignment/scope_and_data_classification.md`] [Source: `Results/Confirmatory_10Seed/final_statistics_summary.json`]

## 6.9 Future work

Pekerjaan berikutnya perlu menguji ulang controller dengan archived as-issued forecasts, multi-location meteorology and soil calibration, equal-total-budget training, hardware-in-the-loop testing, serta field trials dengan sensor dan aktuator nyata. Model daya pompa yang terkalibrasi diperlukan sebelum energy outcome dapat dilaporkan. [Source: `Docs/Reviewer_Alignment/scope_and_data_classification.md`] [Source: `Results/Confirmatory_10Seed/final_statistics_summary.json`]
