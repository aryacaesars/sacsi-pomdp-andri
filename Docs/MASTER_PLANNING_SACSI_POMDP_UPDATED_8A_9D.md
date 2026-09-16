# MASTER PLANNING PROJECT SACSI-POMDP

## Pengembangan Model Soft Actor-Critic for Smart Irrigation dalam Kerangka POMDP untuk Irigasi Cerdas Adaptif

**Tujuan dokumen**  
Dokumen ini menjadi _master execution plan_ untuk mengerjakan ulang seluruh project SACSI-POMDP secara mandiri, terstruktur, reproducible, dan konsisten dengan seluruh keputusan metodologis yang sudah dibahas. Seluruh implementasi disusun modular agar setiap controller dapat diuji, dibandingkan, diganti, atau dikembangkan tanpa mengubah Virtual Garden Core.

**Status revisi master plan**  
Dokumen ini telah diperbarui untuk memasukkan hasil alignment reviewer dan seluruh modul lanjutan **8A–8H serta 9A–9D**. Bagian lanjutan ini menambahkan reward validation, simple-case testing, DDPG, TD3, fair DRL benchmark, incremental POMDP contribution, 10-seed confirmatory benchmark, dashboard evidence integration, dissertation integration, defense package, dan final reproducibility freeze.

**Prinsip revisi**  
Pekerjaan Modul 1–7 tetap dipertahankan. Modul 8–9 berfungsi sebagai _reviewer-alignment extension_ dan _finalization layer_, bukan pembangunan ulang project dari nol.

---

# 1. RINGKASAN PROYEK

## 1.1 Tujuan utama penelitian
Mengembangkan dan mengevaluasi controller irigasi cerdas berbasis **Soft Actor-Critic (SAC)** dalam kerangka **Partially Observable Markov Decision Process (POMDP)** dengan integrasi:

1. current-state observation;
2. temporal memory melalui LSTM;
3. short-term weather forecast;
4. Virtual Garden berbasis neraca air;
5. benchmark terhadap controller non-RL dan varian RL;
6. analisis statistik multi-seed;
7. dashboard terpadu untuk visualisasi dan evaluasi;
8. benchmark continuous-control yang fair terhadap **DDPG, TD3, dan SAC**;
9. validasi reward multi-objective, simple-case testing, dan raw-data episode validation;
10. reviewer-aligned confirmatory analysis dan reproducibility freeze.

Model final disebut **SACSI-POMDP / SACSI Full**.

---

# 2. PRINSIP METODOLOGIS YANG TIDAK BOLEH DIUBAH

## 2.1 Prinsip modularitas
`VirtualGardenCore` harus independen dari controller.

Controller hanya menerima observasi dan mengeluarkan aksi irigasi. Controller tidak boleh mengubah parameter fisika Virtual Garden.

```text
Dataset
   ↓
Virtual Garden Core
   ↓ observation
Controller Interface
   ↓ action
Virtual Garden Core
   ↓ next state
Metrics / Logger / Dashboard
```

## 2.2 Urutan pengembangan
Urutan pengembangan final setelah reviewer alignment:

1. validasi lingkungan fisik;
2. baseline non-RL;
3. SAC Basic;
4. SAC + Forecast;
5. SAC + LSTM;
6. SACSI Full;
7. expanded multi-seed;
8. sinkronisasi rumusan masalah, tujuan, hipotesis, dan scope;
9. reward formalization, ablation, dan weight sensitivity;
10. simple-case testing dan raw-data episode validation;
11. DDPG baseline;
12. TD3 baseline;
13. fair DRL benchmark DDPG–TD3–SAC;
14. incremental POMDP contribution SAC → Forecast → Memory → SACSI;
15. final 10-seed confirmatory benchmark dan statistics;
16. final dashboard evidence integration;
17. dissertation results & discussion integration;
18. reviewer-response dan defense package;
19. reproducibility, repository freeze, dan publication package.

Urutan reviewer-aligned utama:

```text
Virtual Garden
    ↓
Reward + Simple Case
    ↓
DDPG → TD3 → SAC
    ↓
SAC + Forecast / SAC + LSTM
    ↓
SACSI-POMDP
    ↓
10-Seed Confirmatory Benchmark
    ↓
Statistics + Dashboard + Dissertation
    ↓
Repository Freeze / Publication Release
```

## 2.3 Fairness lock
Untuk perbandingan antar-controller, komponen berikut harus sama:

- Virtual Garden Core;
- parameter tanah dan tanaman;
- initial soil moisture;
- target soil-moisture band;
- action bounds;
- reward definition yang sama untuk **DDPG, TD3, SAC, dan SACSI** pada main benchmark;
- train / validation / benchmark split;
- matched seeds;
- metric engine;
- evaluation period;
- comparable network capacity;
- comparable environment-interaction budget;
- checkpoint-selection rule;
- logging schema.

Fairness **tidak berarti** mekanisme internal algoritma harus dibuat identik. OU/noise exploration pada DDPG, target-policy smoothing dan delayed update pada TD3, serta entropy tuning pada SAC tetap dipertahankan karena merupakan karakter intrinsik algoritmanya.

Performa tidak boleh “ditingkatkan” dengan mengubah environment hanya untuk model tertentu.

---

# 3. KEPUTUSAN DATASET FINAL

## 3.1 Dataset utama
Gunakan **dataset yang sudah dimiliki** sebagai sumber utama penelitian.

Periode:

```text
2021–2025
```

Pembagian temporal:

```text
Training   : 2021–2023
Validation : 2024
Benchmark  : 2025
```

## 3.2 Status koordinat
Koordinat **tidak digunakan sebagai bagian analisis**.

Jika metadata spreadsheet menampilkan tanda latitude yang salah, abaikan metadata koordinat tersebut karena dataset cuaca yang digunakan sudah dianggap dataset penelitian final.

Tidak perlu mengunduh ulang dataset hanya untuk memperbaiki tanda koordinat.

## 3.3 External validation 2026
External temporal validation 2026 **tidak menjadi syarat utama penyelesaian disertasi**.

Jika nanti dilakukan, statusnya adalah penelitian tambahan / future work.

## 3.4 Status benchmark 2025
Karena periode 2025 sudah beberapa kali digunakan pada diagnostic dan benchmark selama pengembangan model, dalam naskah akhir lebih aman disebut:

> **retrospective final benchmark 2025**

bukan pristine independent holdout.

---

# 4. INFRASTRUKTUR YANG DIREKOMENDASIKAN

## 4.1 Platform utama
Rekomendasi:

```text
Google Colab Pro / Pro+
+ Google Drive
+ GitHub
+ Streamlit Community Cloud
+ RunPod opsional
```

## 4.2 Pembagian fungsi platform

| Kebutuhan | Platform |
|---|---|
| Coding utama | Google Colab |
| Dataset dan checkpoint | Google Drive |
| Version control | GitHub |
| Training ringan | Colab GPU |
| Training multi-seed/recurrent berat | Colab Pro+ / RunPod |
| Statistik | Colab CPU |
| Dashboard | Streamlit Community Cloud |
| Backup GPU | Kaggle Notebook |

## 4.3 Struktur folder Google Drive

```text
SACSI_Dissertation/
│
├── 00_Dataset/
├── 01_Virtual_Garden/
├── 02_Baseline_Controller/
├── 03_SAC_Basic/
├── 04_SAC_Forecast/
├── 05_SAC_LSTM/
├── 06_SACSI_Full/
├── 07_Final_Experiment/
├── 08_Statistics/
├── 09_Dashboard/
├── 10_Dissertation/
│
├── Checkpoints/
├── Results/
├── Figures/
├── Tables/
└── Logs/
```

## 4.4 Struktur repository GitHub

```text
SACSI-POMDP/
├── src/
│   ├── virtual_garden/
│   ├── controllers/
│   ├── sac_basic/
│   ├── sac_forecast/
│   ├── sac_lstm/
│   ├── sacsi_full/
│   ├── evaluation/
│   └── statistics/
├── notebooks/
├── configs/
├── tests/
├── dashboard/
├── scripts/
└── README.md
```

---

# 5. KONFIGURASI VIRTUAL GARDEN YANG DIKUNCI

Parameter utama yang telah digunakan:

| Parameter | Nilai |
|---|---:|
| Initial theta | 0.27 m³/m³ |
| Wilting point | 0.15 |
| Target minimum | 0.22 |
| Target maximum | 0.32 |
| Field capacity | 0.30–0.35 sesuai config final |
| Saturation | 0.45 |
| Root depth | 300 mm |
| Crop coefficient | 0.85 |
| Maximum irrigation | 5 mm/hour |

> Gunakan satu config final dan jangan mengubahnya antar-controller.

## 5.1 Neraca air
Virtual Garden harus mempertahankan mass balance secara numerik.

Target:

```text
max_abs_mass_balance_error <= 1e-8 mm
```

Idealnya sekitar machine precision.

---

# 6. METRIK UTAMA PENELITIAN

## Primary endpoint

```text
Time in Target (%)
```

Target band:

```text
0.22 <= theta <= 0.32
```

## Secondary metrics

1. total irrigation water (mm);
2. violation rate (%);
3. deficit rate (%);
4. surplus rate (%);
5. RMSE terhadap target band;
6. mean soil moisture;
7. runoff;
8. drainage;
9. action smoothness;
10. cumulative reward;
11. decision latency;
12. water-use efficiency.

## Water Use Efficiency

```text
WUE = target_hours / total_irrigation
```

Digunakan sebagai metrik tambahan, bukan satu-satunya dasar ranking.

---

# 7. MASTER SPRINT PLAN

Rencana awal terdiri dari **16 sprint fondasi** untuk Modul 1–7. Setelah reviewer alignment, sprint fondasi tersebut dipertahankan dan dilanjutkan oleh **12 modul ekstensi: 8A–8H dan 9A–9D**. Detail ekstensi final terdapat pada Bagian 29–31.

Estimasi durasi bersifat fleksibel. Jika sudah menguasai Python/RL, beberapa sprint dapat digabung.

---

# SPRINT 0 — PROJECT SETUP & REPRODUCIBILITY

## Tujuan
Menyiapkan workspace sehingga project dapat dikerjakan ulang tanpa kehilangan file, checkpoint, atau konfigurasi.

## Task

- [ ] Buat folder Google Drive `SACSI_Dissertation`.
- [ ] Buat repository GitHub.
- [ ] Buat `requirements.txt`.
- [ ] Tentukan versi Python.
- [ ] Install:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - plotly
  - torch
  - pytest
  - statsmodels
  - pingouin (opsional)
  - streamlit
- [ ] Buat global random seed utility.
- [ ] Buat config YAML/JSON untuk semua parameter.
- [ ] Mount Google Drive di Colab.
- [ ] Buat mekanisme auto-save checkpoint.

## Output

```text
requirements.txt
README.md
configs/global_config.json
notebooks/00_setup.ipynb
```

## Acceptance criteria

- environment dapat dibuat ulang;
- import seluruh package berhasil;
- GitHub dan Drive sinkron;
- checkpoint tidak disimpan hanya di `/content/`.

---

# SPRINT 1 — DATA PIPELINE & DATA AUDIT

## Tujuan
Membangun data pipeline final 2021–2025.

## Task

- [ ] Load Historical Weather dataset.
- [ ] Load Historical Forecast dataset jika digunakan.
- [ ] Parse timestamp.
- [ ] Audit missing values.
- [ ] Audit duplicate timestamps.
- [ ] Audit hourly continuity.
- [ ] Audit unit setiap variabel.
- [ ] Pisahkan:
  - train 2021–2023;
  - validation 2024;
  - benchmark 2025.
- [ ] Buat scaler hanya dari training split.
- [ ] Simpan normalizer.

## Variabel aktual yang digunakan

- temperature;
- relative humidity;
- precipitation;
- rain;
- ET0;
- VPD;
- shortwave radiation;
- optional wind/cloud features jika dibutuhkan pipeline.

## Output

```text
data_clean.csv
train_2021_2023.csv
validation_2024.csv
benchmark_2025.csv
normalizer.json
data_audit_report.csv
```

## Acceptance criteria

- tidak ada future leakage;
- scaler tidak fit pada validation/test;
- panjang periode tercatat;
- timestamp konsisten.

---

# SPRINT 2 — VIRTUAL GARDEN CORE

## Tujuan
Membangun simulator fisik independen controller.

## Subtahap ekuivalen

```text
1A Blueprint
1B Implementation
1C Validation
```

## Task

- [ ] Buat `VirtualGardenConfig`.
- [ ] Buat `VirtualGardenCore`.
- [ ] Implementasikan:
  - precipitation input;
  - irrigation input;
  - infiltration;
  - evapotranspiration;
  - drainage;
  - runoff;
  - overflow;
  - root-zone storage;
  - theta update.
- [ ] Buat unit test neraca massa.
- [ ] Uji beberapa skenario ekstrem.

## Minimal validation scenarios

1. tanpa hujan dan tanpa irigasi;
2. hujan tinggi;
3. irigasi tinggi;
4. tanah mendekati saturation;
5. periode kering panjang.

## Acceptance criteria

```text
mass balance residual <= 1e-8 mm
```

Semua test harus lulus.

---

# SPRINT 3 — CONTROLLER INTERFACE & NON-RL BASELINES

## Tujuan
Menciptakan benchmark sebelum menggunakan RL.

## Controller

1. No Irrigation
2. Fixed Schedule
3. Threshold-Based
4. Rule-Based Forecast-Aware
5. Fuzzy Controller

## Interface minimum

```python
reset()
select_action(observation)
```

## Task

- [ ] Buat base controller interface.
- [ ] Implementasikan lima baseline.
- [ ] Buat common logger.
- [ ] Buat common metrics.
- [ ] Benchmark train / validation / 2025.

## Benchmark penting yang pernah diperoleh
Sebagai sanity reference, bukan angka yang harus dipaksakan identik:

```text
Rule-Based Forecast-Aware sekitar 270 mm dan ~65.36% time in target
Threshold-Based sekitar 276 mm dan ~65.02%
Fuzzy sekitar 297 mm dan ~61.97%
```

Jika implementasi mandiri berbeda sedikit, audit dahulu pipeline sebelum menganggap salah.

## Acceptance criteria

- semua controller menggunakan environment identik;
- logger schema sama;
- ranking dapat direproduksi.

---

# SPRINT 4 — UNIFIED DASHBOARD V1

## Tujuan
Membangun monitoring sebelum RL.

## Fitur

- single controller selector;
- multi-controller compare;
- soil moisture plot;
- target band;
- precipitation;
- irrigation;
- metrics table;
- export CSV/XLSX/JSON;
- custom simulation window.

## Teknologi

```text
Streamlit + Plotly
```

## Acceptance criteria

Dashboard dapat menampilkan minimal 5 baseline non-RL.

---

# SPRINT 5 — SAC BASIC

## Tujuan
Membangun baseline RL pertama tanpa forecast dan tanpa LSTM.

## Observation 8-D

1. soil moisture;
2. precipitation actual;
3. temperature actual;
4. RH actual;
5. ET0 actual;
6. VPD actual;
7. shortwave radiation;
8. previous irrigation.

## Arsitektur

- stochastic actor;
- twin critics;
- target critics;
- replay buffer;
- entropy auto-tuning;
- no LSTM;
- no forecast.

## Initial reference hyperparameters

```text
hidden_dim = 64
batch_size = 64
warmup = 500
actor_lr = 5e-4
critic_lr = 5e-4
alpha_lr = 1e-4
gamma = 0.99
tau = 0.005
initial_alpha = 0.05
actor_mean_bias = -1.5
```

## Episode length

```text
336 hours
```

## Seeds awal

```text
11, 22, 33
```

## Convergence gate validation 2024

- Time in target >= 50%
- Violation <= 50%
- Total irrigation <= 750 mm
- Mean theta dalam target band
- Deficit <= 20%
- finite loss/metrics
- mass balance valid

## Acceptance criteria

3/3 seed awal diupayakan lulus.

Jika seed gagal, jangan memakai 2025 untuk tuning.

---

# SPRINT 6 — SAC + FORECAST

## Tujuan
Mengukur kontribusi forecast tanpa LSTM.

## Observation
Current 8-D + forecast context.

Desain awal pernah menggunakan 17-D dengan h1–h3.

### Catatan metodologis penting
Historical Forecast continuous series bukan otomatis true as-issued lead-time archive.

Jangan melakukan simple shift lalu mengklaim operational forecast.

Gunakan protokol eksplisit:

```text
Synthetic Forecast SF-20
```

sebagai controlled proxy jika true forecast archive tidak tersedia.

## Robustness forecast

```text
SF10
SF20
SF30
```

## Horizon
Hasil diagnostik sebelumnya menunjukkan h1 lebih menjanjikan daripada h1–h3.

Untuk eksperimen ulang:

- primary configuration harus ditentukan dari validation;
- jangan memilih horizon berdasarkan benchmark 2025.

## Acceptance criteria

- forecast branch benar-benar mengubah action;
- no look-ahead leakage;
- robustness tidak collapse.

---

# SPRINT 7 — SAC + LSTM

## Tujuan
Menguji temporal memory tanpa forecast.

## Sequence

```text
X_t = [o_(t-23), ..., o_t]
shape = 24 x 8
```

## Arsitektur

```text
LSTM
hidden = 64
1 layer
```

## Replay buffer
Harus sequence-aware.

Tidak boleh sequence menyeberang episode boundary.

## Diagnostic wajib

1. current-only;
2. reverse history;
3. shuffle history;
4. zero history;
5. sequence length sensitivity.

## Sequence sensitivity

```text
k = 6, 12, 24, 48
```

## Catatan dari eksperimen sebelumnya
Training recurrent SAC dari random initialization pernah tidak stabil.

Solusi stabil yang ditemukan:

```text
Residual Recurrent Warm-Start (RRWS)
```

Current-state SAC branch dipertahankan, sedangkan recurrent branch ditambahkan sebagai residual.

### Caveat penting
Jika residual LSTM masih tepat nol, model belum boleh diklaim mendapatkan manfaat memory.

## Acceptance criteria

- recurrent pipeline stable;
- LSTM gradient finite;
- history intervention mengubah action jika ingin mengklaim memory active.

---

# SPRINT 8 — SACSI FULL BLUEPRINT & IMPLEMENTATION

## Tujuan
Menggabungkan current state + forecast + LSTM dalam POMDP.

## Konfigurasi final konseptual

```text
Current State 8-D
+ History 24 x 8
+ Forecast h1 3-D
```

Forecast h1:

1. predicted precipitation;
2. predicted ET0;
3. predicted temperature.

## POMDP representation

```text
z_t = LSTM(o_t-23 ... o_t)

c_t = concat(current_state, z_t, forecast_context)

a_t ~ pi(. | c_t)
```

## Fusion architecture

```text
Current encoder
History LSTM encoder
Forecast encoder
       ↓
Fusion MLP
       ↓
Actor + Twin Critics
```

## Primary design

- current encoder: 8 → 64;
- LSTM hidden: 64;
- forecast encoder: 3 → 32;
- fusion MLP: 160 → 128 → 64;
- action: 0–5 mm/hour.

## Training strategy

### Phase 0
Warm-start dari converged SAC Basic.

### Phase 1
Context branch netral / kecil.

### Phase 2
Context-only learning.

### Phase 3
Limited joint fine-tuning.

### Phase 4
Validation 2024 checkpoint selection.

### Phase 5
Retrospective benchmark 2025.

## Gate tambahan
Selain convergence gate, cek:

- context residual norm > 0;
- forecast intervention changes action;
- history intervention changes action.

## Acceptance criteria

Technical validity dan context activation harus dipisahkan dari performance superiority.

---

# SPRINT 9 — SACSI MULTI-SEED TRAINING

## Tujuan
Memperluas hasil dari 3 seed menjadi 10 matched seeds.

## Seed final

```text
11
22
33
44
55
66
77
88
99
110
```

## Empat varian RL

```text
SAC Basic
SAC + Forecast
SAC + LSTM
SACSI Full
```

Total checkpoint:

```text
10 seeds x 4 models = 40 checkpoints
```

## Prinsip confirmatory

- seed gagal tidak boleh dibuang;
- seed gagal tidak diganti;
- budget training harus konsisten;
- jangan memberikan training lebih panjang hanya karena seed tertentu jelek;
- 2025 tidak digunakan untuk checkpoint selection.

## Acceptance criteria

Registry lengkap 40 model dibuat.

Setiap row minimal memiliki:

```text
model
seed
validation_gate
validation metrics
benchmark_2025 metrics
checkpoint path
checkpoint hash
```

---

# SPRINT 10 — FINAL BENCHMARK 9 METHODS

## Metode

1. No Irrigation
2. Fixed Schedule
3. Threshold-Based
4. Rule-Based Forecast-Aware
5. Fuzzy Controller
6. SAC Basic
7. SAC + Forecast
8. SAC + LSTM
9. SACSI Full

## Benchmark common support

Gunakan periode identik 2025.

## Formal metrics

- water;
- target occupancy;
- violation;
- RMSE;
- smoothness;
- deficit;
- surplus;
- drainage;
- runoff.

## Catatan hasil sebelumnya
Strong baseline yang muncul adalah:

```text
Rule-Based Forecast-Aware
```

SACSI Full belum terbukti menjadi metode terbaik secara target occupancy + water efficiency.

Ini harus diterima sebagai hasil ilmiah, bukan dianggap kegagalan implementasi.

---

# SPRINT 11 — ABLATION & ROBUSTNESS

## 11.1 Factorial RL family

| Model | Forecast | Memory |
|---|---:|---:|
| SAC Basic | OFF | OFF |
| SAC + Forecast | ON | OFF |
| SAC + LSTM | OFF | ON |
| SACSI Full | ON | ON |

## 11.2 Context ablation SACSI

- Full
- No History
- No Forecast
- No Context
- Shuffled History
- Reversed History
- Zero History
- Shuffled Forecast
- Zero Forecast

## 11.3 Forecast robustness

```text
SF10
SF20
SF30
```

## 11.4 Sequence sensitivity

```text
6
12
24
48 hours
```

## 11.5 Interaction / synergy

Descriptive interaction:

```text
Interaction = SACSI - Forecast - LSTM + Basic
```

Interpretasi harus mempertimbangkan arah metrik.

Untuk Time in Target, nilai positif = menguntungkan.

Untuk water, RMSE, violation, nilai negatif = menguntungkan.

---

# SPRINT 12 — STATISTICAL ANALYSIS FINAL

## Tujuan
Mengubah 10 matched seeds menjadi bukti statistik yang sah.

## Primary endpoint

```text
Time in Target (%)
```

## Statistical design

### Repeated-measures factorial 2 × 2

Factors:

```text
Forecast OFF / ON
Memory OFF / ON
```

Subject/block:

```text
Seed
```

## Model families

```text
Basic       = F0 M0
Forecast    = F1 M0
LSTM        = F0 M1
SACSI Full  = F1 M1
```

## Output statistik

- Forecast main effect;
- Memory main effect;
- Forecast × Memory interaction;
- F statistic;
- p-value;
- partial eta-squared;
- confidence interval.

## Pairwise comparisons

1. SACSI vs SAC Basic
2. SACSI vs SAC + Forecast
3. SACSI vs SAC + LSTM

Gunakan:

- paired test;
- Holm correction;
- Cohen's dz;
- 95% bootstrap CI.

## Nonparametric / robust confirmation

Exact paired sign-flip permutation test dapat digunakan sebagai robustness check.

## Bootstrap

```text
20,000 resamples
95% CI
```

## Deterministic baseline warning
Rule-Based, Threshold, Fuzzy, dan Fixed Schedule tidak memiliki stochastic training seed.

Jangan membuat “fake seeds” untuk baseline deterministik.

Perbandingan dengan baseline deterministik dilaporkan sebagai trajectory/reference performance, bukan repeated-measures seed analysis palsu.

---

# SPRINT 13 — UNIFIED DASHBOARD FINAL

## Tujuan
Mengintegrasikan 9 metode.

## Fitur final

- 9-method registry;
- selector tunggal;
- multi-selector 2–4 methods;
- metrics table;
- soil moisture plot;
- target-band plot;
- irrigation plot;
- rain actual / forecast;
- cumulative water;
- robustness selector;
- ablation selector;
- seed selector untuk RL;
- export CSV;
- export XLSX;
- export JSON;
- export PNG;
- export experiment ZIP.

## Deployment

```text
GitHub → Streamlit Community Cloud
```

Dashboard tidak disarankan di-host permanen melalui Colab.

---

# SPRINT 14 — BAB HASIL & PEMBAHASAN

## Tujuan
Mengubah output eksperimen menjadi Bab Hasil yang konsisten.

## Struktur yang direkomendasikan

### 14.1 Validasi Virtual Garden

- physics;
- mass balance;
- unit test;
- environmental scenarios.

### 14.2 Benchmark non-RL

- lima controller;
- strong baseline selection.

### 14.3 SAC Basic

- training convergence;
- validation;
- benchmark.

### 14.4 SAC + Forecast

- training;
- forecast diagnostics;
- robustness.

### 14.5 SAC + LSTM

- recurrent stability;
- memory diagnostic;
- caveat jika memory effect kecil.

### 14.6 SACSI Full

- POMDP formulation;
- architecture;
- convergence;
- context activation.

### 14.7 Final 9-method benchmark

- water;
- target occupancy;
- violation;
- RMSE;
- smoothness.

### 14.8 Ablation

- Basic;
- Forecast;
- LSTM;
- Full.

### 14.9 Robustness

- forecast error;
- sequence length.

### 14.10 Statistical analysis

- 2×2 repeated measures;
- pairwise;
- effect size;
- CI.

### 14.11 Discussion

Bandingkan dengan:

- problem formulation;
- POMDP rationale;
- adaptive control;
- forecast-aware irrigation;
- RL irrigation literature;
- recurrent RL.

---

# SPRINT 15 — KESIMPULAN, NOVELTY, LIMITATIONS & FINAL DISSERTATION

## 15.1 Kesimpulan
Kesimpulan harus menjawab rumusan masalah.

Jangan menyatakan SACSI superior jika data tidak mendukung.

## 15.2 Novelty yang aman
Novelty utama lebih kuat jika ditulis sebagai:

> integrasi modular SAC dalam kerangka POMDP untuk smart irrigation dengan kombinasi current-state information, temporal memory, forecast context, validated Virtual Garden, controlled ablation, dan multi-seed evaluation.

Bukan:

> SACSI pasti paling unggul dibanding semua metode.

## 15.3 Klaim yang boleh dibuat jika hasil tetap seperti eksperimen terdahulu

- SACSI berhasil diimplementasikan;
- POMDP representation valid;
- current + history + forecast dapat diintegrasikan;
- context branch aktif;
- SACSI robust terhadap forecast perturbation;
- framework modular dan reproducible;
- SAC/SACSI menghasilkan action yang sangat smooth.

## 15.4 Klaim yang tidak boleh dibuat tanpa bukti tambahan

- SACSI signifikan lebih baik dari seluruh controller;
- LSTM selalu meningkatkan performa;
- forecast selalu meningkatkan performa;
- Forecast × Memory synergy signifikan tanpa statistical evidence.

## 15.5 Keterbatasan

1. Synthetic forecast proxy, bukan archived operational forecast murni.
2. Dataset berasal dari satu rangkaian time-series lingkungan.
3. Virtual Garden tetap model simulasi.
4. Benchmark 2025 bersifat retrospective benchmark.
5. Model RL sensitif terhadap random seed.
6. LSTM context contribution dapat kecil pada beberapa checkpoint.
7. Strong rule-based controller dapat tetap lebih efisien.
8. External 2026 belum diwajibkan.

## 15.6 Future work

- true archived forecast runs;
- field validation;
- IoT soil-moisture observations;
- multi-location datasets;
- attention/gated fusion;
- hierarchical RL;
- offline RL;
- model-based RL;
- multi-objective Pareto irrigation control;
- domain randomization;
- uncertainty-aware policy.

---

# 8. MASTER ACCEPTANCE GATES

## Gate A — Physics

```text
Virtual Garden valid
mass balance <= 1e-8
```

## Gate B — Baseline

```text
5 baseline controllers runnable
common metrics
common environment
```

## Gate C — SAC Basic

```text
finite training
convergence validation
checkpoint saved
```

## Gate D — Forecast

```text
no leakage
forecast affects policy
robustness evaluated
```

## Gate E — LSTM

```text
sequence buffer valid
recurrent gradient valid
history diagnostics available
```

## Gate F — SACSI

```text
current + LSTM + forecast work jointly
context branch nonzero
3-seed initial validation completed
```

## Gate G — Multi-seed

```text
40 checkpoints
10 matched seeds
seed failures preserved
```

## Gate H — Statistics

```text
master seed table complete
paired design valid
Holm correction
CI and effect size
```

## Gate I — Dissertation

```text
all figures generated from source data
all tables traceable
claims <= evidence
reproducibility package complete
```

---

# 9. MASTER RESULT TABLE YANG WAJIB DIMILIKI

Buat satu file source-of-truth:

```text
master_results.csv
```

Minimal kolom final:

```text
experiment_id
module
algorithm_family
model
seed
split
period_start
period_end
reward_version
virtual_garden_version
observation_version
forecast_enabled
memory_enabled
forecast_protocol
forecast_error
forecast_horizon
sequence_length
checkpoint
checkpoint_hash
validation_gate
steps
environment_interactions
training_budget_version
total_water_mm
time_in_target_pct
violation_rate_pct
deficit_rate_pct
surplus_rate_pct
rmse_band
action_smoothness
mean_soil_moisture
runoff_total_mm
drainage_total_mm
reward_mean
mass_balance_error
git_commit
result_status
```

`algorithm_family` minimal menggunakan:

```text
NON_RL
DDPG
TD3
SAC
SAC_FORECAST
SAC_LSTM
SACSI_POMDP
```

`result_status` minimal:

```text
EXPLORATORY
VALIDATION
CONFIRMATORY
FROZEN
```

Semua grafik, tabel, dashboard, dan angka Bab Hasil final harus dibuat dari `master_results.csv`, frozen summary, atau sumber log yang dapat ditelusuri kembali ke registry/hash artefak.

---

# 10. FIGURES FINAL YANG HARUS DIBUAT

## Figure 1
Research framework SACSI-POMDP.

## Figure 2
Virtual Garden water-balance diagram.

## Figure 3
Controller benchmark architecture.

## Figure 4
SAC Basic architecture.

## Figure 5
SAC + Forecast architecture.

## Figure 6
SAC + LSTM architecture.

## Figure 7
SACSI Full architecture.

## Figure 8
Training convergence per model.

## Figure 9
Validation target occupancy per seed.

## Figure 10
Final benchmark 9 methods.

## Figure 11
Water vs Time-in-Target trade-off.

## Figure 12
Soil moisture trajectories.

## Figure 13
Irrigation vs precipitation.

## Figure 14
Forecast-context ablation.

## Figure 15
Memory ablation.

## Figure 16
Forecast robustness SF10–SF30.

## Figure 17
Sequence sensitivity k=6–48.

## Figure 18
Forecast × Memory factorial interaction.

## Figure 19
10-seed distribution boxplot.

## Figure 20
Unified Dashboard final.

## Figure 21
Reward ablation and weight-sensitivity Pareto plot.

## Figure 22
Simple-case validation trajectories.

## Figure 23
Raw-data DRY/WET/MIXED episode comparison.

## Figure 24
DDPG architecture and training convergence.

## Figure 25
TD3 architecture and training convergence.

## Figure 26
Fair DDPG–TD3–SAC benchmark.

## Figure 27
Main 10-seed DDPG–TD3–SAC–SACSI distribution.

## Figure 28
Planned contrast effect-size / confidence-interval plot.

## Figure 29
Reviewer evidence architecture.

## Figure 30
Final freeze / reproducibility pipeline.

---

# 11. TABLES FINAL YANG HARUS DIBUAT

## Table 1
Dataset variables and units.

## Table 2
Virtual Garden parameters.

## Table 3
Controller definitions.

## Table 4
SAC Basic hyperparameters.

## Table 5
SAC + Forecast architecture.

## Table 6
SAC + LSTM architecture.

## Table 7
SACSI Full architecture.

## Table 8
Convergence gate definition.

## Table 9
Validation result per seed.

## Table 10
Final benchmark 9 methods.

## Table 11
10-seed RL family aggregate.

## Table 12
Forecast × Memory factorial effects.

## Table 13
Post-hoc pairwise results.

## Table 14
Effect size and 95% CI.

## Table 15
Forecast robustness.

## Table 16
Sequence sensitivity.

## Table 17
Ablation results.

## Table 18
Hypothesis decision H1–H4.

## Table 19
Reviewer comment → module → evidence mapping.

## Table 20
Reward ablation and weight-sensitivity results.

## Table 21
Simple-case and raw-data episode validation.

## Table 22
DDPG/TD3/SAC fairness configuration.

## Table 23
DDPG–TD3–SAC validation and benchmark results.

## Table 24
Final 10-seed DDPG–TD3–SAC–SACSI results.

## Table 25
Friedman and planned paired contrasts.

## Table 26
Factorial Forecast × Memory inferential results.

## Table 27
Claim-to-evidence release matrix.

## Table 28
Final checkpoint/result/hash registry summary.

---

# 12. QUALITY CONTROL CHECKLIST

## Data

- [ ] Tidak ada data leakage.
- [ ] Scaler fit pada training saja.
- [ ] Timestamp konsisten.
- [ ] Split tahun benar.

## Environment

- [ ] Mass balance valid.
- [ ] Parameter fixed.
- [ ] Initial theta fixed.

## Controller

- [ ] Action 0–5 mm/h.
- [ ] Same controller interface.
- [ ] No controller edits environment.

## RL

- [ ] Seed explicit.
- [ ] Checkpoint metadata lengkap.
- [ ] Validation-based selection.
- [ ] No test-based retuning.

## Forecast

- [ ] Forecast protocol documented.
- [ ] Synthetic proxy diberi label jelas.
- [ ] No look-ahead leakage.

## LSTM

- [ ] Sequence causal.
- [ ] No episode crossing.
- [ ] Memory activation test.

## Statistics

- [ ] 10 matched seeds.
- [ ] Failed seeds retained.
- [ ] Paired design.
- [ ] Holm correction.
- [ ] Effect size.
- [ ] Confidence interval.
- [ ] No fake deterministic seeds.

## Writing

- [ ] No superiority claim without evidence.
- [ ] Separate technical validity vs performance superiority.
- [ ] Limitations explicit.
- [ ] All figures traceable to data.

---

# 13. RISIKO DAN MITIGASI

## Risiko 1 — Colab disconnect

Mitigasi:

- save checkpoint setiap validation cycle;
- simpan di Drive;
- log training ke CSV.

## Risiko 2 — Recurrent training unstable

Mitigasi:

- RRWS;
- gradient clipping;
- warm-start SAC Basic;
- conservative learning rate;
- context-only phase.

## Risiko 3 — Forecast tidak memberi benefit

Mitigasi:

- tetap laporkan hasil;
- robustness analysis;
- interpretasi bahwa information availability ≠ guaranteed performance gain.

## Risiko 4 — LSTM tidak aktif

Mitigasi:

- intervention diagnostics;
- gradient audit;
- residual norm;
- jangan klaim memory benefit jika tidak terbukti.

## Risiko 5 — Seed sensitivity tinggi

Mitigasi:

- 10 matched seeds;
- jangan drop failed seeds;
- report distribution, bukan mean saja.

## Risiko 6 — Rule-Based lebih baik

Mitigasi:

Jangan mengubah baseline atau environment.

Diskusikan bahwa:

- simple domain-informed controllers dapat sangat kuat;
- RL menawarkan adaptability/general framework;
- superiority bukan satu-satunya sumber novelty.

---

# 14. PRIORITAS JIKA WAKTU TERBATAS

Karena Modul 1–7 sudah selesai, prioritas saat ini bergeser ke reviewer-alignment dan confirmatory evidence.

## Must Have — Final Dissertation

1. 8A reviewer alignment locked.
2. 8B reward decision locked.
3. 8C simple-case + raw-data episode validation.
4. 8D DDPG implementation valid.
5. 8E TD3 implementation valid.
6. 8F fair DDPG–TD3–SAC benchmark.
7. 8G POMDP ablation dan context diagnostics.
8. 8H 10 matched seeds confirmatory benchmark.
9. final statistics + effect size + confidence interval.
10. 9A dashboard evidence integration.
11. 9B dissertation evidence integration.
12. 9C reviewer/defense package.
13. 9D reproducibility freeze.

## Should Have

- reward Pareto visualization;
- SF10/SF20/SF30 robustness;
- sequence sensitivity;
- reviewer evidence matrix di dashboard;
- automatic result reconciliation;
- SHA256 artefact manifest;
- publication package map.

## Nice to Have

- public Streamlit deployment;
- external temporal validation 2026;
- additional crop-specific calibration;
- constrained SAC / multi-objective RL extension;
- attention/gating variants;
- benchmark algoritma tambahan di luar DDPG/TD3/SAC.

Jika waktu terbatas, **jangan menambah algoritma baru** sebelum 8H dan final freeze selesai.

---

# 15. RECOMMENDED EXECUTION ORDER DI GOOGLE COLAB

Notebook existing Modul 1–7 tetap dipertahankan. Tambahkan notebook reviewer-alignment berikut:

```text
00_Project_Setup.ipynb
01_Data_Audit.ipynb
02_Virtual_Garden.ipynb
03_NonRL_Baselines.ipynb
04_Dashboard_Baseline.ipynb
05_SAC_Basic.ipynb
06_SAC_Forecast.ipynb
07_SAC_LSTM.ipynb
08_SACSI_Full.ipynb
09_Expanded_10Seed_Training.ipynb
10_Final_Benchmark_Existing.ipynb
11_Ablation_Robustness_Existing.ipynb
12_Statistical_Analysis_Existing.ipynb

13_8A_Reviewer_Alignment.ipynb
14_8B_Reward_Ablation_Sensitivity.ipynb
15_8C_Simple_Case_Raw_Episodes.ipynb
16_8D_DDPG.ipynb
17_8E_TD3.ipynb
18_8F_Fair_DRL_Benchmark.ipynb
19_8G_POMDP_Ablation.ipynb
20_8H_Confirmatory_10Seed.ipynb
21_8H_Final_Statistics.ipynb
22_9A_Dashboard_Integration.ipynb
23_9B_Result_Reconciliation.ipynb
24_9D_Release_Verification.ipynb
```

Modul 9B/9C lebih banyak menggunakan dokumen/CSV evidence dibanding GPU notebook.

## 15.1 Execution rule
Sebelum notebook training:

```text
1. git pull
2. mount Drive
3. verify dataset hash
4. verify common config
5. run unit tests
6. verify reward version
7. verify split
8. run experiment
9. save checkpoint + metadata
10. append result registry
```

---

# 16. RECOMMENDED SPRINT TIMELINE

## 16.1 Historical implementation Modul 1–7
Modul 1–7 sudah selesai dan dianggap sebagai fondasi existing.

## 16.2 Timeline lanjutan Modul 8–9

| Modul | Fokus | Estimasi kerja aktif | Compute dependence |
|---|---|---:|---|
| 8A | Reviewer alignment | 0.5–1 hari | rendah |
| 8B | Reward ablation/sensitivity | 2–4 hari | sedang |
| 8C | Simple/raw episode validation | 1–2 hari | rendah |
| 8D | DDPG | 2–4 hari | sedang |
| 8E | TD3 | 2–4 hari | sedang |
| 8F | Fair DDPG–TD3–SAC | 3–6 hari | tinggi |
| 8G | POMDP ablation/robustness | 3–6 hari | tinggi |
| 8H | 10-seed confirmatory + stats | 5–10 hari | sangat tinggi |
| 9A | Dashboard integration | 1–2 hari | rendah |
| 9B | Dissertation integration | 2–5 hari | rendah |
| 9C | Reviewer/defense package | 1–2 hari | rendah |
| 9D | Freeze/release | 1–2 hari | rendah |

Total lanjutan realistis:

```text
sekitar 3–6 minggu
```

Jika DDPG/TD3/SAC/SACSI 10-seed dapat diparalelkan pada GPU, waktu kalender dapat dipersingkat.

## 16.3 Critical path

```text
8B → 8C → 8D/8E → 8F → 8G → 8H → 9A → 9B → 9C → 9D
```

---

# 17. DEFINITION OF DONE PROJECT

Project dianggap selesai hanya jika seluruh fondasi Modul 1–7 **dan** reviewer-alignment extension Modul 8–9 selesai.

## Foundation done
- [ ] Dataset pipeline terdokumentasi.
- [ ] Virtual Garden tervalidasi.
- [ ] baseline non-RL selesai.
- [ ] SAC Basic selesai.
- [ ] SAC + Forecast selesai.
- [ ] SAC + LSTM selesai.
- [ ] SACSI Full selesai.

## Reviewer-alignment done
- [ ] 8A RM/T/H/scope terkunci.
- [ ] 8B reward decision terkunci.
- [ ] 8C simple/raw episode validation lulus.
- [ ] 8D DDPG implementation valid.
- [ ] 8E TD3 implementation valid.
- [ ] 8F fairness audit lulus.
- [ ] 8G factorial/context diagnostics selesai.
- [ ] 8H 10-seed main benchmark lengkap.
- [ ] 8H 10-seed SAC-family factorial lengkap.
- [ ] statistik confirmatory selesai.

## Integration done
- [ ] dashboard final membaca frozen evidence.
- [ ] Bab Hasil konsisten dengan frozen evidence.
- [ ] hypothesis decision table lengkap.
- [ ] reviewer-response matrix lengkap.
- [ ] defense Q&A bank lengkap.
- [ ] claim matrix lengkap.

## Reproducibility done
- [ ] checkpoint registry lengkap.
- [ ] result registry lengkap.
- [ ] SHA256 manifest lengkap.
- [ ] no synthetic/dummy production result.
- [ ] repository release reproducible.
- [ ] Git tag `v1.0-dissertation-freeze` dibuat.

---

# 18. PRINSIP INTERPRETASI FINAL

Hasil penelitian tidak harus menunjukkan bahwa SACSI adalah controller dengan angka performa tertinggi untuk dianggap bernilai.

Kontribusi ilmiah dapat berada pada:

1. formulasi POMDP;
2. integrasi state-history-forecast;
3. recurrent actor-critic framework;
4. controlled comparison dengan non-RL dan RL ablations;
5. validated Virtual Garden;
6. explicit context-activation diagnostics;
7. multi-seed reproducibility;
8. robustness analysis;
9. framework yang reusable untuk future smart irrigation research.

Jika strong rule-based controller tetap mengungguli SACSI pada penggunaan air dan target occupancy, hasil tersebut harus dilaporkan apa adanya dan menjadi bagian penting diskusi ilmiah.

---

# 19. NEXT ACTION YANG DIREKOMENDASIKAN

Karena Modul 8A–9D sudah dirancang, **next action bukan membuat modul tambahan**. Fokus sekarang adalah eksekusi nyata dan final freeze.

Urutan eksekusi:

```text
1. Masukkan hasil riil Modul 8B dan kunci reward decision.
2. Jalankan/validasi Modul 8C simple/raw episode protocol.
3. Jalankan DDPG 8D dan TD3 8E pada common environment.
4. Jalankan 8F fair DDPG–TD3–SAC benchmark.
5. Sinkronkan SAC-family existing ke protokol 8G.
6. Jalankan 8H 10 matched seeds.
7. Jalankan inferential statistics 8H.
8. Buat RESULT FREEZE.
9. Hubungkan frozen results ke dashboard 9A.
10. Rekonsiliasi angka ke disertasi 9B.
11. Finalkan reviewer/defense package 9C.
12. Jalankan 9D release guard dan buat v1.0 freeze.
```

## 19.1 No-new-module rule
Jangan membuat Modul 10 kecuali terdapat salah satu kondisi berikut:

1. reviewer meminta eksperimen baru secara eksplisit;
2. ditemukan methodological flaw yang memerlukan revisi mayor;
3. dataset/experiment integrity gagal pada final audit.

Selain kondisi tersebut, seluruh pekerjaan baru harus masuk sebagai **execution task** di Modul 8–9, bukan modul metodologis baru.

---

# 20. VERSION CONTROL MILESTONES

Tag historical:

```text
v0.1-data-pipeline
v0.2-virtual-garden
v0.3-baselines
v0.4-sac-basic
v0.5-sac-forecast
v0.6-sac-lstm
v0.7-sacsi-full
v0.8-ten-seed-existing
v0.9-reviewer-alignment-design
```

Tag reviewer-alignment extension:

```text
v0.10-reward-validated
v0.11-simple-case-validated
v0.12-ddpg-ready
v0.13-td3-ready
v0.14-fair-drl-benchmark
v0.15-pomdp-ablation
v0.16-confirmatory-10seed
v0.17-result-freeze
v0.18-dashboard-evidence-freeze
v0.19-dissertation-evidence-freeze
v1.0-dissertation-freeze
v1.1-publication-release
```

Setiap tag minimal memiliki:

- code;
- config;
- test report;
- result summary;
- checkpoint/result registry update;
- README update;
- git commit hash;
- artefact hash untuk freeze tags.

---

# 21. FINAL NOTE

Dokumen ini harus diperlakukan sebagai _execution contract_. Jika selama implementasi ingin mengubah:

- reward;
- target band;
- action bounds;
- sequence length utama;
- forecast horizon utama;
- training split;
- seed list;
- convergence gate;

perubahan harus dicatat secara eksplisit dalam `CHANGELOG.md` beserta alasan dan dampaknya terhadap fairness eksperimen.

Tujuan akhirnya bukan hanya memperoleh model yang “bagus”, tetapi menghasilkan penelitian yang **traceable, repeatable, statistically defensible, dan dapat dipertanggungjawabkan pada ujian disertasi maupun publikasi jurnal**.

---

# 22. CROSSWALK SELURUH MODUL YANG SUDAH DIBAHAS

Bagian ini memetakan seluruh modul lama ke sprint implementasi mandiri.

## TAHAP 1 — VIRTUAL GARDEN CORE

### 1A — Blueprint Virtual Garden Core
**Tujuan:** mendefinisikan state, input, output, parameter, neraca air, dan controller-independent architecture.

Masuk ke:

```text
Sprint 2
```

### 1B — Implementasi Virtual Garden Core
**Tujuan:** menulis source code simulator fisik.

Acceptance:

- unit tests;
- deterministic behavior;
- action tidak tertanam dalam core.

### 1C — Validasi Final Virtual Garden
**Tujuan:** membuktikan physics consistency.

Reference hasil sebelumnya:

```text
12/12 unit tests
~43,824 hourly rows
5 validation scenarios
mass balance ~ 0
```

---

# 23. TAHAP 2 — NON-RL CONTROLLER & DASHBOARD

## 2A — Blueprint Controller Interface & Dashboard

Definisikan:

- controller protocol;
- logging schema;
- result schema;
- metric engine;
- dashboard layout.

## 2B — Implementasi Controller Non-RL

Controller:

```text
No Irrigation
Fixed Schedule
Threshold-Based
Rule-Based Forecast-Aware
Fuzzy Controller
```

Reference sebelumnya:

```text
20/20 tests
```

## 2C — Benchmark Final Controller Non-RL

Reference test 2025:

| Controller | Water | Time in Target | Violation |
|---|---:|---:|---:|
| Rule-Based Forecast-Aware | ~270 mm | ~65.35% | ~34.65% |
| Threshold-Based | ~276 mm | ~65.02% | ~34.98% |
| Fuzzy | ~297 mm | ~61.98% | ~38.02% |
| No Irrigation | 0 | ~28.32% | ~71.68% |
| Fixed Schedule | ~1095 mm | ~12.95% | ~87.05% |

Reference test suite:

```text
30/30 tests
```

Strong active baseline:

```text
Rule-Based Forecast-Aware
```

## 2D — Unified Dashboard

Reference sebelumnya:

```text
42/42 tests
```

Gunakan Streamlit/Plotly dalam pengerjaan ulang.

---

# 24. TAHAP 3 — SAC BASIC

## 3A — Blueprint SAC Basic

Lock:

```text
8-D observation
no forecast
no LSTM
actor
Q1/Q2
target Q1/Q2
replay buffer
auto entropy
```

## 3B — Implementasi SAC Basic

Reference:

```text
30/30 tests
smoke 4 episodes
384 transitions
257 updates
```

## 3C — Training & Validation SAC Basic

Initial pilot pernah tidak convergence.

Extended training menemukan konfigurasi yang lebih stabil.

Reference validation 2024 3-seed:

```text
Seed 11: target ~54.57%
Seed 22: target ~59.10%
Seed 33: target ~55.65%
Mean     : ~56.44%
```

Reference one-shot 2025:

```text
Water ~410.09 ±24.86 mm
Target ~54.97 ±1.49%
Violation ~45.03 ±1.49%
```

## 3D — Benchmark Formal SAC Basic

Kesimpulan historis:

- belum mengalahkan Rule-Based/Threshold/Fuzzy;
- action jauh lebih smooth daripada Rule-Based;
- SAC Basic tetap valid sebagai baseline RL.

---

# 25. TAHAP 4 — SAC + FORECAST

## 4A — Blueprint

Initial design:

```text
17-D observation
8 current features
9 forecast features
h1, h2, h3
```

No LSTM.

## 4B — Implementasi

Reference:

```text
50/50 tests
384 transitions
257 updates
```

## 4C — Training & Validation

Critical methodological correction:

Historical Forecast continuous series tidak diperlakukan sebagai true issued forecast run.

Gunakan controlled forecast proxy:

```text
SF-20
```

Reference validation 2024:

```text
mean target ~56.60 ±1.05%
mean water ~610.53 ±20.13 mm
```

Reference one-shot 2025:

```text
water ~410.13 ±20.30 mm
target ~53.99 ±1.87%
```

## 4D — Benchmark, Ablation & Robustness

Reference finding:

```text
SAC Basic      ~54.97% target
SAC + Forecast ~53.99% target
```

Full h1-h3 belum mengalahkan SAC Basic.

Post-hoc h1 diagnostic pernah menunjukkan:

```text
~56.25% target
~398.50 mm water
```

Tetapi jika eksperimen diulang, horizon primary harus dipilih melalui validation-only protocol.

Robustness SF10–SF30 tidak collapse.

---

# 26. TAHAP 5 — SAC + LSTM

## 5A — Blueprint SAC + LSTM

```text
sequence = 24 x 8
no forecast
LSTM hidden 64
```

## 5B — Implementasi

Reference:

```text
50/50 tests
384 transitions
257 updates
mass balance ~1.42e-14 mm
```

## 5C — Initial Training

Direct recurrent SAC pernah gagal convergence:

```text
Target occupancy hanya sekitar 16–19%
Water sekitar 1,393–1,432 mm
```

Karena gate gagal, 2025 tidak langsung dibuka.

### Extended 5C — RRWS

Solusi:

```text
Residual Recurrent Warm-Start
```

Reference validation:

```text
mean target ~56.44%
mean water ~602.98 mm
```

Reference 2025:

```text
water ~409.98 ±24.35 mm
target ~55.01 ±1.46%
```

## 5D — Temporal Ablation

Reference formal benchmark:

```text
SAC + LSTM target ~54.99 ±1.46%
water ~409.88 ±24.30 mm
```

Critical finding:

```text
RRWS residual LSTM norm = 0
history interventions action delta = 0
sequence k sensitivity ~0
```

Interpretasi:

- recurrent architecture stable;
- standalone temporal benefit belum terbukti;
- jangan klaim LSTM meningkatkan performa.

---

# 27. TAHAP 6 — SACSI FULL

## 6A — Blueprint SACSI Full

Final concept:

```text
POMDP
Current State 8-D
History 24x8
Forecast h1 3-D
```

## 6B — Implementation

Reference:

```text
48/48 tests
384 transitions
257 gradient updates
mass balance ~2.84e-14 mm
context residual becomes non-zero
```

Interpretasi:

memory/forecast branches structurally trainable.

## 6C — Multi-Seed Training & Validation

Reference validation 2024:

| Seed | Water | Target | Violation |
|---|---:|---:|---:|
| 11 | ~613.31 | ~54.26% | ~45.74% |
| 22 | ~553.83 | ~60.68% | ~39.32% |
| 33 | ~652.23 | ~51.24% | ~48.76% |

Mean:

```text
Target ~55.39 ±4.82%
Water ~606.45 ±49.56 mm
```

Reference retrospective 2025:

```text
Water ~413.52 ±63.93 mm
Target ~54.49 ±4.24%
Violation ~45.51 ±4.24%
```

Seed 33 pernah berada sangat dekat gate:

```text
49.93% target occupancy
```

Jangan retune berdasarkan benchmark 2025.

## 6D — Final Benchmark

Reference ranking berdasarkan target occupancy lalu water:

1. Rule-Based Forecast-Aware
2. Threshold-Based
3. Fuzzy
4. SAC + LSTM
5. SAC Basic
6. SACSI Full
7. SAC + Forecast
8. No Irrigation
9. Fixed Schedule

Key conclusion:

```text
SACSI technically valid
context active
robustness passes
performance superiority not proven
```

Context ablation sebelumnya menunjukkan efek tahunan relatif kecil.

Forecast SF10–SF30 tidak collapse.

Sequence k=12/24/48 hampir identik.

---

# 28. TAHAP 7 — FINAL EXPERIMENT & STATISTICS

## 7A — Blueprint Statistical Reinforcement

Decision:

```text
minimum 10 matched seeds
```

Final seeds:

```text
11 22 33 44 55 66 77 88 99 110
```

Factorial design:

```text
Forecast OFF/ON
Memory OFF/ON
```

## 7B — Expanded Multi-Seed

Total:

```text
40 RL checkpoints
```

Reference validation gate counts yang pernah diperoleh:

```text
SAC Basic       8/10
SAC + Forecast  4/10
SAC + LSTM      8/10
SACSI Full      7/10
```

Reference 10-seed retrospective 2025 mean Time in Target:

```text
SAC Basic       ~50.00%
SAC + Forecast  ~42.15%
SAC + LSTM      ~50.01%
SACSI Full      ~51.92%
```

Key finding:

```text
seed sensitivity is substantial
```

Failed seeds must remain in primary analysis.

## 7C — Statistical Analysis

Descriptive factorial effects previously derived from aggregate means:

```text
Forecast main effect       ~ -2.97 pp
Memory main effect         ~ +4.89 pp
Forecast x Memory          ~ +9.76 pp
SACSI - Basic              ~ +1.92 pp
SACSI - Forecast           ~ +9.77 pp
SACSI - LSTM               ~ +1.91 pp
```

These are **descriptive**, not automatically significant.

Final inferential analysis must use the real seed-level 40-row master table.

Never reconstruct p-values from aggregate means.

## 7D — Final Results Chapter

Bab Hasil harus menyajikan:

- Virtual Garden validation;
- non-RL baseline;
- SAC Basic;
- SAC Forecast;
- SAC LSTM;
- SACSI Full;
- final benchmark;
- 10-seed distribution;
- factorial analysis;
- robustness;
- interpretation H1-H4.

Important narrative:

```text
H4 should not be forced accepted if superiority is not statistically supported.
```

---

# 29. TAHAP 8–9 — REVIEWER ALIGNMENT & FINALIZATION EXTENSION

Bagian ini menggantikan rencana Tahap 8 lama. Tahap 8–9 merupakan kelanjutan resmi setelah Modul 1–7 selesai.

---

## 29.1 MODUL 8A — REVIEWER ALIGNMENT: RUMUSAN MASALAH, TUJUAN, HIPOTESIS, DAN SCOPE

### Tujuan
Menyinkronkan struktur akademik penelitian dengan masukan reviewer tanpa membongkar implementasi teknis yang sudah selesai.

### Input
- master planning Modul 1–7;
- proposal/disertasi existing;
- catatan reviewer;
- keputusan metodologis final;
- hasil eksperimen yang sudah tersedia.

### Task utama
1. sinkronkan rumusan masalah dan tujuan secara one-to-one;
2. ubah tujuan menjadi measurable;
3. perkuat argumentasi POMDP;
4. perkuat argumentasi SAC;
5. argumentasikan DDPG, TD3, SAC, LSTM, forecast, dan Virtual Garden;
6. tetapkan scope kebun virtual sebagai **hortikultura generik**;
7. bedakan real/raw data, simulated state, dan synthetic forecast proxy;
8. definisikan optimasi sebagai proses dan efisiensi sebagai outcome;
9. tetapkan fair-comparison hierarchy.

### Struktur rumusan masalah final
```text
RM1 — Bagaimana memformulasikan Virtual Garden dan objective/reward irigasi?
RM2 — Bagaimana kinerja continuous-control DRL pada raw meteorological forcing?
RM3 — Bagaimana kontribusi forecast dan temporal memory dalam partial observability?
RM4 — Bagaimana kinerja dan robustness SACSI-POMDP secara confirmatory dan multi-seed?
```

### Struktur tujuan final
```text
T1 — Mengembangkan dan memvalidasi Virtual Garden dan reward multi-objective.
T2 — Mengukur dan membandingkan DDPG, TD3, dan SAC pada protokol yang fair.
T3 — Mengkuantifikasi kontribusi forecast dan temporal memory melalui ablation.
T4 — Menguji SACSI-POMDP pada 10 matched seeds, robustness, dan statistik final.
```

### POMDP argumentation lock
Internal simulator state:

```text
x_{t+1} = f(x_t, a_t, w_t)
```

Controller hanya menerima observation:

```text
o_t = h(x_t, w_t)
```

Temporal representation:

```text
z_t = LSTM(o_{t-k:t})
```

SACSI context:

```text
c_t = [o_t, z_t, w_hat_{t+1}]
```

### Acceptance gate 8A
- [ ] RM dan tujuan one-to-one;
- [ ] seluruh tujuan measurable;
- [ ] SAC dan POMDP memiliki argumentasi metodologis;
- [ ] scope hortikultura generik jelas;
- [ ] status data real/simulated/synthetic jelas;
- [ ] main comparator hierarchy terkunci;
- [ ] tidak ada klaim field validation yang belum dilakukan.

### Output minimum
```text
reviewer_alignment_matrix.csv
research_question_objective_map.csv
hypothesis_map.csv
scope_and_data_classification.md
```

---

## 29.2 MODUL 8B — REWARD FORMALIZATION, ABLATION, DAN WEIGHT SENSITIVITY

### Tujuan
Memvalidasi bahwa reward SAC/SACSI memiliki dasar multi-objective yang jelas dan tidak hanya hasil trial-and-error.

### Reward formal
Gunakan scalarized reward:

```text
r_t = -[
  w_theta * E_theta,t
  + w_I * E_I,t
  + w_dI * E_dI,t
  + w_V * E_V,t
]
```

Dengan:

```text
E_theta : distance/soil-moisture tracking error
E_I     : irrigation-water penalty
E_dI    : action-change/smoothness penalty
E_V     : target-band violation penalty
```

Target band tetap:

```text
0.22 <= theta_t <= 0.32
```

### Interpretasi objektif
```text
Maintain moisture
+ minimize water
+ smooth actuator
+ avoid target violation
```

### Reward ablation
```text
R-A : moisture only
R-B : moisture + water
R-C : moisture + water + smoothness
R-D : full reward
```

### Weight sensitivity
Fokus pada bobot yang paling memengaruhi trade-off:

```text
w_I multiplier = {0.5, 1.0, 2.0}
w_V multiplier = {0.5, 1.0, 2.0}
```

Total:

```text
9 configurations × 3 seeds = 27 runs
```

### Selection rule
Reward **tidak dipilih dari cumulative reward terbesar**.

Gunakan physical metrics:

1. Time in Target;
2. deficit / violation;
3. total irrigation;
4. RMSE;
5. smoothness;
6. Pareto Water vs Time in Target.

### Reward retention gate
Jika reward existing:
- non-dominated;
- stabil pada seed;
- tidak material lebih buruk dari kandidat terbaik pada validation 2024;

maka:

```text
KEEP CURRENT REWARD
```

Tujuannya menghindari retraining seluruh Modul 3–7 tanpa kebutuhan ilmiah.

### Important distinction
```text
reward weights != SAC entropy alpha
```

SAC mengoptimalkan policy terhadap reward. Bobot reward dipilih melalui outer validation experiment. Entropy temperature alpha dapat dituning/dipelajari oleh SAC secara internal.

### Acceptance gate 8B
- [ ] setiap reward term memiliki alasan fisik/kontrol;
- [ ] semua term ternormalisasi atau terukur skala pengaruhnya;
- [ ] ablation selesai;
- [ ] local sensitivity selesai;
- [ ] Pareto plot tersedia;
- [ ] keputusan KEEP/REVISE reward terdokumentasi;
- [ ] 2025 tidak dipakai untuk memilih reward.

### Output minimum
```text
reward_ablation_results.csv
reward_weight_sensitivity.csv
reward_pareto.csv
reward_decision.json
```

---

## 29.3 MODUL 8C — SIMPLE-CASE TESTING & RAW-DATA EPISODE VALIDATION

### Tujuan
Menjawab reviewer bahwa pengujian harus dimulai dari kasus sederhana dan menggunakan raw meteorological data.

### Simple cases
Minimal enam skenario:

```text
C1 dry-down without irrigation/rain
C2 rainfall pulse
C3 irrigation pulse
C4 near-upper-band protection
C5 below-target recovery
C6 heavy-rain suppression
```

### Raw-data episodes
Gunakan potongan 2024 yang dipilih secara objektif, bukan berdasarkan hasil controller:

```text
DRY   : 14-day low-rain / high-demand episode
WET   : 14-day high-rain episode
MIXED : 14-day mixed episode
```

Reference episode yang sudah digunakan:

```text
DRY   : 16–29 Apr 2024, rain ~0.3 mm
WET   : 27 Nov–10 Dec 2024, rain ~419.4 mm
MIXED : 17–30 Dec 2024, rain ~77.4 mm
```

### Data-classification lock
```text
Meteorological forcing = real/raw data
Soil-water state        = Virtual Garden simulated state
Forecast input          = controlled synthetic forecast proxy
```

### Acceptance gate 8C
- [ ] action selalu 0–5 mm/hour;
- [ ] mass-balance residual <= 1e-8 mm;
- [ ] no NaN/Inf;
- [ ] same environment/config;
- [ ] no episode-specific retuning;
- [ ] dry/wet/mixed episodes terdokumentasi;
- [ ] simple cases menghasilkan respons fisik yang masuk akal.

### Output minimum
```text
simple_case_results.csv
raw_episode_dry.csv
raw_episode_wet.csv
raw_episode_mixed.csv
simple_case_figures/
```

---

## 29.4 MODUL 8D — DDPG CONTINUOUS-CONTROL BASELINE

### Tujuan
Menambahkan baseline continuous-control klasik yang lebih sederhana dari TD3/SAC.

### Architecture lock
```text
Observation : 8-D current state
Action      : 0–5 mm/hour
Actor       : deterministic
Critic      : single Q(o,a)
Forecast    : OFF
Memory      : OFF
```

### Core mechanisms
- deterministic actor;
- single critic;
- replay buffer;
- target actor/critic;
- exploration noise;
- soft target update.

### Fairness rule
DDPG menggunakan:
- Virtual Garden yang sama;
- reward yang sama;
- train/validation split yang sama;
- observation yang sama dengan SAC Basic;
- network capacity yang sebanding;
- budget environment interaction yang sama pada final benchmark.

### Initial seeds
```text
11 22 33
```

### Final seeds
```text
11 22 33 44 55 66 77 88 99 110
```

### Acceptance gate 8D
- [ ] unit tests lulus;
- [ ] action bounds valid;
- [ ] actor/critic loss finite;
- [ ] replay buffer valid;
- [ ] target update valid;
- [ ] checkpoint save/load valid;
- [ ] smoke test selesai;
- [ ] tidak menggunakan forecast/history.

### Output minimum
```text
ddpg_config.yaml
ddpg_seedXX_best.pt
ddpg_validation_results.csv
ddpg_training_log.csv
```

---

## 29.5 MODUL 8E — TD3 CONTINUOUS-CONTROL BASELINE

### Tujuan
Menambahkan baseline deterministic actor-critic yang memperbaiki DDPG.

### Architecture lock
```text
Observation : 8-D
Action      : 0–5 mm/hour
Actor       : deterministic
Critics     : twin Q1/Q2
Forecast    : OFF
Memory      : OFF
```

### Core mechanisms
```text
1. Twin Critics
2. Clipped Double-Q Target
3. Target Policy Smoothing
4. Delayed Actor Update
```

Target:

```text
y_t = r_t + gamma * min(Q1', Q2')
```

### Policy delay
Reference:

```text
policy_delay = 2
```

### Fairness rule
Sama dengan DDPG/SAC pada:
- environment;
- reward;
- observation;
- action;
- data split;
- environment-interaction budget;
- metric engine;
- seed.

### Acceptance gate 8E
- [ ] twin critics independent;
- [ ] target action noise clipped;
- [ ] delayed actor update verified;
- [ ] action finite dan bounded;
- [ ] checkpoint valid;
- [ ] smoke test lulus;
- [ ] no forecast/history.

### Output minimum
```text
td3_config.yaml
td3_seedXX_best.pt
td3_validation_results.csv
td3_training_log.csv
```

---

## 29.6 MODUL 8F — FAIR DRL BENCHMARK: DDPG vs TD3 vs SAC

### Tujuan
Mengisolasi perbedaan algoritmik sebelum menambahkan POMDP context.

### Main benchmark
```text
DDPG
TD3
SAC Basic
```

### Fairness lock
Harus sama:

```text
Virtual Garden
Raw weather data
Observation 8-D
Action 0–5 mm/hour
Reward
Train 2021–2023
Validation 2024
Retrospective benchmark 2025
Matched seeds
Metric engine
Comparable network capacity
Environment-interaction budget
Checkpoint selection rule
```

### Reference budget
Jika memakai keputusan modul 8F existing:

```text
max 20 episodes × 336 h = 6,720 environment interactions / seed
```

Gunakan budget aktual yang sama di seluruh model final dan dokumentasikan jika angka final berubah.

### Checkpoint selection
Hanya menggunakan validation 2024:

```text
1. validation gate pass
2. highest Time in Target
3. lowest Water
4. lowest RMSE
```

### 2025 lock
```text
NO RETUNING AFTER OPENING 2025
```

### Acceptance gate 8F
- [ ] fairness config hash sama untuk common fields;
- [ ] DDPG/TD3/SAC menjalankan matched seeds;
- [ ] training budget terdokumentasi;
- [ ] validation selection tidak menggunakan 2025;
- [ ] benchmark metrics memakai common schema;
- [ ] failed seeds tidak dibuang.

### Output minimum
```text
fair_drl_results_validation.csv
fair_drl_results_2025.csv
fair_drl_checkpoint_registry.csv
fairness_audit.json
```

---

## 29.7 MODUL 8G — INCREMENTAL POMDP CONTRIBUTION & ABLATION STUDY

### Tujuan
Mengukur kontribusi forecast dan temporal memory setelah SAC Basic dikunci sebagai anchor algorithm.

### Factorial design
```text
F0M0 = SAC Basic
F1M0 = SAC + Forecast
F0M1 = SAC + LSTM
F1M1 = SACSI Full
```

### Primary representation
```text
Current observation : 8-D
Forecast context     : h+1, 3-D
History              : 24 × 8
LSTM hidden          : 64
```

### Forecast protocol
Primary:

```text
SF20
```

Robustness:

```text
SF10
SF20
SF30
```

### Sequence sensitivity
```text
k = 6, 12, 24, 48 hours
```

### Context intervention diagnostics
History:

```text
current-only
zero history
shuffle history
reverse history
```

Forecast:

```text
full forecast
zero forecast
shuffle forecast
```

### Factorial effects
Untuk metric Y:

```text
Forecast main effect
= [(F-B) + (X-M)] / 2

Memory main effect
= [(M-B) + (X-F)] / 2

Interaction
= X - F - M + B
```

### Critical claim guard
```text
branch active != performance benefit
performance benefit != statistical superiority
```

### 2025 rule
Pada exploratory/revalidation version 8G, 2025 tidak digunakan untuk retuning.

### Acceptance gate 8G
- [ ] 2×2 family lengkap;
- [ ] forecast/history activation diagnostics selesai;
- [ ] robustness SF10/SF20/SF30 selesai;
- [ ] sequence sensitivity selesai;
- [ ] no branch claimed beneficial without metric evidence;
- [ ] no significance claimed from aggregate means only.

### Output minimum
```text
sac_family_factorial_results.csv
context_intervention_results.csv
forecast_robustness.csv
sequence_sensitivity.csv
factorial_effects.csv
```

---

## 29.8 MODUL 8H — FINAL 10-SEED CONFIRMATORY BENCHMARK & STATISTICAL ANALYSIS

### Tujuan
Menjadi pengujian konfirmatori final untuk main benchmark dan POMDP factorial family.

### Main benchmark algorithms
```text
DDPG
TD3
SAC
SACSI-POMDP
```

### Final matched seeds
```text
11
22
33
44
55
66
77
88
99
110
```

Total main benchmark rows:

```text
4 algorithms × 10 seeds = 40 rows
```

### SAC-family factorial
```text
SAC Basic
SAC + Forecast
SAC + LSTM
SACSI Full
```

Total factorial rows:

```text
4 variants × 10 seeds = 40 rows
```

### Primary endpoint
```text
Time in Target (%)
```

### Secondary endpoints
- total irrigation;
- violation rate;
- RMSE soil moisture;
- action smoothness;
- deficit-related metrics;
- WUE jika definisi final konsisten.

### Main statistical plan
Omnibus:

```text
Friedman test
```

Planned paired contrasts:

```text
SACSI vs SAC
SACSI vs TD3
SACSI vs DDPG
```

Dengan:

```text
exact paired sign-flip test
Holm correction
Cohen's dz
20,000 paired bootstrap CI
```

Parametric sensitivity:

```text
repeated-measures ANOVA
```

### Factorial inferential analysis
```text
2×2 within-seed repeated-measures analysis
Forecast factor
Memory factor
Forecast × Memory interaction
```

Tambahkan robust sign-flip testing terhadap subject-level effects bila dibutuhkan.

### Seed policy
```text
DO NOT DROP FAILED SEEDS
DO NOT REPLACE BAD SEEDS
DO NOT ADD SELECTIVE EXTRA TRAINING AFTER TEST REVIEW
```

### 2025 wording
Gunakan:

```text
retrospective final benchmark 2025
```

### Acceptance gate 8H
- [ ] 40-row main table lengkap;
- [ ] 40-row factorial table lengkap;
- [ ] seed pairing valid;
- [ ] no duplicate algorithm-seed rows;
- [ ] no missing primary endpoint;
- [ ] test leakage audit lulus;
- [ ] Holm/effect size/CI tersedia;
- [ ] negative/null results dilaporkan apa adanya.

### Output minimum
```text
main_10seed_results_2025.csv
sac_family_10seed_factorial.csv
friedman_results.csv
planned_contrasts.csv
holm_adjusted_results.csv
bootstrap_ci.csv
factorial_inference.csv
final_statistics_summary.json
```

---

# 29.9 TAHAP 9 — EVIDENCE INTEGRATION, DISSERTATION, DEFENSE, DAN FREEZE

Tahap 9 bukan pengembangan algoritma baru. Tahap ini mengubah evidence yang sudah dikunci menjadi artefak final penelitian.

---

## 29.9.1 MODUL 9A — FINAL DASHBOARD & EVIDENCE INTEGRATION

### Tujuan
Mengintegrasikan semua evidence Modul 8A–8H ke dashboard final yang reviewer-oriented.

### Page map
```text
1 Research Design
2 Reward Lab
3 Simple-Case & Raw-Data Validation
4 Fair DRL Benchmark
5 POMDP Contribution
6 10-Seed Confirmatory Statistics
7 Robustness & Context Diagnostics
8 Reviewer Evidence Matrix
9 Reproducibility & Provenance
```

### Final-mode guard
Jika hasil riil belum ada:

```text
NOT READY
```

bukan angka dummy.

### Synthetic-data guard
File test / fixture sintetis tidak boleh terbaca sebagai result production.

### Reviewer Evidence Matrix
Setiap reviewer item minimal memiliki:
- pertanyaan/masukan;
- module source;
- evidence file;
- dashboard page;
- claim status;
- readiness status.

### Acceptance gate 9A
- [ ] seluruh page terbuka;
- [ ] real result registry valid;
- [ ] missing data tampil NOT READY;
- [ ] no synthetic production evidence;
- [ ] angka dashboard sama dengan frozen result files;
- [ ] reviewer matrix 100% mapped.

### Output minimum
```text
dashboard/app.py
dashboard/pages/
result_registry.csv
reviewer_evidence_matrix.csv
dashboard_release_metadata.json
```

---

## 29.9.2 MODUL 9B — FINAL DISSERTATION RESULTS & DISCUSSION INTEGRATION

### Tujuan
Mengubah evidence menjadi naskah disertasi yang konsisten dan tidak overclaim.

### Final Bab Hasil structure
```text
5.1 Data Audit & Provenance
5.2 Virtual Garden Validation
5.3 Reward Validation
5.4 Simple-Case & Raw-Data Validation
5.5 Fair DDPG–TD3–SAC Benchmark
5.6 Incremental POMDP Contribution
5.7 10-Seed Confirmatory Benchmark
5.8 Robustness & Diagnostics
5.9 Jawaban RM1–RM4
```

### Final Bab Pembahasan
Bahas secara terpisah:
- reward trade-off;
- DDPG vs TD3 vs SAC;
- partial observability;
- forecast contribution;
- memory contribution;
- SACSI integration;
- precision-farming implication;
- limitations;
- future work.

### Required wording corrections
1. forecast adalah **controlled synthetic forecast proxy** jika tidak ada archived as-issued forecast;
2. IoT adalah konteks sensing/implementation, bukan otomatis field validation;
3. raw meteorology dan simulated soil state harus dibedakan;
4. energy metric hanya dipakai jika power model benar-benar tersedia;
5. H4 tidak dipaksa diterima jika superiority tidak didukung.

### Claim hierarchy
```text
Framework validity
    !=
Context activation
    !=
Performance benefit
    !=
Statistical superiority
```

### Acceptance gate 9B
- [ ] setiap angka Bab Hasil memiliki source file;
- [ ] setiap hypothesis decision memiliki evidence;
- [ ] tidak ada angka manual yang berbeda dengan dashboard;
- [ ] forecast/data wording konsisten;
- [ ] limitations eksplisit;
- [ ] no unsupported superiority claim.

### Output minimum
```text
dissertation_update_map.csv
result_insertion_matrix.csv
hypothesis_decision_table.csv
claim_matrix.csv
chapter_results_draft.docx/md
chapter_discussion_draft.docx/md
```

---

## 29.9.3 MODUL 9C — FINAL REVIEWER RESPONSE, DEFENSE NARRATIVE & CLAIM-TO-EVIDENCE MATRIX

### Tujuan
Mempersiapkan paket pertahanan akademik untuk reviewer dan sidang.

### Reviewer response format
Setiap reviewer memiliki:

```text
Reviewer comment
→ short answer 30–60 sec
→ technical answer
→ mathematical support
→ evidence source
→ claim guard
```

### Defense topics wajib
- mengapa SAC;
- mengapa POMDP;
- mengapa DDPG/TD3 sebagai comparator;
- fairness benchmark;
- reward multi-objective;
- raw vs simulated vs synthetic data;
- virtual-garden scope;
- optimization vs efficiency;
- partial observability;
- negative result interpretation;
- statistical evidence.

### Claim status
Klaim berikut hanya dibuka bila 8H mendukung:

```text
Forecast improves performance
Memory improves performance
SACSI statistically outperforms SAC/TD3/DDPG
```

Klaim berikut tetap tidak boleh tanpa eksperimen baru:

```text
external field effectiveness
```

### Acceptance gate 9C
- [ ] seluruh reviewer item memiliki jawaban;
- [ ] seluruh jawaban menunjuk evidence;
- [ ] defense Q&A bank tersedia;
- [ ] one-page defense card tersedia;
- [ ] unsupported claim memiliki red-flag warning;
- [ ] jawaban tetap konsisten dengan 8H.

### Output minimum
```text
reviewer_response_matrix.csv
defense_qa_bank.csv
claim_to_evidence_matrix.csv
red_flag_wording.md
one_page_defense_card.md
```

---

## 29.9.4 MODUL 9D — REPRODUCIBILITY, REPOSITORY FREEZE, FINAL SUBMISSION & PUBLICATION PACKAGE

### Tujuan
Mengunci artefak final agar penelitian dapat diaudit, direproduksi, dan diturunkan ke disertasi serta publikasi tanpa perubahan diam-diam.

### Freeze levels
```text
PRE-FREEZE
   ↓
RESULT FREEZE
   ↓
DISSERTATION FREEZE
   ↓
PUBLICATION RELEASE
```

### PRE-FREEZE
Diperbolehkan selama:
- hasil riil 8H belum lengkap;
- checkpoint registry belum lengkap;
- statistik final belum terkunci.

### RESULT FREEZE
Syarat:
- main 40-row result valid;
- factorial 40-row result valid;
- statistics generated;
- hashes dibuat;
- no post-test retuning.

### DISSERTATION FREEZE
Syarat tambahan:
- dashboard sama dengan frozen result;
- Bab Hasil sama dengan frozen result;
- hypothesis decisions terkunci;
- claim matrix terkunci.

### PUBLICATION RELEASE
Syarat tambahan:
- manuscript derived from frozen evidence;
- repository sanitized;
- dataset-license/provenance documented;
- reproducibility package generated.

### Required registries
```text
checkpoint_registry.csv
result_registry.csv
artifact_manifest_sha256.csv
publication_package_map.csv
claim_release_matrix.csv
submission_inventory.csv
```

### 40-slot checkpoint registry
Main benchmark:

```text
DDPG   × 10
TD3    × 10
SAC    × 10
SACSI  × 10
= 40 checkpoint slots
```

SAC-family ablation checkpoints dapat disimpan dalam registry terpisah jika berbeda dari main benchmark checkpoints.

### Source-of-truth pipeline
```text
Frozen Results
    ↓
Statistics
    ↓
Frozen Summary
    ↓
Dashboard / Dissertation / Manuscript
```

Dashboard dan naskah **tidak boleh menghitung ulang evidence dengan pipeline yang berbeda**.

### Release guard
Sistem final release harus gagal jika:
- result file missing;
- duplicate seed;
- synthetic fixture terdeteksi;
- hash mismatch;
- checkpoint missing;
- dashboard number mismatch;
- manuscript number mismatch;
- result modified after freeze tanpa version bump.

### Recommended tag
```text
v1.0-dissertation-freeze
```

Publication release dapat menggunakan:

```text
v1.1-publication-release
```

### Acceptance gate 9D
- [ ] SHA256 manifest valid;
- [ ] checkpoint registry complete;
- [ ] result registry complete;
- [ ] synthetic guard lulus;
- [ ] freeze metadata valid;
- [ ] repository README reproducible;
- [ ] environment/requirements tersedia;
- [ ] source-of-truth reconciliation lulus;
- [ ] final Git tag dibuat.

### Output final
```text
SACSI_POMDP_Reproducibility_v1.0/
├── README.md
├── LICENSE_or_data_notice.md
├── requirements.txt
├── environment.yml
├── configs/
├── src/
├── tests/
├── notebooks/
├── data_provenance/
├── checkpoint_registry.csv
├── result_registry.csv
├── master_results.csv
├── statistics/
├── figures/
├── tables/
├── dashboard/
├── dissertation_evidence/
├── publication_package/
├── artifact_manifest_sha256.csv
├── CHANGELOG.md
└── FREEZE_METADATA.json
```

---

## 29.10 MASTER ACCEPTANCE GATE REVIEWER-ALIGNMENT EXTENSION

Project tidak boleh masuk FINAL DISSERTATION FREEZE sebelum seluruh gate berikut selesai:

| Gate | Modul | Status wajib |
|---|---|---|
| RA | 8A | research questions/objectives/scope locked |
| RB | 8B | reward decision locked |
| RC | 8C | simple/raw episode validation passed |
| RD | 8D | DDPG implementation valid |
| RE | 8E | TD3 implementation valid |
| RF | 8F | DDPG–TD3–SAC fairness audit passed |
| RG | 8G | POMDP ablation/diagnostics complete |
| RH | 8H | 10-seed confirmatory statistics complete |
| IA | 9A | dashboard evidence integration passed |
| IB | 9B | dissertation evidence reconciliation passed |
| IC | 9C | reviewer/defense evidence map complete |
| ID | 9D | reproducibility freeze passed |

---

## 29.11 DO-NOT-CHANGE LIST AFTER 8H RESULT FREEZE

Setelah RESULT FREEZE, berikut tidak boleh berubah tanpa membuka versi eksperimen baru:

```text
Virtual Garden parameters
Target band
Action bounds
Reward definition
Raw-data split
Forecast protocol
Sequence protocol
Matched seed set
Training budget
Checkpoint-selection rule
Metric definitions
Statistical plan
```

Jika perubahan memang diperlukan, gunakan versi baru:

```text
v1.x → v2.0 experimental revision
```

bukan overwrite hasil lama.

---

# 30. UPDATED SPRINT / MODULE DEPENDENCY GRAPH

```text
MODUL 1–7 EXISTING
      ↓
8A Reviewer Alignment
      ↓
8B Reward Validation
      ↓
8C Simple Case + Raw Episodes
      ↓
 ┌───────────────┬───────────────┐
 ↓               ↓               ↓
8D DDPG        8E TD3       Existing SAC Basic
 └───────────────┴───────────────┘
                 ↓
        8F Fair DRL Benchmark
                 ↓
        8G POMDP Contribution
                 ↓
        8H 10-Seed Confirmatory
                 ↓
        9A Dashboard Integration
                 ↓
        9B Dissertation Integration
                 ↓
        9C Reviewer / Defense Pack
                 ↓
        9D Reproducibility Freeze
                 ↓
      v1.0-dissertation-freeze
```

## 30.1 Dependency rules
- 8B harus selesai sebelum fair DDPG/TD3/SAC retraining jika reward berubah.
- 8C harus tersedia sebelum 8D–8F agar simple-case protocol sama.
- 8D dan 8E harus stabil sebelum 8F.
- 8F harus dikunci sebelum SACSI dimasukkan ke main confirmatory benchmark.
- 8G menggunakan SAC-family existing tetapi mengikuti reward/fairness decision terbaru.
- 8H adalah satu-satunya sumber inferensi final.
- 9A–9C adalah consumer evidence 8H.
- 9D membekukan seluruh artefak setelah reconciliation.

---

# 31. UPDATED COMPUTE STRATEGY

| Modul/Sprint | CPU | GPU | Recommended Service | Catatan |
|---|---:|---:|---|---|
| 0–4 Setup/Data/VG/Baseline | ✓ | – | Colab | existing |
| 5 SAC Basic | ✓ | ✓ | Colab Pro | existing |
| 6 SAC Forecast | ✓ | ✓ | Colab Pro | existing |
| 7 SAC LSTM | ✓ | ✓✓ | Colab Pro+ / RunPod | existing |
| SACSI Full | ✓ | ✓✓ | Colab Pro+ / RunPod | existing |
| 8A | ✓ | – | Local/Colab | narrative/alignment |
| 8B | ✓ | ✓ | Colab Pro | 27 local sensitivity runs target |
| 8C | ✓ | – | Colab | deterministic/simple validation |
| 8D DDPG | ✓ | ✓ | Colab Pro | 3 seed dev → 10 seed final |
| 8E TD3 | ✓ | ✓ | Colab Pro | 3 seed dev → 10 seed final |
| 8F Fair DRL | ✓ | ✓✓ | Colab Pro+ / RunPod | DDPG/TD3/SAC matched |
| 8G POMDP Ablation | ✓ | ✓✓ | Colab Pro+ / RunPod | forecast/memory diagnostics |
| 8H Confirmatory | ✓✓ | ✓✓✓ | RunPod / Colab Pro+ | 10-seed final + stats |
| 9A Dashboard | ✓ | – | Streamlit Cloud | read-only evidence |
| 9B Dissertation | ✓ | – | Local/Word | evidence reconciliation |
| 9C Defense | ✓ | – | Local | no new training |
| 9D Freeze | ✓ | – | GitHub/Local | hash/release checks |

## 31.1 Compute priority if resources are limited
Prioritas GPU:

```text
1. DDPG/TD3 main 10-seed
2. SAC main 10-seed only if fair retraining required
3. SACSI final 10-seed
4. reward sensitivity
5. robustness / diagnostic repeats
```

Jangan mengorbankan 10-seed confirmatory benchmark demi terlalu banyak exploratory hyperparameter runs.

---

# 32. DAILY WORKFLOW YANG DISARANKAN

Setiap sesi kerja:

```text
1. git pull
2. mount Drive
3. load config
4. run unit tests
5. execute experiment
6. save logs
7. save checkpoint
8. update master_results.csv
9. render figure/table
10. git commit
11. update CHANGELOG.md
```

Commit format contoh:

```text
feat: implement SACSI context encoder
fix: prevent sequence leakage across episodes
exp: add seed 77 SAC Basic validation
stats: add Holm corrected pairwise analysis
```

---

# 33. EXPERIMENT NAMING STANDARD

Gunakan format:

```text
{model}_{seed}_{split}_{config_version}_{date}
```

Contoh:

```text
sacsi_seed44_validation_v1_20260810
```

Checkpoint:

```text
sacsi_full_seed44_best.pt
```

Log:

```text
sacsi_full_seed44_validation.csv
```

Metadata:

```text
sacsi_full_seed44_metadata.json
```

---


## 33.1 Naming standard untuk Modul 8–9

Contoh:

```text
ddpg_seed44_validation_rewardR4_v1
td3_seed44_validation_rewardR4_v1
sac_seed44_validation_rewardR4_v1
sacsi_seed44_validation_sf20_k24_v1
```

Final result file:

```text
main_10seed_results_2025_v1.csv
sac_family_factorial_2025_v1.csv
```

Freeze artefact:

```text
FREEZE_METADATA_v1.0.json
artifact_manifest_sha256_v1.0.csv
```

# 34. MINIMUM CONTENT CHECKPOINT METADATA

Setiap checkpoint harus menyimpan:

```json
{
  "model": "SACSI Full",
  "seed": 44,
  "training_period": "2021-2023",
  "validation_period": "2024",
  "sequence_length": 24,
  "forecast_horizon": 1,
  "reward_version": "locked",
  "virtual_garden_version": "locked",
  "validation_metrics": {},
  "git_commit": "...",
  "timestamp": "..."
}
```

---

# 35. FINAL MANUSCRIPT CLAIM MATRIX

| Claim | Evidence Required | Allowed? |
|---|---|---|
| Virtual Garden physically consistent | mass-balance tests | Yes |
| SACSI integrates POMDP context | architecture + code | Yes |
| Forecast branch trainable | gradient/intervention | Yes |
| Memory branch trainable | gradient/intervention | Yes |
| SACSI robust to forecast error | SF10-SF30 | Yes if reproduced |
| SACSI smoother than Rule-Based | smoothness benchmark | Yes if reproduced |
| SACSI uses less water than Rule-Based | final benchmark | No if current pattern persists |
| SACSI highest target occupancy | final benchmark | No if current pattern persists |
| SACSI statistically superior | inferential 10-seed stats | Only if supported |
| Forecast × Memory synergy significant | interaction p + effect size | Only if supported |

---

# 36. FINAL REPRODUCIBILITY PACKAGE

Sebelum disertasi dinyatakan final, buat ZIP/repository release:

```text
SACSI_POMDP_Reproducibility_v1.0/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
├── src/
├── tests/
├── notebooks/
├── checkpoints_manifest.csv
├── master_results.csv
├── figures/
├── tables/
├── statistics/
├── dashboard/
└── CHANGELOG.md
```

Dataset besar dapat tetap berada di Google Drive jika tidak boleh dipublikasikan ke GitHub.

---

# 37. PROJECT MANAGEMENT BOARD

Gunakan GitHub Projects / Trello / Notion dengan kolom:

```text
BACKLOG
READY
IN PROGRESS
VALIDATION
DONE
BLOCKED
```

Label:

```text
DATA
PHYSICS
BASELINE
RL
FORECAST
LSTM
SACSI
STATISTICS
DASHBOARD
DISSERTATION
BUG
EXPERIMENT
```

Setiap issue harus menyebut:

- objective;
- input;
- expected output;
- acceptance gate;
- related sprint;
- related figure/table jika ada.

---

# 37A. FINAL EXECUTION MATRIX MODUL 8A–9D

| Modul | Jenis kerja | Input utama | Output utama | GPU? | Stop/Go condition |
|---|---|---|---|---:|---|
| 8A | Method alignment | reviewer notes + proposal | RM/T/H/scope lock | No | all reviewer logic mapped |
| 8B | Experiment | SAC Basic + reward | reward decision | Yes | KEEP/REVISE locked |
| 8C | Validation | VG + raw 2024 | simple/raw evidence | No | physics/control sanity pass |
| 8D | Algorithm | common env | DDPG | Yes | implementation pass |
| 8E | Algorithm | common env | TD3 | Yes | implementation pass |
| 8F | Benchmark | DDPG/TD3/SAC | fair benchmark | Yes | fairness audit pass |
| 8G | Ablation | SAC family | POMDP contribution | Yes | diagnostics complete |
| 8H | Confirmatory | 10 seeds | final statistics | Yes | result freeze candidate |
| 9A | Integration | frozen evidence | dashboard | No | numbers reconcile |
| 9B | Writing | frozen evidence | dissertation results | No | claims reconcile |
| 9C | Defense | evidence map | Q&A / reviewer response | No | defense-ready |
| 9D | Freeze | all final artifacts | v1.0 release | No | hash/release guard pass |

---

# 37B. FINAL FILE INVENTORY SETELAH MODUL 9D

Minimum files that must exist:

```text
01_data_provenance/
02_virtual_garden_validation/
03_reward_validation/
04_simple_case_validation/
05_ddpg/
06_td3/
07_sac/
08_sacsi/
09_fair_benchmark/
10_pomdp_ablation/
11_confirmatory_statistics/
12_dashboard/
13_dissertation_evidence/
14_reviewer_defense/
15_publication_package/
```

Master tables:

```text
master_results.csv
main_10seed_results_2025.csv
sac_family_10seed_factorial.csv
checkpoint_registry.csv
result_registry.csv
reviewer_evidence_matrix.csv
claim_matrix.csv
artifact_manifest_sha256.csv
```

---

# 37C. FINAL CLAIM RELEASE POLICY

| Claim | Minimum evidence | Release condition |
|---|---|---|
| Virtual Garden numerically consistent | mass-balance tests | gate Physics pass |
| Reward terms justified | 8B ablation/sensitivity | reward gate pass |
| DDPG/TD3/SAC compared fairly | 8F fairness audit | fairness gate pass |
| SACSI integrates history/forecast | architecture + activation diagnostics | 8G pass |
| Forecast improves performance | 10-seed effect + CI/statistics | only if supported |
| Memory improves performance | 10-seed effect + CI/statistics | only if supported |
| Forecast×Memory interaction exists | factorial inference | only if supported |
| SACSI > SAC/TD3/DDPG | planned contrasts + effect size + CI | only if supported |
| SACSI robust to forecast perturbation | SF10/SF20/SF30 | only if reproduced |
| Field effectiveness | real field test | NOT allowed without new evidence |

---

# 37D. FINAL STOP CONDITIONS

Hentikan eksperimen tambahan dan masuk freeze jika:

```text
1. reward decision locked;
2. simple/raw episode validation passed;
3. DDPG/TD3/SAC fairness audit passed;
4. SACSI POMDP diagnostics complete;
5. all 10 matched seeds complete;
6. confirmatory statistical plan executed;
7. no unresolved data leakage;
8. dashboard/dissertation numbers reconcile;
9. reviewer evidence matrix complete.
```

Jangan menambah algoritma baru setelah kondisi ini tercapai kecuali reviewer secara eksplisit meminta benchmark tambahan.

---

# 38. PENUTUP MASTER PLANNING

Jika seluruh sprint dikerjakan ulang secara mandiri, urutan utama yang harus selalu dijaga adalah:

```text
VALIDATE ENVIRONMENT
      ↓
ESTABLISH BASELINES
      ↓
BUILD SAC / FORECAST / MEMORY / SACSI
      ↓
REVIEWER ALIGNMENT
      ↓
VALIDATE REWARD
      ↓
SIMPLE CASE + RAW DATA EPISODES
      ↓
DDPG → TD3 → SAC FAIR BENCHMARK
      ↓
INCREMENTAL POMDP ABLATION
      ↓
10-SEED CONFIRMATORY BENCHMARK
      ↓
STATISTICALLY TEST
      ↓
INTEGRATE DASHBOARD + DISSERTATION + DEFENSE
      ↓
FREEZE REPOSITORY
      ↓
REPORT WITHOUT OVERCLAIM
```

Kualitas project dinilai bukan dari apakah SACSI selalu menang, melainkan dari apakah eksperimen dilakukan secara konsisten, bebas leakage, fair, multi-seed, reproducible, dan menghasilkan kesimpulan yang benar-benar mengikuti data.


---

# 39. MASTER PLAN REVISION LOG — MODUL 8A–9D

**Revision:** Reviewer-Alignment Extension

**Scope added:**

```text
8A Reviewer Alignment
8B Reward Formalization / Ablation / Sensitivity
8C Simple-Case + Raw-Data Episode Validation
8D DDPG
8E TD3
8F Fair DDPG–TD3–SAC Benchmark
8G Incremental POMDP Contribution
8H Final 10-Seed Confirmatory Benchmark
9A Final Dashboard & Evidence Integration
9B Dissertation Results & Discussion Integration
9C Reviewer Response / Defense Narrative
9D Reproducibility / Repository Freeze / Publication Package
```

**Core principle:** Modul 1–7 tetap menjadi fondasi. Modul 8–9 menambahkan reviewer alignment dan confirmatory evidence tanpa menghapus hasil negatif atau mengubah metodologi setelah benchmark dibuka.
