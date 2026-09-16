# MASTER PLANNING PROJECT SACSI-POMDP

## Pengembangan Model Soft Actor-Critic for Smart Irrigation dalam Kerangka POMDP untuk Irigasi Cerdas Adaptif

**Tujuan dokumen**  
Dokumen ini menjadi _master execution plan_ untuk mengerjakan ulang seluruh project SACSI-POMDP secara mandiri, terstruktur, reproducible, dan konsisten dengan seluruh keputusan metodologis yang sudah dibahas. Seluruh implementasi disusun modular agar setiap controller dapat diuji, dibandingkan, diganti, atau dikembangkan tanpa mengubah Virtual Garden Core.

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
7. dashboard terpadu untuk visualisasi dan evaluasi.

Model final disebut **SACSI Full**.

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
SAC tidak boleh langsung menjadi metode pertama. Urutan harus:

1. validasi lingkungan fisik;
2. baseline non-RL;
3. SAC Basic;
4. SAC + Forecast;
5. SAC + LSTM;
6. SACSI Full;
7. expanded multi-seed;
8. statistik final;
9. penulisan hasil dan kesimpulan.

## 2.3 Fairness lock
Untuk perbandingan antar-controller, komponen berikut harus sama:

- Virtual Garden Core;
- parameter tanah dan tanaman;
- initial soil moisture;
- target soil-moisture band;
- action bounds;
- reward definition untuk keluarga SAC;
- train / validation / benchmark split;
- seeds;
- metric engine;
- evaluation period;
- logging schema.

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

Project dibagi menjadi **16 sprint**.

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

Buat satu file:

```text
master_results.csv
```

Minimal kolom:

```text
experiment_id
model
seed
split
forecast_enabled
memory_enabled
forecast_error
sequence_length
checkpoint
validation_gate
steps
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
```

Semua grafik dan tabel final harus dibuat dari `master_results.csv` atau sumber log yang dapat ditelusuri kembali.

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

## Must Have

1. Virtual Garden validated.
2. Baseline benchmark.
3. SAC Basic.
4. SAC + Forecast.
5. SAC + LSTM.
6. SACSI Full.
7. 10 matched seeds.
8. Final benchmark.
9. Statistical analysis.
10. Final dissertation figures/tables.

## Should Have

- dashboard final;
- forecast robustness;
- sequence sensitivity;
- interaction plots.

## Nice to Have

- Streamlit public deployment;
- external temporal validation;
- hyperparameter optimization besar;
- attention/gating variants.

---

# 15. RECOMMENDED EXECUTION ORDER DI GOOGLE COLAB

Gunakan notebook terpisah.

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
10_Final_Benchmark.ipynb
11_Ablation_Robustness.ipynb
12_Statistical_Analysis.ipynb
13_Final_Figures_Tables.ipynb
14_Dashboard_Final.ipynb
```

---

# 16. RECOMMENDED SPRINT TIMELINE

Jika dikerjakan intensif:

| Sprint | Fokus | Estimasi |
|---|---|---:|
| 0 | Setup | 1 hari |
| 1 | Dataset | 1–2 hari |
| 2 | Virtual Garden | 2–3 hari |
| 3 | Baseline | 2–3 hari |
| 4 | Dashboard V1 | 1–2 hari |
| 5 | SAC Basic | 3–5 hari |
| 6 | SAC Forecast | 3–5 hari |
| 7 | SAC LSTM | 4–7 hari |
| 8 | SACSI Full | 4–7 hari |
| 9 | 10-seed training | 5–10 hari compute-dependent |
| 10 | Final benchmark | 2–3 hari |
| 11 | Ablation/robustness | 3–5 hari |
| 12 | Statistics | 2–3 hari |
| 13 | Dashboard final | 1–2 hari |
| 14 | Bab Hasil | 3–5 hari |
| 15 | Conclusion/finalization | 2–4 hari |

Total realistis:

```text
sekitar 6–10 minggu
```

Jika coding dan pipeline sudah familiar, dapat dipercepat.

---

# 17. DEFINITION OF DONE PROJECT

Project dianggap selesai jika:

- [ ] Dataset pipeline terdokumentasi.
- [ ] Virtual Garden tervalidasi.
- [ ] Lima baseline non-RL selesai.
- [ ] SAC Basic selesai.
- [ ] SAC + Forecast selesai.
- [ ] SAC + LSTM selesai.
- [ ] SACSI Full selesai.
- [ ] 40 checkpoint 10-seed tersimpan.
- [ ] Benchmark 9 metode selesai.
- [ ] Ablation selesai.
- [ ] Forecast robustness selesai.
- [ ] Sequence sensitivity selesai.
- [ ] Master result table selesai.
- [ ] Statistik final selesai.
- [ ] Dashboard final selesai.
- [ ] Bab Hasil selesai.
- [ ] Novelty, limitations, conclusion selesai.
- [ ] Semua klaim sesuai bukti.
- [ ] Reproducibility package tersimpan di Drive + GitHub.

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

Mulai project mandiri dengan urutan berikut:

```text
1. Setup Drive + GitHub
2. Data audit
3. Virtual Garden
4. Non-RL baselines
5. SAC Basic
6. SAC + Forecast
7. SAC + LSTM
8. SACSI Full
9. 10-seed expansion
10. Benchmark + ablation
11. Statistics
12. Dashboard
13. Dissertation writing
```

Jangan mulai 10-seed training sebelum 1-seed smoke test dan 3-seed validation benar-benar stabil.

---

# 20. VERSION CONTROL MILESTONES

Buat Git tag setiap milestone:

```text
v0.1-data-pipeline
v0.2-virtual-garden
v0.3-baselines
v0.4-sac-basic
v0.5-sac-forecast
v0.6-sac-lstm
v0.7-sacsi-full
v0.8-ten-seed
v0.9-final-benchmark
v1.0-dissertation-final
```

Setiap tag minimal memiliki:

- code;
- config;
- test report;
- result summary;
- README update.

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

# 29. RECOMMENDED TAHAP 8 — FINALIZATION

Tahap 8 belum perlu memodifikasi model.

Fokus:

## 8A — Kesimpulan
Jawab seluruh rumusan masalah.

## 8B — Implikasi

- methodological;
- computational;
- smart irrigation;
- adaptive control.

## 8C — Novelty Statement

Gunakan novelty framework, bukan overclaim superiority.

## 8D — Limitations

Nyatakan synthetic forecast, simulation environment, seed sensitivity, retrospective benchmark, dan lack of field validation.

## 8E — Recommendations

Future research.

## 8F — Reproducibility Appendix

- GitHub commit;
- requirements;
- configs;
- checkpoints;
- master result table;
- notebook execution order.

---

# 30. SPRINT DEPENDENCY GRAPH

```text
Sprint 0 Setup
   ↓
Sprint 1 Data
   ↓
Sprint 2 Virtual Garden
   ↓
Sprint 3 Baselines
   ├────────────→ Sprint 4 Dashboard V1
   ↓
Sprint 5 SAC Basic
   ├────────────→ Sprint 6 SAC Forecast
   └────────────→ Sprint 7 SAC LSTM
                         ↓
                   Sprint 8 SACSI Full
                         ↓
                   Sprint 9 10-Seed
                         ↓
                  Sprint 10 Benchmark
                         ↓
                  Sprint 11 Ablation
                         ↓
                  Sprint 12 Statistics
                         ↓
                  Sprint 13 Dashboard Final
                         ↓
                  Sprint 14 Bab Hasil
                         ↓
                  Sprint 15 Finalization
```

---

# 31. COMPUTE STRATEGY PER SPRINT

| Sprint | CPU | GPU | Recommended Service |
|---|---:|---:|---|
| 0 Setup | ✓ | – | Colab |
| 1 Data | ✓ | – | Colab |
| 2 Virtual Garden | ✓ | – | Colab |
| 3 Baseline | ✓ | – | Colab |
| 4 Dashboard | ✓ | – | Local/Colab + Streamlit |
| 5 SAC Basic | ✓ | ✓ | Colab Pro |
| 6 SAC Forecast | ✓ | ✓ | Colab Pro |
| 7 SAC LSTM | ✓ | ✓✓ | Colab Pro+ / RunPod |
| 8 SACSI Full | ✓ | ✓✓ | Colab Pro+ / RunPod |
| 9 10-Seed | ✓ | ✓✓✓ | RunPod optional |
| 10 Benchmark | ✓ | ✓ | Colab |
| 11 Ablation | ✓ | ✓ | Colab / RunPod |
| 12 Statistics | ✓✓ | – | Colab CPU |
| 13 Dashboard | ✓ | – | Streamlit Cloud |
| 14 Writing | – | – | Local/Word |
| 15 Finalization | – | – | Local/GitHub |

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

# 38. PENUTUP MASTER PLANNING

Jika seluruh sprint dikerjakan ulang secara mandiri, urutan utama yang harus selalu dijaga adalah:

```text
VALIDATE ENVIRONMENT
      ↓
ESTABLISH BASELINES
      ↓
BUILD SIMPLE RL
      ↓
ADD FORECAST
      ↓
ADD MEMORY
      ↓
BUILD SACSI FULL
      ↓
EXPAND SEEDS
      ↓
ABLATE
      ↓
STATISTICALLY TEST
      ↓
REPORT WITHOUT OVERCLAIM
```

Kualitas project dinilai bukan dari apakah SACSI selalu menang, melainkan dari apakah eksperimen dilakukan secara konsisten, bebas leakage, fair, multi-seed, reproducible, dan menghasilkan kesimpulan yang benar-benar mengikuti data.
