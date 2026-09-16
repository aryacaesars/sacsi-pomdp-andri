# SACSI-POMDP Dashboard v2 -- Reviewer Ready Revision Planning

## 1. Tujuan Revisi

Tujuan revisi ini adalah menyempurnakan aplikasi Streamlit SACSI-POMDP
agar selaras dengan:

-   Modul 8A--8H Reviewer Alignment;
-   Modul 9A--9D Final Dashboard dan Reproducibility;
-   masukan reviewer terkait fairness benchmark;
-   kebutuhan analisis superiority antar metode.

Revisi tidak melakukan pembangunan ulang sistem, tetapi menambahkan
lapisan analisis ilmiah dan memperbaiki struktur evaluasi.

------------------------------------------------------------------------

# 2. Kondisi Existing

## Komponen yang sudah tersedia

Status:

-   Virtual Garden: tersedia
-   Reward Lab: tersedia
-   Simple Case: tersedia
-   DDPG module: tersedia
-   TD3 module: tersedia
-   SAC Basic: tersedia
-   SACSI-POMDP: tersedia
-   POMDP ablation: tersedia
-   Statistics module: tersedia
-   Reviewer evidence: tersedia
-   Reproducibility structure: tersedia

## Gap yang ditemukan

1.  Benchmark utama masih mencampurkan:

    -   SAC Basic
    -   SAC + Forecast
    -   SAC + LSTM
    -   SACSI Full

2.  DDPG dan TD3 belum ditempatkan sebagai benchmark utama.

3.  Belum terdapat modul keputusan:

    -   metode superior;
    -   metode terbaik berdasarkan objective tertentu;
    -   metode Pareto dominant.

4.  Statistik benchmark utama dan ablation belum dipisahkan.

------------------------------------------------------------------------

# 3. Target Struktur Dashboard Final

Struktur akhir:

    SACSI-POMDP Dashboard v2

    ├── Research Design
    ├── Reward Lab
    ├── Simple Case Validation
    ├── Fair DRL Benchmark
    │   ├── DDPG
    │   ├── TD3
    │   ├── SAC
    │   └── SACSI
    ├── POMDP Contribution
    │   ├── SAC
    │   ├── SAC + Forecast
    │   ├── SAC + LSTM
    │   └── SACSI
    ├── Method Superiority Analysis
    ├── Statistical Confirmation
    ├── Robustness Analysis
    ├── Reviewer Evidence Matrix
    └── Reproducibility & Freeze

------------------------------------------------------------------------

# 4. Tahapan Revisi

# Sprint 1 -- Benchmark Reorganization

## Tujuan

Memisahkan benchmark utama dan ablation.

## Perubahan

Ubah primary benchmark menjadi:

    DDPG
    TD3
    SAC Basic
    SACSI-POMDP

Sementara:

    SAC + Forecast
    SAC + LSTM

dipindahkan menjadi:

    POMDP Ablation Study

## File yang perlu diperiksa

-   final_benchmark.py
-   statistics.py
-   result registry
-   dashboard fair DRL page

## Acceptance Criteria

-   Benchmark utama hanya berisi 4 metode.
-   Ablation terpisah.
-   Tidak ada pencampuran statistik.

------------------------------------------------------------------------

# Sprint 2 -- Method Superiority Analysis

## Tujuan

Membuat mekanisme keputusan metode unggul.

## Modul Baru

Tambahkan:

    Method Superiority & Decision

## Input

Metrics:

-   Time in Target
-   Total Irrigation
-   Violation Rate
-   RMSE
-   Action Smoothness

## Decision Logic

Metode superior jika:

1.  Primary metric lebih baik.

\[ `\Delta `{=tex}TIT \> 0 \]

2.  Signifikan secara statistik.

\[ p\_{Holm}\<0.05 \]

3.  Confidence interval mendukung.

\[ CI\_{95%} \> 0 \]

4.  Effect size bermakna.

\[ \|d_z\| `\geq 0.2`{=tex} \]

## Output Label

Dashboard memberikan:

-   STATISTICALLY SUPERIOR
-   DESCRIPTIVELY BETTER
-   COMPARABLE
-   INFERIOR
-   INSUFFICIENT EVIDENCE

------------------------------------------------------------------------

# Sprint 3 -- Pareto Analysis

## Tujuan

Menampilkan trade-off:

\[ `\max`{=tex}(Time in Target) \]

dan:

\[ `\min`{=tex}(Water Usage) \]

## Visualisasi

Scatter plot:

X-axis:

    Total Irrigation

Y-axis:

    Time in Target

## Analisis

Metode diberi label:

-   Pareto Dominant
-   Water Efficiency Leader
-   Moisture Control Leader

------------------------------------------------------------------------

# Sprint 4 -- Statistical Module Refinement

## Main Benchmark Statistics

Untuk:

    DDPG
    TD3
    SAC
    SACSI

Gunakan:

-   Friedman test;
-   paired comparison;
-   Holm correction;
-   Cohen dz;
-   bootstrap CI.

## POMDP Ablation Statistics

Untuk:

    SAC
    SAC Forecast
    SAC LSTM
    SACSI

Gunakan:

Forecast × Memory:

    F0M0
    F1M0
    F0M1
    F1M1

Analisis:

-   Forecast effect;
-   Memory effect;
-   Interaction effect.

------------------------------------------------------------------------

# Sprint 5 -- Simple Case Enhancement

## Tambahan Visualisasi

Tambahkan:

1.  Dry-down response

2.  Rainfall pulse

3.  Irrigation pulse

4.  Moisture recovery

5.  Target band protection

## Tujuan

Menjawab reviewer:

"Mulai dari case sederhana."

------------------------------------------------------------------------

# Sprint 6 -- Reward Decision Layer

## Tambahan Dashboard

Reward Lab ditambah:

-   reward configuration comparison;
-   Pareto reward;
-   selected reward;
-   validation-only decision.

Output:

    Reward v4 selected

    Reason:
    Pareto efficient
    Validation 2024 only
    No benchmark leakage

------------------------------------------------------------------------

# Sprint 7 -- Reviewer Evidence Upgrade

Tambahkan halaman:

    Reviewer Evidence Matrix

Format:

  Reviewer   Comment              Evidence             Status
  ---------- -------------------- -------------------- --------
  Wayan      Fair benchmark       DDPG-TD3-SAC-SACSI   Ready
  Hiron      Raw data             Dataset provenance   Ready
  Oka        Reward formulation   Reward Lab           Ready

------------------------------------------------------------------------

# Sprint 8 -- Final Reproducibility

Tambahkan:

-   dataset hash;
-   model checkpoint hash;
-   configuration version;
-   reward version;
-   experiment timestamp;
-   git commit.

Status:

    PRE-FREEZE
    RESULT-FREEZE
    DISSERTATION-FREEZE
    PUBLICATION-RELEASE

------------------------------------------------------------------------

# 5. Prioritas Implementasi

Urutan pengerjaan:

    1. Benchmark separation
            ↓
    2. Method superiority engine
            ↓
    3. Pareto analysis
            ↓
    4. Statistics refinement
            ↓
    5. Simple case visualization
            ↓
    6. Reward decision
            ↓
    7. Reviewer matrix
            ↓
    8. Final freeze

------------------------------------------------------------------------

# 6. Perubahan File Utama

Kemungkinan file yang berubah:

    src/evaluation/final_benchmark.py

    src/evaluation/statistics.py

    Dashboard/views/fair_drl.py

    Dashboard/views/statistics.py

    Dashboard/views/final_evidence.py

    Dashboard/views/new_superiority.py

    src/evaluation/pareto.py

    src/evaluation/decision_engine.py

------------------------------------------------------------------------

# 7. Acceptance Criteria Final

Dashboard dinyatakan siap jika:

-   [ ] DDPG-TD3-SAC-SACSI menjadi benchmark utama.
-   [ ] SAC Forecast/LSTM menjadi ablation.
-   [ ] Statistical superiority tersedia.
-   [ ] Pareto analysis tersedia.
-   [ ] Tidak ada klaim "best model" tanpa statistik.
-   [ ] Reward decision terdokumentasi.
-   [ ] Reviewer evidence lengkap.
-   [ ] Reproducibility metadata tersedia.
-   [ ] Dashboard dapat menghasilkan kesimpulan otomatis berbasis
    evidence.

------------------------------------------------------------------------

# 8. Kesimpulan

Revisi dashboard bukan pembangunan ulang.

Fokus utama:

\[ `\boxed{
Existing\ SACSI-POMDP
+
Statistical\ Decision\ Layer
+
Reviewer\ Alignment
}`{=tex} \]

Target akhir:

Dashboard tidak hanya menunjukkan hasil eksperimen, tetapi mampu
menjawab:

"Metode mana yang lebih unggul, pada aspek apa, dan apakah keunggulan
tersebut didukung bukti statistik?"
