# Modul 8B — Reward Formalization and Decision Lock

Status: **COMPLETE — REWARD_V4 FINAL LOCKED**  
Selection split: **validation 2024 only**  
Benchmark 2025 accessed for reward selection: **No**

## Scalarized objective

Untuk setiap langkah:

```text
d_t = max(0.22 - theta_t, 0)
s_t = max(theta_t - 0.32, 0)
V_t = 1 jika theta_t di luar target band; selainnya 0

L_t = 100(2 d_t + s_t)
      + 0.01 I_t
      + 0.01 |I_t - I_(t-1)|
      + 2.00 V_t

r_t = 2.00 - L_t
```

Offset konstan `2.00` tidak mengubah ranking policy pada episode dengan horizon tetap. Bentuk ini mempertahankan interpretasi target-band bonus dari implementasi historis.

## Alasan fisik dan kontrol

| Term | Weight terkunci | Alasan |
|---|---:|---|
| Tracking `100(2d+s)` | 100; deficit ratio 2:1 | Defisit air lebih dapat dikontrol dan lebih diprioritaskan daripada surplus akibat hujan |
| Water `I_t` | 0.01 | Mencegah irigasi berlebih dan mengukur trade-off penggunaan air |
| Smoothness `|delta I|` | 0.01 | Mengurangi perubahan aktuator yang tidak perlu |
| Violation `V_t` | 2.00 | Memberi tekanan eksplisit agar kelembapan berada pada target band 0.22–0.32 |

Semua term dicatat sebagai total, mean, dan maximum contribution pada setiap run. Dengan demikian pengaruh skala term dapat diaudit langsung pada `reward_ablation_results.csv` dan `reward_weight_sensitivity.csv`.

## Eksperimen

- Ablation: `R-A`, `R-B`, `R-C`, `R-D`; masing-masing seeds 11, 22, 33.
- Sensitivity: `wI multiplier = {0.5, 1.0, 2.0}` × `wV multiplier = {0.5, 1.0, 2.0}`; masing-masing tiga seeds.
- Budget: 20 episode × 336 jam = 6,720 environment interactions per run.
- Total output: 12 ablation rows dan 27 sensitivity rows.
- Konfigurasi default `R-D / wI=1.0 / wV=1.0` dipakai ulang di kedua tabel sehingga hanya 36 training unik diperlukan.

## Decision

Reward historis `reward_v2` gagal retention gate:

| Candidate | Mean Time in Target | Mean irrigation | Pareto |
|---|---:|---:|---|
| reward_v2: `wI=1.0, wV=1.0` | 58.1360% | 807.0041 mm | Dominated |
| reward_v4: `wI=0.5, wV=2.0` | 58.7128% | 580.5137 mm | Non-dominated |

Keputusan final:

```text
REVISE REWARD
FINAL LOCK reward_v4
water_weight = 0.01
violation_weight = 2.00
```

Pemilihan menggunakan physical metrics: kandidat dengan mean Time in Target tertinggi dipilih dan water menjadi tie-breaker. Cumulative reward tidak digunakan untuk memilih konfigurasi.

Tiga kandidat (`reward_v2`, provisional `reward_v4`, dan `wI=1.0/wV=2.0`) dikonfirmasi pada 10 matched seeds. Kandidat harus Pareto non-dominated dan memiliki SD Time in Target <= 10 pp. Di antara kandidat dalam 0.5 pp dari target mean tertinggi, dipilih kandidat dengan mean irrigation terendah.

| 10-seed candidate | Mean Time in Target | Mean irrigation | Pareto |
|---|---:|---:|---|
| `reward_v4`: `wI=0.5, wV=2.0` | 56.8283% | 613.4448 mm | Non-dominated |
| `reward_v5` candidate: `wI=1.0, wV=2.0` | 56.0223% | 622.4924 mm | Dominated |
| `reward_v2`: `wI=1.0, wV=1.0` | 54.6847% | 719.1464 mm | Dominated |

Hanya `reward_v4` memenuhi seluruh confirmation gate, sehingga statusnya berubah dari provisional menjadi final locked.

## Provenance rule

Checkpoint dan hasil `reward_v2` tetap disimpan sebagai artefak historis dan tidak ditimpa. Seluruh model yang masuk fair benchmark baru Modul 8F–8H harus dilatih dengan `reward_v4` pada environment, split, seed, budget, dan metric engine yang sama.

## Acceptance gate 8B

- [x] Setiap reward term memiliki alasan fisik/kontrol.
- [x] Skala kontribusi setiap term direkam.
- [x] Reward ablation selesai.
- [x] Local weight sensitivity selesai.
- [x] Pareto table dan interactive plot tersedia.
- [x] Keputusan REVISE dan provisional `reward_v4` terdokumentasi.
- [x] Final `reward_v4` dikunci setelah 10-seed confirmation.
- [x] Data 2025 tidak digunakan dalam pemilihan reward.
