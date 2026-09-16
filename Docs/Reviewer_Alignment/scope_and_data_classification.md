# Modul 8A — Scope and Data Classification Lock

Status: **LOCKED FOR MODULES 8B–8H**  
Tanggal: 2026-08-18  
Sumber rencana: `Docs/MASTER_PLANNING_SACSI_POMDP_UPDATED_8A_9D.md`

## Scope penelitian

Penelitian ini mengembangkan dan mengevaluasi controller irigasi kontinu untuk **hortikultura generik** di dalam Virtual Garden. Virtual Garden adalah simulator neraca air yang konsisten untuk eksperimen terkontrol; ia bukan digital twin tanaman, kebun, atau perangkat IoT tertentu.

Ruang lingkup evaluasi adalah simulasi retrospektif berbasis forcing meteorologi 2021–2025. Training memakai 2021–2023, pemilihan reward/checkpoint memakai 2024, dan 2025 hanya dibuka sebagai **retrospective final benchmark**. Field validation has not been performed. Karena itu, klaim efektivitas lapangan, deployment real-time, hasil panen, energi pompa, dan generalisasi lintas lokasi berada di luar scope.

Target kelembapan tanah dikunci pada `0.22 <= theta <= 0.32 m3/m3`, aksi irigasi pada `0–5 mm/hour`, dan primary endpoint pada `Time in Target (%)`.

## Klasifikasi data dan evidence

| Artefak | Kelas | Peran | Batas klaim |
|---|---|---|---|
| `Historical Weather 2021-2025.csv` dan `data_clean.csv` | Real/raw meteorological observations | Forcing cuaca untuk simulator dan split eksperimen | Nyata sebagai data meteorologi; bukan pengukuran soil moisture atau respons kebun lapangan |
| `Historical Forecast 2021-2025.csv` dan `forecast_clean.csv` | Archived/raw forecast dataset yang diaudit | Sumber data tersedia, tetapi bukan input forecast terkunci pada controller final | Tidak disebut sebagai as-issued operational forecast tanpa audit provenance tambahan |
| `synthetic_forecast_sf20.csv` | Controlled synthetic forecast proxy | Input h+1 precipitation, ET0, dan temperature untuk SAC + Forecast/SACSI; diturunkan secara kausal per tahun dengan perturbasi terkontrol | Bukan forecast operasional arsip; robustness hanya berlaku untuk protokol SF10/SF20/SF30 |
| `theta`, runoff, drainage, deficit, surplus, dan mass-balance terms | Virtual Garden simulated state/output | Respons fisik simulator terhadap cuaca dan aksi | Bukti konsistensi numerik/simulasi, bukan field measurement |
| Aksi controller, reward, trajectory, dan benchmark metrics | Simulation-derived experimental evidence | Membandingkan policy/controller pada common environment | Berlaku untuk konfigurasi, split, dan benchmark yang dikunci |
| Checkpoint model | Learned simulation policy | Parameter policy hasil training pada Virtual Garden | Bukan model yang telah divalidasi pada aktuator/kebun nyata |

## Formulasi POMDP yang dikunci

Simulator memiliki state internal yang berevolusi sebagai:

```text
x_(t+1) = f(x_t, a_t, w_t)
```

Controller tidak menerima seluruh `x_t`. Ia menerima observation terbatas:

```text
o_t = h(x_t, w_t)
```

History kausal 24 jam dienkode sebagai:

```text
z_t = LSTM(o_(t-k:t))
```

Context SACSI adalah:

```text
c_t = [o_t, z_t, w_hat_(t+1)]
```

Partial observability berasal dari keterbatasan observation terhadap seluruh kondisi/proses laten tanah dan ketidakpastian forcing mendatang. History dan forecast diperlakukan sebagai informasi pembentuk representasi context, bukan sebagai bukti otomatis bahwa performa meningkat.

## Argumentasi metode

- **Virtual Garden** menyediakan common environment, neraca massa yang dapat diaudit, dan eksperimen berulang tanpa mengklaim realisme lapangan penuh.
- **DDPG** adalah pembanding deterministic actor–single critic yang sederhana untuk continuous control.
- **TD3** memisahkan pengaruh twin critics, clipped target, target-policy smoothing, dan delayed actor update dari context POMDP.
- **SAC Basic** menjadi anchor stochastic off-policy karena action kontinu, replay efficiency, twin critics, dan entropy regularization. Mekanisme internal algoritma tidak disamakan secara artifisial dalam fairness audit.
- **Forecast h+1** menguji anticipatory context dengan error yang terkontrol dan eksplisit.
- **LSTM** menguji temporal memory ketika current observation tidak cukup merepresentasikan dinamika laten.
- **SACSI-POMDP** mengintegrasikan current observation, history representation, dan forecast proxy; nilai kontribusinya harus ditentukan dari ablation, intervention, matched seeds, dan inferensi.

## Optimasi dan efisiensi

**Optimasi** adalah proses training dan pemilihan policy untuk memaksimalkan scalarized multi-objective reward di bawah protocol yang dikunci. **Efisiensi** adalah outcome fisik, bukan sinonim optimasi. Efisiensi dilaporkan melalui total irrigation, Time in Target, violation/deficit, RMSE band, action smoothness, serta Pareto Water versus Time in Target. Cumulative reward tidak menjadi dasar tunggal pemilihan reward atau ranking final.

## Fair-comparison hierarchy

1. Validasi numerik dan simple-case Virtual Garden.
2. Non-RL controllers sebagai trajectory references.
3. DDPG–TD3–SAC dengan observation 8-D dan common reward/environment/split/budget/seeds.
4. SAC-family `F0M0`, `F1M0`, `F0M1`, `F1M1` untuk kontribusi forecast dan memory.
5. DDPG–TD3–SAC–SACSI pada 10 matched seeds untuk benchmark konfirmatori.

Checkpoint hanya dipilih pada validation 2024. Tidak ada retuning, penggantian failed seed, atau checkpoint reselection setelah membuka 2025.

## Claim guards

```text
Framework validity
    != context activation
    != performance benefit
    != statistical superiority
```

- Aktivasi branch memerlukan evidence arsitektur/gradient/intervention.
- Manfaat performa memerlukan delta metric pada matched conditions.
- Statistical superiority memerlukan uji seed-level sesuai rencana, arah efek yang mendukung, koreksi multiplicity, effect size, dan interval kepercayaan.
- Hasil nol/negatif dilaporkan apa adanya.
- H4 tidak dipaksa diterima.
- Klaim external field effectiveness tidak diizinkan tanpa eksperimen lapangan baru.

## Acceptance gate 8A

- [x] RM1–RM4 dan T1–T4 one-to-one.
- [x] Seluruh tujuan memiliki indikator dan decision rule terukur.
- [x] Argumentasi SAC dan POMDP dikunci.
- [x] Scope hortikultura generik dinyatakan eksplisit.
- [x] Real/raw, simulated, dan synthetic proxy dibedakan.
- [x] Fair-comparison hierarchy dikunci.
- [x] Tidak ada klaim field validation.

Artefak pendamping: `research_question_objective_map.csv`, `hypothesis_map.csv`, dan `reviewer_alignment_matrix.csv`.
