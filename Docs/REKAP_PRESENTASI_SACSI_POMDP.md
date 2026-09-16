# Rekap Presentasi SACSI-POMDP

Dokumen ini adalah bahan presentasi Bahasa Indonesia yang disusun dari evidence beku `v1.0-dissertation-freeze`. Angka utama berasal dari `Results/Confirmatory_10Seed`, bukan dari perkiraan atau mock data.

## Identitas penelitian

**Judul:** Pengembangan Model Soft Actor-Critic for Smart Irrigation dalam Kerangka POMDP untuk Irigasi Cerdas Adaptif

**Satu kalimat kontribusi:**

SACSI-POMDP adalah kerangka continuous smart-irrigation control yang menggabungkan kondisi saat ini, temporal memory, dan konteks forecast dalam Virtual Garden, kemudian mengevaluasinya melalui ablation terkontrol dan benchmark konfirmatori 10 matched seeds.

**Versi sangat singkat untuk pembukaan:**

> Penelitian ini mengembangkan dan menguji controller irigasi berbasis SAC dalam kerangka POMDP. Tujuan utamanya adalah menjaga kelembapan tanah di target 0,22–0,32 m³/m³ dengan penggunaan air dan perubahan aksi yang tetap terkendali. Hasil akhir menunjukkan pipeline SACSI memiliki Time in Target lebih tinggi daripada SAC, TD3, dan DDPG pada benchmark simulasi retrospektif 2025, tetapi klaimnya dibatasi pada pipeline dan simulator yang diuji.

## Struktur presentasi 12–15 menit

| Slide | Topik | Durasi |
|---:|---|---:|
| 1 | Judul dan inti penelitian | 30 detik |
| 2 | Latar belakang masalah | 45 detik |
| 3 | Tujuan dan pertanyaan penelitian | 60 detik |
| 4 | Ruang lingkup dan klasifikasi data | 60 detik |
| 5 | Virtual Garden dan formulasi POMDP | 60 detik |
| 6 | Controller dan arsitektur SACSI | 75 detik |
| 7 | Reward multi-objective | 60 detik |
| 8 | Protokol eksperimen dan fairness | 75 detik |
| 9 | Validasi simulator | 45 detik |
| 10 | Hasil benchmark utama | 90 detik |
| 11 | Statistik dan kontribusi context | 90 detik |
| 12 | Pembacaan baseline threshold | 60 detik |
| 13 | Demo dashboard | 2–3 menit |
| 14 | Kontribusi, keterbatasan, dan future work | 75 detik |
| 15 | Kesimpulan | 45 detik |

---

## Slide 1 — Judul

### Isi slide

- Judul penelitian.
- Nama, program studi, dan pembimbing.
- Subjudul: **Continuous smart-irrigation control dalam Virtual Garden berbasis SAC dan POMDP**.

### Narasi

> Penelitian ini berfokus pada pengambilan keputusan irigasi kontinu. Controller menerima kondisi lingkungan, menentukan jumlah irigasi antara 0 sampai 5 mm per jam, dan berusaha mempertahankan kelembapan tanah di dalam target band.

## Slide 2 — Latar belakang masalah

### Isi slide

- Irigasi terlalu sedikit menyebabkan deficit atau under-irrigation.
- Irigasi terlalu banyak meningkatkan surplus, drainage, dan pemborosan air.
- Kondisi tanah tidak seluruhnya teramati dari satu observation.
- Cuaca mendatang juga tidak diketahui secara sempurna.

### Narasi

> Masalah irigasi bukan sekadar meminimalkan air. Controller yang memakai nol air memang paling hemat, tetapi gagal menjaga tanaman di kondisi target. Karena itu performa harus membaca Time in Target, penggunaan air, violation, deficit, dan kestabilan aksi secara bersamaan.

## Slide 3 — Tujuan dan pertanyaan penelitian

### Isi slide

1. Memvalidasi simulator Virtual Garden dan reward untuk kontrol irigasi.
2. Membandingkan DDPG, TD3, dan SAC dengan protokol yang fair.
3. Menguji kontribusi forecast dan temporal memory terhadap SAC.
4. Menguji pipeline SACSI terhadap SAC, TD3, dan DDPG pada 10 matched seeds.

### Narasi

> Penelitian disusun bertahap. Simulator dan reward divalidasi lebih dahulu. Setelah itu comparator continuous-control diuji secara fair. Kontribusi forecast dan memory dipisahkan melalui desain factorial, lalu pipeline final diuji pada sepuluh seed yang sama.

## Slide 4 — Ruang lingkup dan klasifikasi data

### Isi slide

| Komponen | Status |
|---|---|
| Meteorologi 2021–2025 | Data real/raw observasional |
| Kelembapan tanah, runoff, drainage | Hasil simulasi Virtual Garden |
| Forecast SF-20 h+1 | Controlled synthetic forecast proxy |
| Field trial | Belum dilakukan |
| Model daya pompa | Belum tersedia |

### Narasi

> Hal pentingnya adalah tidak semua variabel merupakan pengukuran lapangan. Meteorologi berasal dari data observasional, tetapi soil state merupakan keluaran simulator. Forecast yang dipakai juga merupakan proxy terkontrol, bukan archived operational forecast. Karena belum ada model daya pompa yang terkalibrasi, penelitian tidak mengklaim penghematan energi.

## Slide 5 — Virtual Garden dan formulasi POMDP

### Isi slide

- Target kelembapan: **0,22–0,32 m³/m³**.
- Aksi irigasi kontinu: **0–5 mm/jam**.
- State berkembang melalui neraca air: hujan, irigasi, evapotranspirasi, runoff, dan drainage.
- Controller hanya menerima observation, bukan seluruh state laten.

```text
Meteorologi + observation + history + forecast
                     ↓
                 Controller
                     ↓
              Irigasi 0–5 mm/jam
                     ↓
              Virtual Garden
                     ↓
       Soil moisture + reward + metrik
```

### Narasi

> Masalah diformulasikan sebagai POMDP karena policy tidak mengetahui seluruh kondisi laten tanah dan forcing masa depan secara sempurna. History digunakan untuk merangkum dinamika temporal, sedangkan forecast memberi konteks mengenai kemungkinan hujan mendatang.

## Slide 6 — Controller dan arsitektur SACSI

### Isi slide

**Baseline non-RL:**

- No Irrigation.
- Fixed Schedule.
- Threshold-Based.
- Rule-Based Forecast-Aware.
- Fuzzy Controller.

**Controller RL:**

- DDPG: deterministic actor dan single critic.
- TD3: twin critics, target smoothing, dan delayed policy update.
- SAC: stochastic actor, twin critics, dan entropy tuning.
- SAC + Forecast.
- SAC + LSTM.
- SACSI Full: current observation + forecast + LSTM memory.

### Narasi

> DDPG dan TD3 dipakai sebagai tangga pembanding continuous control sebelum SAC. SAC kemudian menjadi anchor untuk menguji tambahan konteks. Desain factorial terdiri dari SAC Basic, SAC dengan forecast, SAC dengan LSTM, dan SACSI Full yang menggabungkan keduanya.

## Slide 7 — Reward multi-objective

### Isi slide

Reward final yang dikunci adalah `reward_v4`:

```text
r_t = 2 − [100(2d_t + s_t) + 0,01I_t + 0,01|ΔI_t| + 2V_t]
```

Keterangan:

- `d_t`: jarak kondisi di bawah target band.
- `s_t`: jarak kondisi di atas target band.
- `I_t`: jumlah irigasi.
- `|ΔI_t|`: perubahan aksi irigasi.
- `V_t`: penalti pelanggaran target.

### Narasi

> Reward memberi penalti lebih besar pada deficit karena kekurangan air diprioritaskan dibanding surplus. Reward juga menghukum penggunaan air, perubahan aksi, dan violation. Reward dipilih pada validasi 2024 menggunakan aturan Pareto dan stabilitas, tanpa membuka benchmark 2025.

**Angka validasi reward:** Time in Target **56,828 ± 5,152%**, irigasi **613,445 ± 146,976 mm**, 10 seeds, Pareto non-dominated.

## Slide 8 — Protokol eksperimen dan fairness

### Isi slide

```text
2021–2023                2024                         2025
Training        Reward/checkpoint selection   Final retrospective benchmark
```

- Sepuluh matched seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, dan 110.
- Environment, observation, action bounds, reward, split, seeds, metric engine, dan checkpoint rule dikunci.
- Failed validation seeds tetap dipertahankan.
- Tidak ada retraining atau reselection setelah benchmark 2025 dibuka.

### Narasi

> Fairness berarti protokol eksternal dibuat sama, bukan mekanisme algoritmanya dipaksa identik. DDPG tetap single critic, TD3 tetap twin critic dengan policy delay, dan SAC tetap stochastic dengan entropy tuning.

### Batas fairness yang wajib disebut

> Comparator non-context menggunakan 6.720 total interactions. Context variants menggunakan SAC anchor 6.720 interactions ditambah 6.720 adaptation interactions. Karena itu hasil mendukung locked training pipeline, bukan klaim arsitektur dengan total budget yang sama.

## Slide 9 — Validasi simulator

### Isi slide

- **6/6 simple cases lulus**.
- Kasus mencakup pengeringan, pulse hujan, pulse irigasi, batas atas, pemulihan dari deficit, dan antisipasi hujan.
- Semua aksi bounded dan semua output finite.
- Residual neraca massa maksimum episode raw-data: **2,842 × 10⁻¹⁴ mm**.

### Narasi

> Validasi ini menunjukkan konsistensi numerik dan respons fisik dasar simulator. Hasil tersebut tidak boleh disebut field validation atau validasi digital twin tanaman tertentu.

## Slide 10 — Hasil utama benchmark 10 seeds

### Isi slide

Primary endpoint: **Time in Target (%)**.

| Pipeline | Time in Target mean ± SD | Irigasi mean ± SD |
|---|---:|---:|
| **SACSI-POMDP** | **55,018 ± 1,927%** | **372,197 ± 42,601 mm** |
| SAC | 54,116 ± 2,082% | 394,098 ± 87,313 mm |
| TD3 | 41,850 ± 12,696% | 176,153 ± 186,339 mm |
| DDPG | 41,279 ± 12,798% | 151,509 ± 167,889 mm |

### Narasi

> SACSI memperoleh rata-rata Time in Target tertinggi pada empat pipeline konfirmatori. DDPG dan TD3 menggunakan air lebih sedikit, tetapi juga memiliki Time in Target lebih rendah, deficit lebih besar, dan variasi antar-seed yang tinggi. Jadi penggunaan air minimum saja tidak cukup untuk menyatakan efisiensi.

## Slide 11 — Statistik dan kontribusi context

### Isi slide

Friedman omnibus: **χ²(3) = 15,064; p = 0,001763**.

| Planned contrast | Selisih | 95% bootstrap CI | Exact-Holm p |
|---|---:|---:|---:|
| SACSI − SAC | +0,902 pp | [0,336; 1,531] | 0,046875 |
| SACSI − TD3 | +13,168 pp | [5,402; 20,854] | 0,046875 |
| SACSI − DDPG | +13,740 pp | [5,660; 21,105] | 0,046875 |

Hasil desain factorial:

| Efek | Mean | 95% bootstrap CI | Exact-Holm p | Keputusan |
|---|---:|---:|---:|---|
| Forecast | −0,107 pp | [−0,632; 0,422] | 0,964844 | Tidak didukung |
| Memory | +1,009 pp | [0,579; 1,438] | 0,017578 | Didukung |
| Forecast × Memory | +1,501 pp | [−0,865; 4,771] | 0,964844 | Tidak didukung |

### Narasi

> Hasil pentingnya tidak semuanya positif. Memory memberikan efek yang didukung, sedangkan forecast secara mandiri dan interaksi forecast dengan memory tidak didukung. Null result ini tetap dipertahankan sebagai temuan ilmiah. Selisih SACSI terhadap SAC signifikan tetapi tipis, hanya 0,902 percentage points, sehingga tidak disebut sebagai peningkatan operasional yang besar.

## Slide 12 — Mengapa Threshold-Based terlihat lebih tinggi?

### Isi slide

Pada benchmark deskriptif sembilan metode:

- Threshold-Based: Time in Target **60,126%**, irigasi **290 mm**.
- SACSI Full historical run family: Time in Target **55,266 ± 4,001%**, irigasi **359,831 ± 41,705 mm**.

### Narasi yang aman

> Threshold-Based merupakan baseline yang sangat kuat dalam Virtual Garden generik dan bahkan menghasilkan nilai deskriptif lebih tinggi. Karena itu penelitian ini tidak mengklaim SACSI mengalahkan seluruh heuristic controller. Kontribusi SACSI adalah pipeline continuous-control berbasis context yang diuji secara modular dan lebih baik daripada comparator RL yang ditetapkan pada benchmark konfirmatori. Hasil threshold juga menunjukkan bahwa kompleksitas machine learning harus dibenarkan oleh kondisi yang memang membutuhkan adaptasi dan partial observability.

### Jika ditanya “Kalau threshold lebih bagus, buat apa ML?”

> Threshold efektif karena simulator dan target band masih relatif terstruktur. ML diteliti untuk controller kontinu yang dapat mengintegrasikan lebih banyak context dan beradaptasi pada dinamika yang lebih kompleks. Namun berdasarkan evidence sekarang, threshold tetap benchmark praktis yang sangat kompetitif; keunggulan operasional ML belum boleh diklaim sebelum field validation.

## Slide 13 — Demo dashboard

### Jalankan

```powershell
cd D:\ARYA\SACSI_Dissertation
python -m streamlit run Dashboard\app.py
```

### Alur demo 2–3 menit

1. Pilih **Bahasa Indonesia**.
2. Buka **Kebun Virtual**.
3. Jelaskan target band 0,22–0,32 dan arti warna meter.
4. Pilih mode **Bandingkan 2–4 metode**.
5. Bandingkan No Irrigation, Threshold-Based, dan SACSI Full.
6. Geser replay hour untuk menunjukkan kelembapan dan aksi irigasi dari waktu ke waktu.
7. Buka **Lab Reward** untuk menunjukkan reward terkunci.
8. Buka **Benchmark DRL Fair** untuk DDPG, TD3, SAC, dan SACSI.
9. Buka **Statistik Konfirmatori 10-Seed** untuk hasil final.
10. Buka **Matriks Evidence Reviewer** atau **Reproducibility & Provenance** sebagai bukti keterlacakan.

### Catatan penting saat demo

- Kebun Virtual adalah visualisasi replay, bukan evidence statistik baru.
- Klaim final harus diambil dari halaman konfirmatori/final evidence.
- Jangan menyebut trajectory tanah sebagai sensor tanah real.

## Slide 14 — Kontribusi, keterbatasan, dan future work

### Kontribusi

1. Virtual Garden modular dengan gate konsistensi numerik dan respons fisik.
2. Reward multi-objective yang dipilih melalui ablation, sensitivity, Pareto, dan 10-seed validation.
3. Benchmark DDPG–TD3–SAC dengan fairness lock.
4. Desain factorial untuk memisahkan kontribusi forecast dan memory.
5. Benchmark konfirmatori 10 matched seeds dengan inferensi berpasangan.
6. Dashboard evidence, claim-to-evidence matrix, dan reproducibility freeze.

### Keterbatasan

- Belum ada field trial atau hardware-in-the-loop.
- Soil state merupakan hasil simulasi.
- Forecast merupakan controlled synthetic proxy.
- Hanya satu setting meteorologi retrospektif.
- Budget efektif context dan non-context tidak sama.
- Tidak tersedia model daya pompa; **energy savings tidak dilaporkan**.

### Future work

- Archived as-issued operational forecast.
- Kalibrasi multi-lokasi dan jenis tanah/tanaman.
- Equal-total-budget retraining.
- Model pompa terkalibrasi untuk metrik energi.
- Hardware-in-the-loop dan field trial.
- Integrasi sensor, aktuator, fail-safe, dan monitoring real-time.

## Slide 15 — Kesimpulan

### Isi slide

1. Virtual Garden dan reward_v4 lulus gate yang dikunci.
2. Benchmark DDPG–TD3–SAC memenuhi fairness audit.
3. Locked SACSI pipeline menghasilkan Time in Target lebih tinggi daripada SAC, TD3, dan DDPG pada benchmark simulasi 2025.
4. Memory effect didukung; forecast dan forecast × memory tidak didukung.
5. Threshold-Based tetap menjadi baseline praktis yang sangat kompetitif.
6. Hasil belum membuktikan efektivitas lapangan, kesiapan deployment, atau penghematan energi.

### Kalimat penutup

> Kesimpulan utama penelitian ini bukan bahwa satu algoritma selalu paling baik, tetapi bahwa SACSI menyediakan pipeline kontrol irigasi berbasis context yang dapat diuji secara transparan. Evidence mendukung memory dan performa locked pipeline terhadap comparator RL, sekaligus menunjukkan batas manfaat forecast dan pentingnya baseline sederhana.

---

## Cheat sheet angka penting

| Item | Nilai |
|---|---:|
| Target band | 0,22–0,32 m³/m³ |
| Aksi irigasi | 0–5 mm/jam |
| Training | 2021–2023 |
| Validasi reward/checkpoint | 2024 |
| Benchmark final | 2025 retrospektif |
| Matched seeds | 10 |
| SACSI Time in Target | 55,018 ± 1,927% |
| SAC Time in Target | 54,116 ± 2,082% |
| TD3 Time in Target | 41,850 ± 12,696% |
| DDPG Time in Target | 41,279 ± 12,798% |
| SACSI − SAC | +0,902 pp |
| SACSI − TD3 | +13,168 pp |
| SACSI − DDPG | +13,740 pp |
| Exact-Holm p ketiga contrast | 0,046875 |
| Memory main effect | +1,009 pp; p = 0,017578 |
| Forecast main effect | −0,107 pp; tidak didukung |
| Forecast × Memory | +1,501 pp; tidak didukung |
| Simple cases | 6/6 lulus |
| Maximum raw-episode mass-balance residual | 2,842 × 10⁻¹⁴ mm |

## Jawaban cepat pertanyaan penguji

### “Apakah ini sudah memakai machine learning?”

Ya. DDPG, TD3, SAC, SAC + Forecast, SAC + LSTM, dan SACSI adalah model deep reinforcement learning. Baseline non-RL tetap disediakan agar peningkatan kompleksitas bisa dinilai secara jujur.

### “Apakah datanya mock?”

Tidak seluruhnya. Meteorologi adalah data real/raw, soil moisture dan komponen neraca air adalah hasil simulasi, sedangkan forecast SF-20 h+1 adalah controlled synthetic proxy.

### “Apakah SACSI terbukti superior?”

Ya, tetapi hanya untuk **locked warm-start training pipeline**, primary endpoint Time in Target, dan benchmark simulasi retrospektif 2025 terhadap SAC, TD3, dan DDPG. Ini bukan klaim universal, equal-total-budget architecture, atau efektivitas lapangan.

### “Mengapa memilih SAC?”

Karena aksi irigasi bersifat kontinu. SAC mendukung stochastic off-policy learning, twin critics, dan entropy regularization, serta cocok sebagai anchor untuk varian context.

### “Mengapa POMDP?”

Karena controller tidak melihat seluruh state laten tanah dan tidak mengetahui cuaca masa depan secara sempurna. History membantu merangkum dinamika yang tidak terlihat dari satu observation.

### “Forecast-nya ternyata tidak signifikan. Apakah model gagal?”

Tidak. Branch forecast aktif, tetapi standalone performance benefit tidak didukung pada desain final. Null result ini menunjukkan bahwa forecast proxy yang digunakan belum memberi tambahan informasi yang cukup di luar observation dan memory.

### “Mengapa DDPG/TD3 lebih hemat air?”

Penggunaan air rendah terjadi bersama Time in Target yang lebih rendah dan deficit yang lebih tinggi. Controller dapat terlihat hemat karena under-irrigation; efisiensi harus membaca air dan kualitas kontrol secara bersamaan.

### “Apakah sistem siap dipasang di kebun?”

Belum. Masih diperlukan kalibrasi tanah/tanaman, integrasi sensor dan aktuator, fail-safe, hardware-in-the-loop, operational forecast, serta field trial.

### “Di mana modelnya?”

- Implementasi model: folder `03_SAC_Basic`, `04_SAC_Forecast`, `05_SAC_LSTM`, `06_SACSI_Full`, dan source controller terkait.
- Model terlatih: folder `Checkpoints` dan package freeze.
- Evidence final: `Results/Confirmatory_10Seed`.
- Dashboard: `Dashboard/app.py`.

### “Bagaimana dengan energi?”

Penelitian melaporkan aksi dan total irigasi, tetapi belum mengubahnya menjadi energi karena belum ada kurva daya pompa, head, flow rate, dan efisiensi pompa yang terkalibrasi. Karena itu klaim energy savings sengaja tidak dirilis.

## Wording yang harus dihindari

| Jangan katakan | Katakan |
|---|---|
| “Sudah divalidasi di lapangan.” | “Simulator lulus gate numerik dan respons fisik.” |
| “Forecast meningkatkan performa.” | “Standalone forecast effect tidak didukung.” |
| “SACSI selalu paling baik.” | “Locked SACSI pipeline lebih tinggi terhadap comparator RL pada benchmark yang diuji.” |
| “Semua datanya real.” | “Meteorologi real; soil state simulated; forecast controlled synthetic proxy.” |
| “Paling sedikit air berarti paling efisien.” | “Air dibaca bersama Time in Target, violation, dan deficit.” |
| “SACSI fair dengan budget total yang sama.” | “Hasil berlaku pada locked warm-start pipelines dengan effective budget berbeda.” |
| “Sistem siap deploy.” | “Sistem masih memerlukan HIL dan field validation.” |
| “Terbukti hemat energi.” | “Metrik energi belum dilaporkan karena belum ada calibrated pump-power model.” |

## Evidence utama untuk dibuka saat ditanya

- `Results/Confirmatory_10Seed/main_10seed_results_2025.csv`
- `Results/Confirmatory_10Seed/planned_contrasts.csv`
- `Results/Confirmatory_10Seed/factorial_inference.csv`
- `Results/Confirmatory_10Seed/final_statistics_summary.json`
- `Results/Fair_DRL/fairness_audit.json`
- `Results/Reward_Validation/reward_confirmation_decision.json`
- `Results/Simple_Case_Validation/simple_case_results.csv`
- `Results/Reviewer_Defense/claim_to_evidence_matrix.csv`
- `Docs/Reviewer_Defense/one_page_defense_card.md`

## Status akhir proyek

- Modul 8A–8H: evidence dan benchmark selesai.
- Modul 9A dashboard: READY.
- Modul 9B dissertation integration: READY.
- Modul 9C defense package: READY.
- Modul 9D reproducibility freeze: PASS.
- Git tag: `v1.0-dissertation-freeze`.
- Publication package dibuat tanpa raw dataset karena izin redistribusi dataset belum tercatat.
