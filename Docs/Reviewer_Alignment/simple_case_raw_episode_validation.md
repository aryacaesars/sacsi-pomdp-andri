# Modul 8C — Simple-Case and Raw-Data Episode Validation

Status: **COMPLETE**  
Environment: `VirtualGardenConfig` default, field capacity `0.35`  
Action bound: `0–5 mm/hour`  
Mass-balance tolerance: `1e-8 mm`

## Data classification

- Meteorological forcing: **real/raw validation 2024 data**.
- Soil-water state and trajectory: **Virtual Garden simulated output**.
- Forecast input: **controlled synthetic forecast proxy SF-20**.
- Controller parameters: fixed defaults; **no episode-specific retuning**.
- Reward: `reward_v4` remains locked for subsequent DRL experiments; simple physical trajectories do not use reward for controller selection.

## Simple-case results

| Case | Scenario | Initial theta | Final theta | Rain | Irrigation | Result |
|---|---|---:|---:|---:|---:|---|
| C1 | Dry-down without rain/irrigation | 0.2700 | 0.2157 | 0 mm | 0 mm | Pass |
| C2 | Rainfall pulse | 0.2700 | 0.2761 | 10 mm | 0 mm | Pass |
| C3 | Irrigation pulse | 0.2700 | 0.2595 | 0 mm | 5 mm | Pass |
| C4 | Near-upper-band protection | 0.3190 | 0.3122 | 0 mm | 0 mm | Pass |
| C5 | Below-target recovery | 0.2000 | 0.2599 | 0 mm | 20 mm | Pass |
| C6 | Heavy-rain suppression | 0.2000 | 0.2650 | 25 mm | 0 mm | Pass |

C3 memeriksa peningkatan theta tepat setelah pulse; nilai akhir 24 jam juga memuat evapotranspirasi setelah pulse. Semua action bounded, semua nilai finite, dan residual neraca massa maksimum memenuhi `<= 1e-8 mm`.

## Locked raw episodes

| Episode | Period | Hours | Rain |
|---|---|---:|---:|
| DRY | 16–29 Apr 2024 | 336 | 0.3 mm |
| WET | 27 Nov–10 Dec 2024 | 336 | 419.4 mm |
| MIXED | 17–30 Dec 2024 | 336 | 77.4 mm |

Setiap episode dijalankan dari initial theta dan config yang sama memakai:

1. No Irrigation;
2. Threshold-Based;
3. Rule-Based Forecast-Aware.

Total output adalah sembilan trajectory groups (`3 episodes × 3 controllers`), masing-masing dengan action bound, finite-output, dan mass-balance audit.

## Main observations

- DRY: No Irrigation hanya mencapai `24.11%` Time in Target; kedua controller aktif mengairi `65 mm` dan mencapai `100%` pada episode ini.
- WET: Semua controller menahan irigasi; hujan `419.4 mm` mendominasi trajectory sehingga Time in Target hanya `5.06%` akibat surplus.
- MIXED: Semua controller juga menahan irigasi; Time in Target `61.61%`.
- Maximum raw-episode mass-balance residual adalah sekitar `2.84e-14 mm`.

Hasil ini adalah sanity evidence pada episode terpilih secara objektif, bukan bukti superiority controller.

## Acceptance gate 8C

- [x] Action selalu 0–5 mm/hour.
- [x] Mass-balance residual <= 1e-8 mm.
- [x] Tidak ada NaN/Inf.
- [x] Environment/config sama.
- [x] Tidak ada episode-specific retuning.
- [x] DRY/WET/MIXED terdokumentasi.
- [x] Enam simple cases memberikan respons fisik/kontrol yang masuk akal.
