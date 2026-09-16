# SACSI-POMDP

Reproducible research project for continuous smart-irrigation control in a Virtual Garden. The final comparison covers DDPG, TD3, SAC, and SACSI-POMDP on ten matched seeds using the locked `reward_v4` protocol.

## Scientific scope

- Training: meteorological forcing from 2021–2023.
- Reward and checkpoint selection: validation 2024 only.
- Final evaluation: retrospective Virtual Garden benchmark 2025.
- Primary endpoint: Time in Target (%), target band 0.22–0.32 m³/m³.
- Forecast input: SF-20 h+1 controlled synthetic forecast proxy.
- Soil moisture, runoff, drainage, and trajectories are simulated outputs.
- No field-effectiveness, crop-yield, deployment-readiness, or energy-saving claim is released.

The supported superiority statement applies to the locked warm-start training pipelines. Context variants use a 6,720-interaction SAC anchor plus 6,720 adaptation interactions, so the results are not an equal-total-budget architecture comparison.

## Environment

The frozen development environment used Python 3.14.3, PyTorch 2.13.0+cu130, and CUDA 13.0. GPU is required for reproducing the locked training protocol; evidence validation and the dashboard can run on CPU.

```powershell
cd D:\ARYA\SACSI_Dissertation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda users can run:

```powershell
conda env create -f environment.yml
conda activate sacsi-pomdp
```

## Required local data

Place the two user-supplied input files in `00_Dataset/`:

```text
Historical Weather 2021-2025.csv
Historical Forecast 2021-2025.csv
```

The public reproducibility package contains hashes and provenance metadata, not the raw files. Upstream provider and redistribution license were not recorded in the project materials; see `LICENSE_or_data_notice.md` before distributing data.

## Verify the frozen evidence

```powershell
python scripts\build_dashboard_release.py
python scripts\build_dissertation_release.py
python scripts\build_defense_package.py
python scripts\build_reproducibility_release.py
python -m pytest -p no:cacheprovider tests -q
```

Every builder must report a ready state. The final test count is recorded in the hand-off message and release metadata. The 9D builder verifies source hashes, 40 main and 40 factorial rows, 40 main checkpoint slots, all checkpoint hashes, synthetic-evidence guards, and reconciliation of dashboard/dissertation/defense outputs.

## Run the dashboard

```powershell
python -m streamlit run Dashboard\app.py
```

Open `http://localhost:8501`. The dashboard stops with `NOT READY` when a registered source is missing or its SHA-256 digest changes.

## Reproduce individual protocols

Smoke checks:

```powershell
python scripts\run_reward_validation.py --smoke
python scripts\run_simple_case_validation.py
python scripts\run_fair_drl_benchmark.py --smoke
python scripts\run_pomdp_ablation.py --smoke
python scripts\run_confirmatory_benchmark.py --smoke
```

Confirmatory workflow modes:

```powershell
python scripts\run_confirmatory_benchmark.py --train-only
python scripts\run_confirmatory_benchmark.py --evaluate-only
python scripts\run_confirmatory_benchmark.py --statistics-only
```

Training is resume-safe and uses validation-only checkpoint selection. Do not retrain, replace failed seeds, or reselect checkpoints after opening the 2025 benchmark when reproducing the frozen study.

## Source-of-truth hierarchy

```text
Results/Confirmatory_10Seed
  → Results/Dashboard
  → Results/Dissertation_Evidence
  → Results/Reviewer_Defense
  → Results/Reproducibility_Freeze
```

Historical Sprint 13 `reward_v2` artifacts remain auditable but are excluded from final `reward_v4` claims.

## Final entry points

- Dashboard: `Dashboard/app.py`
- Confirmatory builder: `scripts/run_confirmatory_benchmark.py`
- Dashboard evidence builder: `scripts/build_dashboard_release.py`
- Dissertation builder: `scripts/build_dissertation_release.py`
- Defense builder: `scripts/build_defense_package.py`
- Reproducibility freeze builder: `scripts/build_reproducibility_release.py`
- Final package: `SACSI_POMDP_Reproducibility_v1.0/`
