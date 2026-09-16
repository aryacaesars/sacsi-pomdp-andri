# Module 9A — Final Dashboard and Evidence Integration

## Release status

Module 9A is **READY**. The dashboard is a read-only presentation layer over the frozen Module 8A–8H evidence. It does not recompute scientific claims or substitute missing artifacts with demo data.

The final statistical source of truth is `Results/Confirmatory_10Seed`. Historical Sprint 13 results based on `reward_v2` remain available for audit but are excluded from final claims.

## Dashboard pages

1. Research Design
2. Reward Lab
3. Simple-Case & Raw-Data Validation
4. Fair DRL Benchmark
5. POMDP Contribution
6. 10-Seed Confirmatory Statistics
7. Robustness & Context Diagnostics
8. Reviewer Evidence Matrix
9. Reproducibility & Provenance

Every page is available in English and Bahasa Indonesia through the dashboard language selector.

## Evidence controls

- The result registry contains 24 explicitly whitelisted production artifacts.
- Every registered artifact records its readiness status, SHA-256 digest, byte size, and tabular row count where applicable.
- Missing evidence is shown as `NOT READY`; it is never silently replaced.
- Paths recognized as synthetic fixtures are rejected as production evidence.
- The reviewer evidence matrix maps all 12 reviewer items to a module, evidence file, dashboard page, bounded claim, and readiness state.
- Release metadata verifies the registry, reviewer matrix, and confirmatory manifest digests before the dashboard presents a `READY` release.

## Frozen headline results

- SACSI-POMDP mean Time in Target: 55.0183% across 10 matched seeds.
- SAC mean Time in Target: 54.1164% across 10 matched seeds.
- The locked SACSI training pipeline exceeded SAC by 0.9018 percentage points; the exact sign-flip contrast passed Holm correction at `p=0.046875`.
- The memory pipeline effect was supported; forecast and forecast-memory interaction effects were not supported.
- The released claim applies to the locked warm-start training pipelines. An unqualified equal-total-budget architecture-superiority claim is not released because context variants include SAC-anchor training plus adaptation interactions.

## Acceptance record

- All nine pages opened through Streamlit AppTest with zero application exceptions.
- Dashboard reconciliation tests passed for registry completeness, reviewer mapping, source-file equality, missing-file handling, and synthetic-fixture rejection.
- Reviewer evidence coverage is 12/12 (100%).
- Synthetic production evidence count is zero.

## Manual test

Run from PowerShell:

```powershell
cd D:\ARYA\SACSI_Dissertation
python scripts\build_dashboard_release.py
python -m pytest -p no:cacheprovider tests\test_dashboard.py -q
python -m streamlit run Dashboard\app.py
```

Expected results:

1. The release builder prints `status: READY`, 24 registry rows, 12 reviewer items, and zero synthetic production artifacts.
2. The dashboard test reports five passing tests.
3. Open `http://localhost:8501`, switch between English and Bahasa Indonesia, and visit all nine pages.
4. Confirm the top-level release status stays `READY` and that the Reproducibility page reports 24 ready artifacts, 100% reviewer mapping, and zero synthetic evidence.
5. Stop Streamlit with `Ctrl+C` after testing.

If a registered evidence file is changed, removed, or replaced, rebuild the release before deployment. Until the release hashes match, the dashboard must show `NOT READY`.
