import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.simple_case_validation import RAW_EPISODES, run_simple_cases, select_raw_episodes


def test_six_simple_cases_pass_physics_and_control_guards():
    summary, logs = run_simple_cases()
    assert summary["case_id"].tolist() == ["C1", "C2", "C3", "C4", "C5", "C6"]
    assert summary["passed"].all()
    assert set(logs) == set(summary["case_id"])
    assert summary["max_abs_mass_balance_error_mm"].max() <= 1e-8
    assert summary["action_bounded"].all() and summary["all_finite"].all()


def test_raw_episode_dates_hours_and_rain_are_locked():
    validation = pd.read_csv(
        ROOT / "00_Dataset" / "Processed" / "validation_2024.csv",
        parse_dates=["timestamp"],
    )
    episodes = select_raw_episodes(validation)
    assert set(episodes) == set(RAW_EPISODES)
    for name, (_, _, expected_rain) in RAW_EPISODES.items():
        assert len(episodes[name]) == 336
        assert np.isclose(episodes[name]["precipitation_mm"].sum(), expected_rain)


def test_module_8c_production_artifacts_are_complete():
    output = ROOT / "Results" / "Simple_Case_Validation"
    simple = pd.read_csv(output / "simple_case_results.csv")
    raw_summary = pd.read_csv(output / "raw_episode_summary.csv")
    assert len(simple) == 6 and simple["passed"].all()
    assert len(raw_summary) == 3 * 3
    assert set(raw_summary["episode"]) == {"DRY", "WET", "MIXED"}
    assert raw_summary.groupby("episode")["controller"].nunique().eq(3).all()
    assert raw_summary["action_bounded"].all() and raw_summary["all_finite"].all()
    assert raw_summary["max_abs_mass_balance_error_mm"].max() <= 1e-8
    for episode in ("dry", "wet", "mixed"):
        log = pd.read_csv(output / f"raw_episode_{episode}.csv")
        assert len(log) == 336 * 3
        assert set(log["weather_data_class"]) == {"real_raw_meteorological_forcing"}
        assert set(log["soil_state_class"]) == {"virtual_garden_simulated"}
        assert set(log["forecast_data_class"]) == {"controlled_synthetic_proxy_sf20"}
        assert set(log["controller_tuning"]) == {"fixed_default_no_episode_retuning"}
    figures = ROOT / "Figures" / "Simple_Case_Validation" / "simple_case_figures"
    assert (figures / "simple_case_theta.html").exists()
    assert (figures / "simple_case_water.html").exists()
