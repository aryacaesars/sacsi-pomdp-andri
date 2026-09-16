"""Generate all Module 8C simple-case and raw-episode evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers import NoIrrigationController, RuleBasedForecastController, ThresholdController
from evaluation import compute_metrics, run_controller
from evaluation.simple_case_validation import RAW_EPISODES, run_simple_cases, select_raw_episodes


def main() -> None:
    data_dir = ROOT / "00_Dataset" / "Processed"
    output_dir = ROOT / "Results" / "Simple_Case_Validation"
    figure_dir = ROOT / "Figures" / "Simple_Case_Validation" / "simple_case_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    simple_results, simple_logs = run_simple_cases()
    if not simple_results["passed"].all():
        raise RuntimeError("one or more simple cases failed")
    simple_results.to_csv(output_dir / "simple_case_results.csv", index=False)
    simple_timeseries = pd.concat(simple_logs.values(), ignore_index=True)
    theta_figure = px.line(
        simple_timeseries, x="timestamp", y="theta", color="case_id",
        title="Module 8C — Simple-Case Soil-Moisture Responses",
    )
    theta_figure.add_hrect(y0=0.22, y1=0.32, fillcolor="green", opacity=0.10, line_width=0)
    theta_figure.write_html(figure_dir / "simple_case_theta.html", include_plotlyjs="directory")
    water = simple_timeseries.melt(
        id_vars=["case_id", "timestamp"],
        value_vars=["precipitation_mm", "irrigation_mm"],
        var_name="water_source", value_name="water_mm",
    )
    px.line(
        water, x="timestamp", y="water_mm", color="water_source", facet_row="case_id",
        title="Module 8C — Simple-Case Rain and Irrigation",
    ).write_html(figure_dir / "simple_case_water.html", include_plotlyjs="directory")

    validation = pd.read_csv(data_dir / "validation_2024.csv", parse_dates=["timestamp"])
    forecast = pd.read_csv(data_dir / "synthetic_forecast_sf20.csv", parse_dates=["timestamp"])
    forecast = forecast.rename(columns={
        "forecast_precipitation_mm": "forecast_precipitation_h1_mm",
    })[["timestamp", "forecast_precipitation_h1_mm"]]
    episodes = select_raw_episodes(validation)
    controllers = (NoIrrigationController, ThresholdController, RuleBasedForecastController)
    summary_rows = []
    for episode_name, weather in episodes.items():
        episode_forecast = forecast.loc[forecast["timestamp"].isin(weather["timestamp"])]
        controller_logs = []
        for controller_type in controllers:
            controller = controller_type()
            log = run_controller(weather, controller, episode_forecast)
            log.insert(0, "episode", episode_name.upper())
            log["weather_data_class"] = "real_raw_meteorological_forcing"
            log["soil_state_class"] = "virtual_garden_simulated"
            log["forecast_data_class"] = "controlled_synthetic_proxy_sf20"
            log["controller_tuning"] = "fixed_default_no_episode_retuning"
            if not log["irrigation_mm"].between(0.0, 5.0).all():
                raise RuntimeError(f"action bound failed for {episode_name}/{controller.name}")
            if not np.isfinite(log.select_dtypes(include="number").to_numpy()).all():
                raise RuntimeError(f"non-finite output for {episode_name}/{controller.name}")
            metrics = compute_metrics(log)
            if metrics["max_abs_mass_balance_error_mm"] > 1e-8:
                raise RuntimeError(f"mass balance failed for {episode_name}/{controller.name}")
            summary_rows.append({
                "episode": episode_name.upper(),
                "controller": controller.name,
                "start": weather.timestamp.min(),
                "end": weather.timestamp.max(),
                "hours": len(weather),
                "episode_rain_mm": weather.precipitation_mm.sum(),
                "action_bounded": True,
                "all_finite": True,
                "same_config": "VirtualGardenConfig_default_field_capacity_0.35",
                "episode_specific_retuning": False,
                **metrics,
            })
            controller_logs.append(log)
        pd.concat(controller_logs, ignore_index=True).to_csv(
            output_dir / f"raw_episode_{episode_name}.csv", index=False
        )

    raw_summary = pd.DataFrame(summary_rows)
    raw_summary.to_csv(output_dir / "raw_episode_summary.csv", index=False)
    print(simple_results[["case_id", "expected_response", "passed"]].to_string(index=False))
    print(raw_summary[[
        "episode", "controller", "episode_rain_mm", "total_irrigation_mm",
        "time_in_target_pct", "max_abs_mass_balance_error_mm",
    ]].to_string(index=False))
    print(f"outputs={output_dir}")


if __name__ == "__main__":
    main()
