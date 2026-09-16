import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_garden import VirtualGardenConfig, VirtualGardenCore


def assert_balanced(result) -> None:
    assert abs(result.mass_balance_error_mm) <= 1e-8


def test_defaults_match_locked_methodology() -> None:
    cfg = VirtualGardenConfig()
    assert cfg.initial_theta == 0.27
    assert (cfg.target_min, cfg.target_max) == (0.22, 0.32)
    assert cfg.field_capacity == 0.35
    assert cfg.max_irrigation_mm_h == 5.0


def test_no_rain_no_irrigation_dries_soil() -> None:
    garden = VirtualGardenCore()
    result = garden.step(precipitation_mm=0, et0_mm=0.4, irrigation_mm=0)
    assert result.theta < garden.config.initial_theta
    assert result.runoff_mm == 0
    assert_balanced(result)


def test_heavy_rain_produces_runoff_and_overflow() -> None:
    garden = VirtualGardenCore()
    result = garden.step(precipitation_mm=100, et0_mm=0, irrigation_mm=0)
    assert result.runoff_mm > 0
    assert result.theta <= garden.config.saturation
    assert_balanced(result)


def test_irrigation_is_clipped_to_action_bound() -> None:
    garden = VirtualGardenCore()
    result = garden.step(precipitation_mm=0, et0_mm=0, irrigation_mm=99)
    assert result.irrigation_mm == garden.config.max_irrigation_mm_h
    assert_balanced(result)


def test_near_saturation_drains_without_exceeding_saturation() -> None:
    garden = VirtualGardenCore()
    garden.reset(theta=0.449)
    result = garden.step(precipitation_mm=0, et0_mm=0, irrigation_mm=5)
    assert result.overflow_mm > 0
    assert result.drainage_mm > 0
    assert result.theta < garden.config.saturation
    assert_balanced(result)


def test_long_dry_period_stops_at_wilting_point() -> None:
    garden = VirtualGardenCore()
    results = [garden.step(0, 1.0, 0) for _ in range(2_000)]
    assert garden.theta == pytest.approx(garden.config.wilting_point, abs=1e-12)
    assert max(abs(result.mass_balance_error_mm) for result in results) <= 1e-8


def test_invalid_weather_is_rejected() -> None:
    garden = VirtualGardenCore()
    with pytest.raises(ValueError):
        garden.step(-1, 0, 0)
    with pytest.raises(ValueError):
        garden.step(0, float("nan"), 0)
