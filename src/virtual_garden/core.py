"""Controller-independent hourly root-zone water-balance simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class VirtualGardenConfig:
    initial_theta: float = 0.27
    wilting_point: float = 0.15
    target_min: float = 0.22
    target_max: float = 0.32
    field_capacity: float = 0.35
    saturation: float = 0.45
    root_depth_mm: float = 300.0
    crop_coefficient: float = 0.85
    max_irrigation_mm_h: float = 5.0
    infiltration_capacity_mm_h: float = 20.0
    drainage_fraction_h: float = 0.10

    def __post_init__(self) -> None:
        if not (
            0 <= self.wilting_point < self.target_min
            < self.target_max <= self.field_capacity < self.saturation <= 1
        ):
            raise ValueError("Invalid soil-moisture thresholds")
        if not (self.wilting_point <= self.initial_theta <= self.saturation):
            raise ValueError("initial_theta must be between wilting point and saturation")
        if self.root_depth_mm <= 0 or self.crop_coefficient < 0:
            raise ValueError("root depth must be positive and crop coefficient non-negative")
        if self.max_irrigation_mm_h <= 0 or self.infiltration_capacity_mm_h <= 0:
            raise ValueError("water-input limits must be positive")
        if not 0 <= self.drainage_fraction_h <= 1:
            raise ValueError("drainage_fraction_h must be in [0, 1]")


@dataclass(frozen=True)
class StepResult:
    theta: float
    storage_mm: float
    precipitation_mm: float
    irrigation_mm: float
    infiltration_mm: float
    evapotranspiration_mm: float
    drainage_mm: float
    runoff_mm: float
    overflow_mm: float
    mass_balance_error_mm: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class VirtualGardenCore:
    """Hourly water balance; receives weather/action and owns no controller logic."""

    def __init__(self, config: VirtualGardenConfig | None = None) -> None:
        self.config = config or VirtualGardenConfig()
        self.reset()

    def reset(self, theta: float | None = None) -> float:
        theta = self.config.initial_theta if theta is None else float(theta)
        if not self.config.wilting_point <= theta <= self.config.saturation:
            raise ValueError("reset theta must be between wilting point and saturation")
        self.storage_mm = theta * self.config.root_depth_mm
        self.max_abs_mass_balance_error_mm = 0.0
        return theta

    @property
    def theta(self) -> float:
        return self.storage_mm / self.config.root_depth_mm

    def step(self, precipitation_mm: float, et0_mm: float, irrigation_mm: float) -> StepResult:
        values = (precipitation_mm, et0_mm, irrigation_mm)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("step inputs must be finite")
        if precipitation_mm < 0 or et0_mm < 0:
            raise ValueError("precipitation and ET0 must be non-negative")

        cfg = self.config
        old_storage = self.storage_mm
        irrigation = min(max(float(irrigation_mm), 0.0), cfg.max_irrigation_mm_h)
        total_input = float(precipitation_mm) + irrigation

        potential_infiltration = min(total_input, cfg.infiltration_capacity_mm_h)
        surface_runoff = total_input - potential_infiltration
        provisional_storage = old_storage + potential_infiltration
        saturation_storage = cfg.saturation * cfg.root_depth_mm
        overflow = max(provisional_storage - saturation_storage, 0.0)
        storage = provisional_storage - overflow

        wilting_storage = cfg.wilting_point * cfg.root_depth_mm
        stress_denominator = (cfg.target_min - cfg.wilting_point) * cfg.root_depth_mm
        water_stress = min(max((storage - wilting_storage) / stress_denominator, 0.0), 1.0)
        potential_et = float(et0_mm) * cfg.crop_coefficient * water_stress
        actual_et = min(potential_et, max(storage - wilting_storage, 0.0))
        storage -= actual_et

        field_capacity_storage = cfg.field_capacity * cfg.root_depth_mm
        drainage = max(storage - field_capacity_storage, 0.0) * cfg.drainage_fraction_h
        storage -= drainage

        runoff = surface_runoff + overflow
        residual = old_storage + total_input - (storage + actual_et + drainage + runoff)
        self.storage_mm = storage
        self.max_abs_mass_balance_error_mm = max(
            self.max_abs_mass_balance_error_mm, abs(residual)
        )

        return StepResult(
            theta=self.theta,
            storage_mm=storage,
            precipitation_mm=float(precipitation_mm),
            irrigation_mm=irrigation,
            infiltration_mm=potential_infiltration,
            evapotranspiration_mm=actual_et,
            drainage_mm=drainage,
            runoff_mm=runoff,
            overflow_mm=overflow,
            mass_balance_error_mm=residual,
        )
