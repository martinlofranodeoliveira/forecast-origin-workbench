"""Public and report data transfer objects, enumerations, and domain constants."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "forecast-origin-workbench/1"
PREFIX_MAGIC = b"FOWprefix1\x00"
MODEL_ORDER: tuple[str, str, str] = ("hgb", "persistence", "seasonal")

MIN_ROWS = 128
MAX_ROWS_CAP = 4096
MAX_HGB_FITS_CAP = 64
MAX_DESIGN_CELLS_CAP = 8_388_608
MAX_JSON_BYTES_CAP = 1_048_576
MAX_CSV_BYTES_CAP = 1_048_576

MIN_LAG_COUNT = 1
MAX_LAG_COUNT = 32
MIN_HORIZON_COUNT = 1
MAX_HORIZON_COUNT = 8
MIN_SEASONAL_PERIOD = 2
MAX_SEASONAL_PERIOD = 32
MIN_ORIGINS_COUNT = 1
MAX_ORIGINS_COUNT = 8
MIN_HGB_TRAINING_ROWS = 64
MIN_BAND_HISTORY = 3
MAX_BAND_HISTORY = 7

MAX_ABS_VALUE = 1_000_000.0
MAX_ABS_TICK = 9_007_199_254_740_991

CSV_HEADER: tuple[str, ...] = (
    "model",
    "origin",
    "horizon",
    "target",
    "prediction",
    "training_status",
    "anchor_start",
    "anchor_end",
    "training_rows",
    "latest_training_target",
    "observed_support",
    "prefix_sha256",
    "band_status",
    "band_count",
    "band_rank",
    "band_radius",
    "band_lower",
    "band_upper",
    "eligible_residual_origins",
    "actual",
    "error",
)


class Status(StrEnum):
    """Outcome status codes for BacktestResult and EncodedReport."""

    COMPLETE = "COMPLETE"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_PLAN = "INVALID_PLAN"
    FIT_FAILED = "FIT_FAILED"
    NUMERICALLY_INDETERMINATE = "NUMERICALLY_INDETERMINATE"
    UNKNOWN_BUDGET = "UNKNOWN_BUDGET"
    INCOMPLETE = "INCOMPLETE"
    NOT_COMPLETE = "NOT_COMPLETE"


BACKTEST_RESULT_STATUSES: frozenset[Status] = frozenset(
    {
        Status.COMPLETE,
        Status.INVALID_INPUT,
        Status.INVALID_PLAN,
        Status.FIT_FAILED,
        Status.NUMERICALLY_INDETERMINATE,
        Status.UNKNOWN_BUDGET,
        Status.INCOMPLETE,
    }
)

ENCODED_REPORT_STATUSES: frozenset[Status] = frozenset(
    {
        Status.COMPLETE,
        Status.INVALID_INPUT,
        Status.NOT_COMPLETE,
        Status.UNKNOWN_BUDGET,
        Status.INCOMPLETE,
    }
)


class BandStatus(StrEnum):
    """Empirical residual band availability status."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE_INSUFFICIENT_HISTORY = "UNAVAILABLE_INSUFFICIENT_HISTORY"


class TrainingStatus(StrEnum):
    """Lineage training applicability status."""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Reason(StrEnum):
    """Machine-readable failure reason codes."""

    INPUT_SCHEMA = "INPUT_SCHEMA"
    PLAN_HISTORY = "PLAN_HISTORY"
    ADMISSION_BUDGET = "ADMISSION_BUDGET"
    JSON_BUDGET = "JSON_BUDGET"
    CSV_BUDGET = "CSV_BUDGET"
    SOURCE_NOT_COMPLETE = "SOURCE_NOT_COMPLETE"
    FIT_VALUE_ERROR = "FIT_VALUE_ERROR"
    NUMERIC_RESULT = "NUMERIC_RESULT"
    MEMORY_FAILURE = "MEMORY_FAILURE"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    IO_FAILURE = "IO_FAILURE"
    UNEXPECTED_OPERATIONAL = "UNEXPECTED_OPERATIONAL"


class _ReadOnlyMappingMixin(Mapping[str, Any]):
    """Provides read-only mapping access alongside frozen dataclass attributes."""

    def __getitem__(self, key: str) -> Any:
        for f in fields(self):  # type: ignore[arg-type]
            if f.name == key:
                return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for f in fields(self):  # type: ignore[arg-type]
            yield f.name

    def __len__(self) -> int:
        return len(fields(self))  # type: ignore[arg-type]


@dataclass(frozen=True)
class BacktestPlan(_ReadOnlyMappingMixin):
    """Passive immutable backtest schedule metadata."""

    lag_count: int
    horizon_count: int
    origins: tuple[int, ...] | Any
    seasonal_period: int

    def __post_init__(self) -> None:
        if type(self.origins) is list:
            object.__setattr__(self, "origins", tuple(self.origins))


@dataclass(frozen=True)
class Budget(_ReadOnlyMappingMixin):
    """Passive immutable admission and output budget caps."""

    max_rows: int = MAX_ROWS_CAP
    max_hgb_fits: int = MAX_HGB_FITS_CAP
    max_design_cells: int = MAX_DESIGN_CELLS_CAP
    max_json_bytes: int = MAX_JSON_BYTES_CAP


@dataclass(frozen=True)
class InputSummary(_ReadOnlyMappingMixin):
    """Summary metadata and whole-input digest for the admitted series."""

    count: int
    start_tick: int
    step: int
    sha256: str


@dataclass(frozen=True)
class LineageRecord(_ReadOnlyMappingMixin):
    """Forecast lineage metadata and prefix receipt."""

    training_status: str
    anchor_start: int | None
    anchor_end: int | None
    training_rows: int | None
    latest_training_target: int | None
    observed_support: tuple[int, ...]
    prefix_sha256: str


@dataclass(frozen=True)
class BandRecord(_ReadOnlyMappingMixin):
    """Historical empirical residual band record."""

    status: BandStatus
    count: int
    rank: int | None
    radius: float | None
    lower: float | None
    upper: float | None
    eligible_origins: tuple[int, ...]


@dataclass(frozen=True)
class ForecastRecord(_ReadOnlyMappingMixin):
    """Issued forecast cell record prior to target observation."""

    model: str
    origin: int
    horizon: int
    target: int
    prediction: float
    lineage: LineageRecord
    band: BandRecord


@dataclass(frozen=True)
class ScoreRecord(_ReadOnlyMappingMixin):
    """Post-emission realized actual and signed error record."""

    model: str
    origin: int
    horizon: int
    target: int
    actual: float
    error: float


@dataclass(frozen=True)
class MetricRecord(_ReadOnlyMappingMixin):
    """Model overall or per-horizon accuracy summary."""

    model: str
    horizon: int | None
    count: int
    mae: float
    rmse: float


@dataclass(frozen=True)
class WorkSummary(_ReadOnlyMappingMixin):
    """Admitted and executed work counters."""

    hgb_fits: int
    design_cells: int
    forecast_cells: int


@dataclass(frozen=True)
class BacktestReport(_ReadOnlyMappingMixin):
    """Complete immutable backtest report."""

    schema: str
    status: Status
    unit: str
    input: InputSummary
    plan: BacktestPlan
    forecasts: tuple[ForecastRecord, ...]
    scores: tuple[ScoreRecord, ...]
    metrics: tuple[MetricRecord, ...]
    work: WorkSummary


@dataclass(frozen=True)
class BacktestResult:
    """Typed outcome returned by run_backtest."""

    status: Status
    reason: str | None
    report: BacktestReport | None


@dataclass(frozen=True)
class EncodedReport:
    """Typed outcome returned by serialize_json and serialize_csv."""

    status: Status
    reason: str | None
    data: bytes | None
    required_bytes: int | None
