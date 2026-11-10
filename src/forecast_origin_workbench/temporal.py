"""Temporal indexing, prefix hashing, lineage construction, and empirical bands."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import Any

from .types import (
     MAX_HORIZON_COUNT,
    MAX_LAG_COUNT,
    MAX_ORIGINS_COUNT,
    MAX_SEASONAL_PERIOD,
    MIN_BAND_HISTORY,
    MIN_HGB_TRAINING_ROWS,
    MIN_HORIZON_COUNT,
    MIN_LAG_COUNT,
    MIN_ORIGINS_COUNT,
    MIN_SEASONAL_PERIOD,
    PREFIX_MAGIC,
    BacktestPlan,
    BandRecord,
    BandStatus,
    LineageRecord,
    TrainingStatus,
    WorkSummary,
)


def is_strict_int(value: Any) -> bool:
    """Return True iff value is an exact Python int and not a bool."""
    return type(value) is int


def canonicalize_zero(value: float) -> float:
    """Canonicalize -0.0 to +0.0 while preserving all other float64 values."""
    return 0.0 if value == 0.0 else value


def compute_fow_sha256(
    values: Sequence[float],
    start_tick: int,
    step: int,
    count: int,
) -> str:
    """Compute the FOWprefix1 SHA256 digest over the first `count` observations."""
    hasher = hashlib.sha256()
    hasher.update(PREFIX_MAGIC)
    hasher.update(struct.pack("<qQI", start_tick, step, count))
    canonical = [canonicalize_zero(values[i]) for i in range(count)]
    hasher.update(struct.pack(f"<{count}d", *canonical))
    return hasher.hexdigest()


def compute_seasonal_support_index(
    origin: int,
    horizon: int,
    seasonal_period: int,
) -> int:
    """Compute s = o + h - ceil(h / P) * P using exact integer arithmetic."""
    cycles = (horizon + seasonal_period - 1) // seasonal_period
    return origin + horizon - cycles * seasonal_period


def validate_plan_and_compute_work(
    plan: Any,
    n_rows: int,
) -> WorkSummary | None:
    """Validate BacktestPlan fields and history rules against n_rows.

    Returns WorkSummary if valid, or None if any plan or history rule is violated.
    """
    if type(plan) is not BacktestPlan:
        return None

    lag_count = plan.lag_count
    horizon_count = plan.horizon_count
    seasonal_period = plan.seasonal_period
    origins = plan.origins

    if not is_strict_int(lag_count) or not (MIN_LAG_COUNT <= lag_count <= MAX_LAG_COUNT):
        return None
    if not is_strict_int(horizon_count) or not (
        MIN_HORIZON_COUNT <= horizon_count <= MAX_HORIZON_COUNT
    ):
        return None
    if not is_strict_int(seasonal_period) or not (
        MIN_SEASONAL_PERIOD <= seasonal_period <= MAX_SEASONAL_PERIOD
    ):
        return None
    if type(origins) is not tuple:
        return None
    if not (MIN_ORIGINS_COUNT <= len(origins) <= MAX_ORIGINS_COUNT):
        return None

    min_required_origin = max(lag_count - 1, seasonal_period - 1)
    prev_origin: int | None = None
    design_cells = 0

    for origin in origins:
        if not is_strict_int(origin):
            return None
        if prev_origin is not None and origin <= prev_origin:
            return None
        prev_origin = origin

        if origin < min_required_origin:
            return None
        if origin + horizon_count >= n_rows:
            return None

        for horizon in range(1, horizon_count + 1):
            n_train = origin - horizon - lag_count + 2
            if n_train < MIN_HGB_TRAINING_ROWS:
                return None
            s_idx = compute_seasonal_support_index(origin, horizon, seasonal_period)
            if s_idx < 0 or s_idx > origin:
                return None
            design_cells += lag_count * n_train + lag_count

    hgb_fits = len(origins) * horizon_count
    forecast_cells = hgb_fits * 3
    return WorkSummary(
        hgb_fits=hgb_fits,
        design_cells=design_cells,
        forecast_cells=forecast_cells,
    )


def build_hgb_lineage(
    origin: int,
    horizon: int,
    lag_count: int,
    prefix_sha256: str,
) -> LineageRecord:
    """Construct the APPLICABLE lineage record for a direct (origin, horizon) HGB fit."""
    anchor_start = lag_count - 1
    anchor_end = origin - horizon
    training_rows = origin - horizon - lag_count + 2
    observed_support = tuple(range(origin, origin - lag_count, -1))
    return LineageRecord(
        training_status=TrainingStatus.APPLICABLE,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        training_rows=training_rows,
        latest_training_target=origin,
        observed_support=observed_support,
        prefix_sha256=prefix_sha256,
    )


def build_baseline_lineage(
    support_index: int,
    prefix_sha256: str,
) -> LineageRecord:
    """Construct the NOT_APPLICABLE lineage record for persistence or seasonal baseline."""
    return LineageRecord(
        training_status=TrainingStatus.NOT_APPLICABLE,
        anchor_start=None,
        anchor_end=None,
        training_rows=None,
        latest_training_target=None,
        observed_support=(support_index,),
        prefix_sha256=prefix_sha256,
    )


def build_hgb_design_arrays(
    np_module: Any,
    observed_prefix: Sequence[float],
    origin: int,
    horizon: int,
    lag_count: int,
) -> tuple[Any, Any, Any]:
    """Construct owned X_train, y_train, and X_query arrays using only y[0..origin]."""
    if len(observed_prefix) != origin + 1:
        raise RuntimeError("Prefix length mismatch with origin")

    anchor_start = lag_count - 1
    anchor_end = origin - horizon
    n_train = anchor_end - anchor_start + 1

    x_rows: list[list[float]] = []
    y_targets: list[float] = []
    for anchor in range(anchor_start, anchor_end + 1):
        row = [observed_prefix[anchor - lag] for lag in range(lag_count)]
        x_rows.append(row)
        y_targets.append(observed_prefix[anchor + horizon])

    query_row = [[observed_prefix[origin - lag] for lag in range(lag_count)]]

    x_train = np_module.array(x_rows, dtype=np_module.float64, copy=True)
    y_train = np_module.array(y_targets, dtype=np_module.float64, copy=True)
    x_query = np_module.array(query_row, dtype=np_module.float64, copy=True)

    if x_train.shape != (n_train, lag_count) or y_train.shape != (n_train,) or x_query.shape != (1, lag_count):
        raise RuntimeError("Constructed design matrix shape mismatch")

    return x_train, y_train, x_query


def compute_empirical_band(
    prediction: float,
    origin: int,
    horizon: int,
    prior_emissions: Sequence[tuple[int, float]],
    observed_prefix: Sequence[float],
) -> BandRecord | None:
    """Compute historical empirical residual band from mature prior emissions.

    `prior_emissions` is ordered by prior origin `r < origin` for the same `(model, horizon)`.
    Only emissions with `r + horizon <= origin` are eligible.
    Returns None if any computed band bound is non-finite.
    """
    eligible_origins: list[int] = []
    abs_errors: list[float] = []

    for prior_origin, prior_pred in prior_emissions:
        if prior_origin < origin and prior_origin + horizon <= origin:
            target_idx = prior_origin + horizon
            realized = observed_prefix[target_idx]
            err = abs(realized - prior_pred)
            if not math.isfinite(err):
                return None
            eligible_origins.append(prior_origin)
            abs_errors.append(canonicalize_zero(err))

    m = len(eligible_origins)
    if m < MIN_BAND_HISTORY:
        return BandRecord(
            status=BandStatus.UNAVAILABLE_INSUFFICIENT_HISTORY,
            count=m,
            rank=None,
            radius=None,
            lower=None,
            upper=None,
            eligible_origins=tuple(eligible_origins),
        )

    rank = (4 * m + 4) // 5
    sorted_errors = sorted(abs_errors)
    radius = canonicalize_zero(float(sorted_errors[rank - 1]))
    lower = canonicalize_zero(float(prediction - radius))
    upper = canonicalize_zero(float(prediction + radius))

    if not (math.isfinite(radius) and math.isfinite(lower) and math.isfinite(upper)):
        return None

    return BandRecord(
        status=BandStatus.AVAILABLE,
        count=m,
        rank=rank,
        radius=radius,
        lower=lower,
        upper=upper,
        eligible_origins=tuple(eligible_origins),
    )
