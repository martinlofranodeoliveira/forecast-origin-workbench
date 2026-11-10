"""Core backtest execution, validation, and canonical JSON/CSV serialization."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .temporal import (
    build_baseline_lineage,
    build_hgb_design_arrays,
    build_hgb_lineage,
    canonicalize_zero,
    compute_empirical_band,
    compute_fow_sha256,
    compute_seasonal_support_index,
    is_strict_int,
    validate_plan_and_compute_work,
)
from .types import (
    BACKTEST_RESULT_STATUSES,
    CSV_HEADER,
    MAX_ABS_TICK,
    MAX_ABS_VALUE,
    MAX_BAND_HISTORY,
    MAX_CSV_BYTES_CAP,
    MAX_DESIGN_CELLS_CAP,
    MAX_HGB_FITS_CAP,
    MAX_JSON_BYTES_CAP,
    MAX_ROWS_CAP,
    MIN_BAND_HISTORY,
    MIN_ROWS,
    MODEL_ORDER,
    SCHEMA_VERSION,
    BacktestPlan,
    BacktestReport,
    BacktestResult,
    BandRecord,
    BandStatus,
    Budget,
    EncodedReport,
    ForecastRecord,
    InputSummary,
    LineageRecord,
    MetricRecord,
    Reason,
    ScoreRecord,
    Status,
    TrainingStatus,
    WorkSummary,
)

_UNIT_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_HGB_ITERATIONS = 64



def _metrics_for_scores(scores: Sequence[ScoreRecord]) -> tuple[float, float] | None:
    """Compute owned metric arithmetic; overflow/nonfinite means unresolved numeric."""
    count = len(scores)
    try:
        mae = canonicalize_zero(math.fsum(abs(score.error) for score in scores) / count)
        rmse = canonicalize_zero(math.sqrt(math.fsum(score.error * score.error for score in scores) / count))
    except OverflowError:
        return None
    if not (math.isfinite(mae) and math.isfinite(rmse)):
        return None
    return mae, rmse


def _is_valid_unit(unit: Any) -> bool:
    return type(unit) is str and unit.isascii() and bool(_UNIT_PATTERN.fullmatch(unit))


def _is_valid_sha256(digest: Any) -> bool:
    return type(digest) is str and bool(_SHA256_PATTERN.fullmatch(digest))


def _is_valid_budget(budget: Any) -> bool:
    if type(budget) is not Budget:
        return False
    return (
        is_strict_int(budget.max_rows)
        and 0 <= budget.max_rows <= MAX_ROWS_CAP
        and is_strict_int(budget.max_hgb_fits)
        and 0 <= budget.max_hgb_fits <= MAX_HGB_FITS_CAP
        and is_strict_int(budget.max_design_cells)
        and 0 <= budget.max_design_cells <= MAX_DESIGN_CELLS_CAP
        and is_strict_int(budget.max_json_bytes)
        and 0 <= budget.max_json_bytes <= MAX_JSON_BYTES_CAP
    )


def _validate_container_and_ticks(
    values: Any,
    start_tick: Any,
    step: Any,
    unit: Any,
) -> int | None:
    """Validate metadata and cheap container length without accessing elements."""
    if not _is_valid_unit(unit):
        return None
    if not is_strict_int(start_tick) or not (-MAX_ABS_TICK <= start_tick <= MAX_ABS_TICK):
        return None
    if not is_strict_int(step) or step <= 0:
        return None

    if isinstance(values, (str, bytes, bytearray, memoryview, Mapping, set, frozenset)):
        return None
    if hasattr(values, "ndim"):
        try:
            if values.ndim != 1:
                return None
        except (TypeError, ValueError, OverflowError, AttributeError):
            return None
    if not hasattr(values, "__len__") or not hasattr(values, "__getitem__"):
        return None

    try:
        n_rows = len(values)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None

    if not is_strict_int(n_rows) or not (MIN_ROWS <= n_rows <= MAX_ROWS_CAP):
        return None

    last_tick = start_tick + (n_rows - 1) * step
    if not (-MAX_ABS_TICK <= last_tick <= MAX_ABS_TICK):
        return None

    return n_rows


def _extract_owned_float64_values(
    values: Any,
    n_rows: int,
    np_module: Any,
) -> tuple[float, ...] | None:
    """Validate every scalar element and return an owned tuple of canonical floats."""
    if hasattr(values, "dtype"):
        kind = getattr(values.dtype, "kind", None)
        if kind not in ("f", "i", "u"):
            return None

    allowed_types = (int, float, np_module.integer, np_module.floating)
    excluded_types = (bool, np_module.bool_)

    owned: list[float] = []
    for idx in range(n_rows):
        item = values[idx]
        if isinstance(item, excluded_types) or not isinstance(item, allowed_types):
            return None
        try:
            fval = float(item)
        except (OverflowError, TypeError, ValueError):
            return None
        if not math.isfinite(fval) or fval < -MAX_ABS_VALUE or fval > MAX_ABS_VALUE:
            return None
        owned.append(canonicalize_zero(fval))

    if len(owned) != n_rows:
        return None
    return tuple(owned)


def _report_to_wire(report: BacktestReport) -> dict[str, Any]:
    """Convert an immutable BacktestReport into a JSON-serializable wire dictionary."""
    return {
        "schema": report.schema,
        "status": str(report.status),
        "unit": report.unit,
        "input": {
            "count": report.input.count,
            "start_tick": report.input.start_tick,
            "step": report.input.step,
            "sha256": report.input.sha256,
        },
        "plan": {
            "lag_count": report.plan.lag_count,
            "horizon_count": report.plan.horizon_count,
            "origins": list(report.plan.origins),
            "seasonal_period": report.plan.seasonal_period,
        },
        "forecasts": [
            {
                "model": f.model,
                "origin": f.origin,
                "horizon": f.horizon,
                "target": f.target,
                "prediction": f.prediction,
                "lineage": {
                    "training_status": str(f.lineage.training_status),
                    "anchor_start": f.lineage.anchor_start,
                    "anchor_end": f.lineage.anchor_end,
                    "training_rows": f.lineage.training_rows,
                    "latest_training_target": f.lineage.latest_training_target,
                    "observed_support": list(f.lineage.observed_support),
                    "prefix_sha256": f.lineage.prefix_sha256,
                },
                "band": {
                    "status": str(f.band.status),
                    "count": f.band.count,
                    "rank": f.band.rank,
                    "radius": f.band.radius,
                    "lower": f.band.lower,
                    "upper": f.band.upper,
                    "eligible_origins": list(f.band.eligible_origins),
                },
            }
            for f in report.forecasts
        ],
        "scores": [
            {
                "model": s.model,
                "origin": s.origin,
                "horizon": s.horizon,
                "target": s.target,
                "actual": s.actual,
                "error": s.error,
            }
            for s in report.scores
        ],
        "metrics": [
            {
                "model": m.model,
                "horizon": m.horizon,
                "count": m.count,
                "mae": m.mae,
                "rmse": m.rmse,
            }
            for m in report.metrics
        ],
        "work": {
            "hgb_fits": report.work.hgb_fits,
            "design_cells": report.work.design_cells,
            "forecast_cells": report.work.forecast_cells,
        },
    }


def _encode_canonical_json_bytes(report: BacktestReport) -> bytes:
    """Build canonical UTF-8 JSON bytes terminated with a single LF."""
    wire = _report_to_wire(report)
    serialized = json.dumps(
        wire,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return serialized.encode("utf-8") + b"\n"


def _format_csv_float(value: float) -> str:
    return json.dumps(canonicalize_zero(float(value)), allow_nan=False)


def _format_csv_int_list(values: Sequence[int]) -> str:
    return json.dumps(
        list(values),
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _encode_csv_bytes(report: BacktestReport) -> bytes:
    """Build canonical UTF-8 CSV bytes with LF line terminators."""
    buf = io.StringIO()
    writer = csv.writer(buf, dialect="excel", lineterminator="\n")
    writer.writerow(CSV_HEADER)

    for f, sc in zip(report.forecasts, report.scores, strict=True):
        lin = f.lineage
        bnd = f.band
        row = [
            f.model,
            str(f.origin),
            str(f.horizon),
            str(f.target),
            _format_csv_float(f.prediction),
            str(lin.training_status),
            "" if lin.anchor_start is None else str(lin.anchor_start),
            "" if lin.anchor_end is None else str(lin.anchor_end),
            "" if lin.training_rows is None else str(lin.training_rows),
            "" if lin.latest_training_target is None else str(lin.latest_training_target),
            _format_csv_int_list(lin.observed_support),
            lin.prefix_sha256,
            str(bnd.status),
            str(bnd.count),
            "" if bnd.rank is None else str(bnd.rank),
            "" if bnd.radius is None else _format_csv_float(bnd.radius),
            "" if bnd.lower is None else _format_csv_float(bnd.lower),
            "" if bnd.upper is None else _format_csv_float(bnd.upper),
            _format_csv_int_list(bnd.eligible_origins),
            _format_csv_float(sc.actual),
            _format_csv_float(sc.error),
        ]
        writer.writerow(row)

    return buf.getvalue().encode("utf-8")


def _is_finite_float(val: Any) -> bool:
    return type(val) is float and math.isfinite(val)


def _validate_complete_report(report: Any) -> bool:
    """Validate public DTO types and all lineage, band, score, and count relations."""
    if type(report) is not BacktestReport:
        return False
    if type(report.schema) is not str or report.schema != SCHEMA_VERSION or type(report.status) is not Status or report.status != Status.COMPLETE:
        return False
    if not _is_valid_unit(report.unit):
        return False

    inp = report.input
    if type(inp) is not InputSummary:
        return False
    if not is_strict_int(inp.count) or not (MIN_ROWS <= inp.count <= MAX_ROWS_CAP):
        return False
    if not is_strict_int(inp.start_tick) or not (-MAX_ABS_TICK <= inp.start_tick <= MAX_ABS_TICK):
        return False
    if not is_strict_int(inp.step) or inp.step <= 0:
        return False
    if not (-MAX_ABS_TICK <= inp.start_tick + (inp.count - 1) * inp.step <= MAX_ABS_TICK):
        return False
    if not _is_valid_sha256(inp.sha256):
        return False

    plan = report.plan
    expected_work = validate_plan_and_compute_work(plan, inp.count)
    if expected_work is None:
        return False

    work = report.work
    if (type(work) is not WorkSummary
            or not all(is_strict_int(getattr(work, name)) for name in ("hgb_fits", "design_cells", "forecast_cells"))
            or work != expected_work):
        return False
    if work.hgb_fits > MAX_HGB_FITS_CAP or work.design_cells > MAX_DESIGN_CELLS_CAP:
        return False

    forecasts = report.forecasts
    scores = report.scores
    metrics = report.metrics
    if (
        type(forecasts) is not tuple
        or type(scores) is not tuple
        or type(metrics) is not tuple
    ):
        return False
    if len(forecasts) != work.forecast_cells or len(scores) != work.forecast_cells:
        return False

    origins: tuple[int, ...] = plan.origins
    lag_count = plan.lag_count
    horizon_count = plan.horizon_count
    seasonal_period = plan.seasonal_period

    expected_cells = [
        (o, h, m)
        for o in origins
        for h in range(1, horizon_count + 1)
        for m in MODEL_ORDER
    ]

    origin_prefix_hashes: dict[int, str] = {}
    validated_cells: dict[tuple[str, int, int], tuple[ForecastRecord, ScoreRecord]] = {}
    actual_by_target: dict[int, float] = {}

    for idx, (exp_o, exp_h, exp_m) in enumerate(expected_cells):
        f = forecasts[idx]
        sc = scores[idx]
        if type(f) is not ForecastRecord or type(sc) is not ScoreRecord:
            return False
        if (not is_strict_int(f.origin) or not is_strict_int(f.horizon) or type(f.model) is not str
                or not is_strict_int(sc.origin) or not is_strict_int(sc.horizon) or type(sc.model) is not str):
            return False
        if (f.origin, f.horizon, f.model) != (exp_o, exp_h, exp_m):
            return False
        if (sc.origin, sc.horizon, sc.model) != (exp_o, exp_h, exp_m):
            return False
        exp_target = exp_o + exp_h
        if not is_strict_int(f.target) or f.target != exp_target:
            return False
        if not is_strict_int(sc.target) or sc.target != exp_target:
            return False
        if not _is_finite_float(f.prediction):
            return False
        if not _is_finite_float(sc.actual) or abs(sc.actual) > MAX_ABS_VALUE:
            return False
        if not _is_finite_float(sc.error):
            return False
        if sc.error != canonicalize_zero(sc.actual - f.prediction):
            return False
        if exp_target in actual_by_target and actual_by_target[exp_target] != sc.actual:
            return False
        actual_by_target[exp_target] = sc.actual

        lin = f.lineage
        if type(lin) is not LineageRecord:
            return False
        if not _is_valid_sha256(lin.prefix_sha256):
            return False
        if exp_o in origin_prefix_hashes:
            if lin.prefix_sha256 != origin_prefix_hashes[exp_o]:
                return False
        else:
            origin_prefix_hashes[exp_o] = lin.prefix_sha256

        expected_support_count = lag_count if exp_m == "hgb" else 1
        if (type(lin.training_status) not in (str, TrainingStatus)
                or type(lin.observed_support) is not tuple
                or len(lin.observed_support) != expected_support_count
                or not all(is_strict_int(idx_val) for idx_val in lin.observed_support)) :
            return False

        if exp_m == "hgb":
            exp_train_rows = exp_o - exp_h - lag_count + 2
            if (
                lin.training_status != TrainingStatus.APPLICABLE
                or not is_strict_int(lin.anchor_start)
                or lin.anchor_start != lag_count - 1
                or not is_strict_int(lin.anchor_end)
                or lin.anchor_end != exp_o - exp_h
                or not is_strict_int(lin.training_rows)
                or lin.training_rows != exp_train_rows
                or not is_strict_int(lin.latest_training_target)
                or lin.latest_training_target != exp_o
                or lin.observed_support != tuple(range(exp_o, exp_o - lag_count, -1))
            ):
                return False
        elif exp_m == "persistence":
            if (
                lin.training_status != TrainingStatus.NOT_APPLICABLE
                or lin.anchor_start is not None
                or lin.anchor_end is not None
                or lin.training_rows is not None
                or lin.latest_training_target is not None
                or lin.observed_support != (exp_o,)
            ):
                return False
        elif exp_m == "seasonal":
            exp_s = compute_seasonal_support_index(exp_o, exp_h, seasonal_period)
            if (
                lin.training_status != TrainingStatus.NOT_APPLICABLE
                or lin.anchor_start is not None
                or lin.anchor_end is not None
                or lin.training_rows is not None
                or lin.latest_training_target is not None
                or lin.observed_support != (exp_s,)
            ):
                return False

        bnd = f.band
        if type(bnd) is not BandRecord or type(bnd.status) is not BandStatus:
            return False
        exp_eligible = tuple(r for r in origins if r < exp_o and r + exp_h <= exp_o)
        if (type(bnd.eligible_origins) is not tuple
                or len(bnd.eligible_origins) != len(exp_eligible)
                or not all(is_strict_int(r) for r in bnd.eligible_origins)
                or bnd.eligible_origins != exp_eligible):
            return False
        exp_m_count = len(exp_eligible)
        if not is_strict_int(bnd.count) or bnd.count != exp_m_count:
            return False
        if not (0 <= bnd.count <= MAX_BAND_HISTORY):
            return False

        if exp_m_count < MIN_BAND_HISTORY:
            if (
                bnd.status != BandStatus.UNAVAILABLE_INSUFFICIENT_HISTORY
                or bnd.rank is not None
                or bnd.radius is not None
                or bnd.lower is not None
                or bnd.upper is not None
            ):
                return False
        else:
            exp_rank = (4 * exp_m_count + 4) // 5
            if (
                bnd.status != BandStatus.AVAILABLE
                or not is_strict_int(bnd.rank)
                or bnd.rank != exp_rank
                or not _is_finite_float(bnd.radius)
                or bnd.radius < 0.0
                or not _is_finite_float(bnd.lower)
                or not _is_finite_float(bnd.upper)
                or bnd.lower != canonicalize_zero(f.prediction - bnd.radius)
                or bnd.upper != canonicalize_zero(f.prediction + bnd.radius)
            ):
                return False
            mature_errors = sorted(
                abs(previous_score.actual - previous_forecast.prediction)
                for r in exp_eligible
                for previous_forecast, previous_score in [validated_cells[(exp_m, r, exp_h)]]
            )
            if bnd.radius != canonicalize_zero(mature_errors[exp_rank - 1]):
                return False
        validated_cells[(exp_m, exp_o, exp_h)] = (f, sc)

    expected_metric_keys: list[tuple[str, int | None]] = [
        (m, h)
        for m in MODEL_ORDER
        for h in (None, *range(1, horizon_count + 1))
    ]
    if len(metrics) != len(expected_metric_keys):
        return False

    for mr, (exp_m, exp_h) in zip(metrics, expected_metric_keys, strict=True):
        if type(mr) is not MetricRecord:
            return False
        if (type(mr.model) is not str or (mr.horizon is not None and not is_strict_int(mr.horizon))
                or mr.model != exp_m or mr.horizon != exp_h):
            return False
        exp_count = len(origins) * horizon_count if exp_h is None else len(origins)
        if not is_strict_int(mr.count) or mr.count != exp_count:
            return False
        if not _is_finite_float(mr.mae) or mr.mae < 0.0:
            return False
        if not _is_finite_float(mr.rmse) or mr.rmse < 0.0:
            return False
        selected = [sc for sc in scores if sc.model == exp_m and (exp_h is None or sc.horizon == exp_h)]
        try:
            expected_mae = canonicalize_zero(math.fsum(abs(sc.error) for sc in selected) / exp_count)
            expected_rmse = canonicalize_zero(math.sqrt(math.fsum(sc.error * sc.error for sc in selected) / exp_count))
        except (OverflowError, ValueError):
            return False
        if not (math.isfinite(expected_mae) and math.isfinite(expected_rmse)):
            return False
        if mr.mae != expected_mae or mr.rmse != expected_rmse:
            return False

    return True


def run_backtest(
    values: Sequence[float],
    plan: BacktestPlan,
    *,
    start_tick: int = 0,
    step: int = 1,
    unit: str = "synthetic_units",
    budget: Budget = Budget(),
) -> BacktestResult:
    """Execute a deterministic expanding-origin forecast backtest."""
    try:
        if not _is_valid_budget(budget):
            return BacktestResult(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None)

        n_rows = _validate_container_and_ticks(values, start_tick, step, unit)
        if n_rows is None:
            return BacktestResult(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None)

        work = validate_plan_and_compute_work(plan, n_rows)
        if work is None:
            return BacktestResult(Status.INVALID_PLAN, Reason.PLAN_HISTORY, None)

        if (
            n_rows > budget.max_rows
            or work.hgb_fits > budget.max_hgb_fits
            or work.design_cells > budget.max_design_cells
        ):
            return BacktestResult(Status.UNKNOWN_BUDGET, Reason.ADMISSION_BUDGET, None)

        if budget.max_json_bytes == 0:
            return BacktestResult(Status.UNKNOWN_BUDGET, Reason.JSON_BUDGET, None)

        import numpy as np
        import scipy  # noqa: F401
        import sklearn.ensemble

        from threadpoolctl import threadpool_limits

        owned_values = _extract_owned_float64_values(values, n_rows, np)
        if owned_values is None:
            return BacktestResult(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None)

        origins: tuple[int, ...] = plan.origins
        lag_count = plan.lag_count
        horizon_count = plan.horizon_count
        seasonal_period = plan.seasonal_period

        prior_emissions: dict[tuple[str, int], list[tuple[int, float]]] = {
            (model_name, h): []
            for model_name in MODEL_ORDER
            for h in range(1, horizon_count + 1)
        }

        forecasts: list[ForecastRecord] = []
        thread_ctx = threadpool_limits(limits=1)

        with thread_ctx:
            for origin in origins:
                observed_prefix = owned_values[: origin + 1]
                prefix_sha256 = compute_fow_sha256(
                    observed_prefix,
                    start_tick,
                    step,
                    origin + 1,
                )

                for horizon in range(1, horizon_count + 1):
                    target_idx = origin + horizon

                    x_train, y_train, x_query = build_hgb_design_arrays(
                        np,
                        observed_prefix,
                        origin,
                        horizon,
                        lag_count,
                    )

                    estimator = sklearn.ensemble.HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.1,
                        max_iter=_REQUIRED_HGB_ITERATIONS,
                        max_leaf_nodes=15,
                        max_depth=6,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        max_bins=128,
                        categorical_features=None,
                        max_features=1.0,
                        early_stopping=False,
                        warm_start=False,
                        random_state=42,
                    )

                    try:
                        estimator.fit(x_train, y_train)
                    except ValueError:
                        return BacktestResult(
                            Status.FIT_FAILED,
                            Reason.FIT_VALUE_ERROR,
                            None,
                        )

                    if getattr(estimator, "n_iter_", None) != _REQUIRED_HGB_ITERATIONS:
                        raise RuntimeError("Unexpected HGB n_iter_")

                    raw_pred = estimator.predict(x_query)
                    if getattr(raw_pred, "shape", None) != (1,):
                        raise RuntimeError("Unexpected HGB prediction shape")

                    hgb_pred = canonicalize_zero(float(raw_pred[0]))
                    if not math.isfinite(hgb_pred):
                        return BacktestResult(
                            Status.NUMERICALLY_INDETERMINATE,
                            Reason.NUMERIC_RESULT,
                            None,
                        )

                    persistence_pred = canonicalize_zero(float(observed_prefix[origin]))
                    s_idx = compute_seasonal_support_index(
                        origin,
                        horizon,
                        seasonal_period,
                    )
                    seasonal_pred = canonicalize_zero(float(observed_prefix[s_idx]))

                    if not (math.isfinite(persistence_pred) and math.isfinite(seasonal_pred)):
                        return BacktestResult(
                            Status.NUMERICALLY_INDETERMINATE,
                            Reason.NUMERIC_RESULT,
                            None,
                        )

                    model_Step_specs = (
                        (
                            "hgb",
                            hgb_pred,
                            build_hgb_lineage(origin, horizon, lag_count, prefix_sha256),
                        ),
                        (
                            "persistence",
                            persistence_pred,
                            build_baseline_lineage(origin, prefix_sha256),
                        ),
                        (
                            "seasonal",
                            seasonal_pred,
                            build_baseline_lineage(s_idx, prefix_sha256),
                        ),
                    )

                    for model_name, pred_val, lineage in model_Step_specs:
                        history_list = prior_emissions[(model_name, horizon)]
                        band = compute_empirical_band(
                            pred_val,
                            origin,
                            horizon,
                            history_list,
                            observed_prefix,
                        )
                        if band is None:
                            return BacktestResult(
                                Status.NUMERICALLY_INDETERMINATE,
                                Reason.NUMERIC_RESULT,
                                None,
                            )
                        forecasts.append(
                            ForecastRecord(
                                model=model_name,
                                origin=origin,
                                horizon=horizon,
                                target=target_idx,
                                prediction=pred_val,
                                lineage=lineage,
                                band=band,
                            )
                        )
                        history_list.append((origin, pred_val))

        scores: list[ScoreRecord] = []
        scores_by_model: dict[str, list[ScoreRecord]] = {m: [] for m in MODEL_ORDER}
        scores_by_model_horizon: dict[tuple[str, int], list[ScoreRecord]] = {
            (m, h): []
            for m in MODEL_ORDER
            for h in range(1, horizon_count + 1)
        }

        for f in forecasts:
            actual_val = canonicalize_zero(float(owned_values[f.target]))
            error_val = canonicalize_zero(float(actual_val - f.prediction))
            if not (math.isfinite(actual_val) and math.isfinite(error_val)):
                return BacktestResult(
                    Status.NUMERICALLY_INDETERMINATE,
                    Reason.NUMERIC_RESULT,
                    None,
                )
            sc = ScoreRecord(
                model=f.model,
                origin=f.origin,
                horizon=f.horizon,
                target=f.target,
                actual=actual_val,
                error=error_val,
            )
            scores.append(sc)
            scores_by_model[f.model].append(sc)
            scores_by_model_horizon[(f.model, f.horizon)].append(sc)

        metrics: list[MetricRecord] = []
        for model_name in MODEL_ORDER:
            all_model_scores = scores_by_model[model_name]
            overall_count = len(all_model_scores)
            overall_metrics = _metrics_for_scores(all_model_scores)
            if overall_metrics is None:
                return BacktestResult(Status.NUMERICALLY_INDETERMINATE, Reason.NUMERIC_RESULT, None)
            overall_mae, overall_rmse = overall_metrics
            metrics.append(
                MetricRecord(
                    model=model_name,
                    horizon=None,
                    count=overall_count,
                    mae=overall_mae,
                    rmse=overall_rmse,
                )
            )

            for horizon in range(1, horizon_count + 1):
                h_scores = scores_by_model_horizon[(model_name, horizon)]
                h_count = len(h_scores)
                horizon_metrics = _metrics_for_scores(h_scores)
                if horizon_metrics is None:
                    return BacktestResult(Status.NUMERICALLY_INDETERMINATE, Reason.NUMERIC_RESULT, None)
                h_mae, h_rmse = horizon_metrics
                metrics.append(
                    MetricRecord(
                        model=model_name,
                        horizon=horizon,
                        count=h_count,
                        mae=h_mae,
                        rmse=h_rmse,
                    )
                )

        whole_input_sha256 = compute_fow_sha256(
            owned_values,
            start_tick,
            step,
            n_rows,
        )
        report = BacktestReport(
            schema=SCHEMA_VERSION,
            status=Status.COMPLETE,
            unit=unit,
            input=InputSummary(
                count=n_rows,
                start_tick=start_tick,
                step=step,
                sha256=whole_input_sha256,
            ),
            plan=BacktestPlan(
                lag_count=lag_count,
                horizon_count=horizon_count,
                origins=origins,
                seasonal_period=seasonal_period,
            ),
            forecasts=tuple(forecasts),
            scores=tuple(scores),
            metrics=tuple(metrics),
            work=work,
        )

        canonical_json = _encode_canonical_json_bytes(report)
        if len(canonical_json) > budget.max_json_bytes:
            return BacktestResult(Status.UNKNOWN_BUDGET, Reason.JSON_BUDGET, None)

        return BacktestResult(Status.COMPLETE, None, report)

    except MemoryError:
        return BacktestResult(Status.INCOMPLETE, Reason.MEMORY_FAILURE, None)
    except ImportError:
        return BacktestResult(Status.INCOMPLETE, Reason.DEPENDENCY_UNAVAILABLE, None)
    except TimeoutError:
        return BacktestResult(Status.INCOMPLETE, Reason.TIMEOUT, None)
    except OSError:
        return BacktestResult(Status.INCOMPLETE, Reason.IO_FAILURE, None)
    except Exception:
        return BacktestResult(Status.INCOMPLETE, Reason.UNEXPECTED_OPERATIONAL, None)


def _validate_result_for_serialization(
    result: Any,
    max_bytes: Any,
    hard_cap: int,
) -> EncodedReport | None:
    """Validate max_bytes cap and source BacktestResult before encoding."""
    if not is_strict_int(max_bytes) or not (0 <= max_bytes <= hard_cap):
        return EncodedReport(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None, None)

    if type(result) is not BacktestResult:
        return EncodedReport(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None, None)

    if type(result.status) is not Status or result.status not in BACKTEST_RESULT_STATUSES:
        return EncodedReport(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None, None)

    if result.status != Status.COMPLETE:
        if (result.report is not None or type(result.reason) not in (str, Reason)
                or result.reason not in {reason.value for reason in Reason}):
            return EncodedReport(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None, None)
        return EncodedReport(Status.NOT_COMPLETE, Reason.SOURCE_NOT_COMPLETE, None, None)

    if result.reason is not None or not _validate_complete_report(result.report):
        return EncodedReport(Status.INVALID_INPUT, Reason.INPUT_SCHEMA, None, None)

    return None


def serialize_json(
    result: BacktestResult,
    *,
    max_bytes: int = MAX_JSON_BYTES_CAP,
) -> EncodedReport:
    """Serialize a COMPLETE BacktestResult into canonical UTF-8 JSON bytes."""
    try:
        precheck = _validate_result_for_serialization(result, max_bytes, MAX_JSON_BYTES_CAP)
        if precheck is not None:
            return precheck

        assert result.report is not None
        payload = _encode_canonical_json_bytes(result.report)
        req_bytes = len(payload)
        if req_bytes > max_bytes:
            return EncodedReport(
                Status.UNKNOWN_BUDGET,
                Reason.JSON_BUDGET,
                None,
                req_bytes,
            )
        return EncodedReport(Status.COMPLETE, None, payload, req_bytes)
    except MemoryError:
        return EncodedReport(Status.INCOMPLETE, Reason.MEMORY_FAILURE, None, None)
    except ImportError:
        return EncodedReport(Status.INCOMPLETE, Reason.DEPENDENCY_UNAVAILABLE, None, None)
    except TimeoutError:
        return EncodedReport(Status.INCOMPLETE, Reason.TIMEOUT, None, None)
    except OSError:
        return EncodedReport(Status.INCOMPLETE, Reason.IO_FAILURE, None, None)
    except Exception:
        return EncodedReport(Status.INCOMPLETE, Reason.UNEXPECTED_OPERATIONAL, None, None)


def serialize_csv(
    result: BacktestResult,
    *,
    max_bytes: int = MAX_CSV_BYTES_CAP,
) -> EncodedReport:
    """Serialize a COMPLETE BacktestResult into canonical UTF-8 CSV bytes."""
    try:
        precheck = _validate_result_for_serialization(result, max_bytes, MAX_CSV_BYTES_CAP)
        if precheck is not None:
            return precheck

        assert result.report is not None
        payload = _encode_csv_bytes(result.report)
        req_bytes = len(payload)
        if req_bytes > max_bytes:
            return EncodedReport(
                Status.UNKNOWN_BUDGET,
                Reason.CSV_BUDGET,
                None,
                req_bytes,
            )
        return EncodedReport(Status.COMPLETE, None, payload, req_bytes)
    except MemoryError:
        return EncodedReport(Status.INCOMPLETE, Reason.MEMORY_FAILURE, None, None)
    except ImportError:
        return EncodedReport(Status.INCOMPLETE, Reason.DEPENDENCY_UNAVAILABLE, None, None)
    except TimeoutError:
        return EncodedReport(Status.INCOMPLETE, Reason.TIMEOUT, None, None)
    except OSError:
        return EncodedReport(Status.INCOMPLETE, Reason.IO_FAILURE, None, None)
    except Exception:
        return EncodedReport(Status.INCOMPLETE, Reason.UNEXPECTED_OPERATIONAL, None, None)
