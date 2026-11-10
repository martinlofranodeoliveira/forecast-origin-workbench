"""Comprehensive contract verification tests for Forecast Origin Workbench."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal, localcontext
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from forecast_origin_workbench import (
    BacktestPlan,
    BacktestResult,
    BandStatus,
    Budget,
    EncodedReport,
    Status,
    run_backtest,
    serialize_csv,
    serialize_json,
)

EXPECTED_CONSTANT_CSV = (
    "model,origin,horizon,target,prediction,training_status,anchor_start,anchor_end,"
    "training_rows,latest_training_target,observed_support,prefix_sha256,band_status,"
    "band_count,band_rank,band_radius,band_lower,band_upper,eligible_residual_origins,"
    "actual,error\n"
    "hgb,64,1,65,7.0,APPLICABLE,0,63,64,64,[64],"
    "211970272cbe440db9f91626330e1b657827887aeed68ec82ac433c374186e3f,"
    "UNAVAILABLE_INSUFFICIENT_HISTORY,0,,,,,[],7.0,0.0\n"
    "persistence,64,1,65,7.0,NOT_APPLICABLE,,,,,[64],"
    "211970272cbe440db9f91626330e1b657827887aeed68ec82ac433c374186e3f,"
    "UNAVAILABLE_INSUFFICIENT_HISTORY,0,,,,,[],7.0,0.0\n"
    "seasonal,64,1,65,7.0,NOT_APPLICABLE,,,,,[62],"
    "211970272cbe440db9f91626330e1b657827887aeed68ec82ac433c374186e3f,"
    "UNAVAILABLE_INSUFFICIENT_HISTORY,0,,,,,[],7.0,0.0\n"
)


def _make_structured_series() -> list[float]:
    """Create a deterministic 128-row series with explicit baseline values."""
    series = [float((idx % 5) * 2 + idx * 0.25) for idx in range(128)]
    series[67] = 15.0
    series[68] = 14.0
    series[69] = 18.0
    series[70] = 16.0
    series[71] = 21.0
    series[72] = 19.0
    series[73] = 24.0
    series[74] = 22.0
    series[75] = 27.0
    return series


class _NoAccessSequence(Sequence[float]):
    """Sequence wrapper that fails if any element is read."""

    def __init__(self, length: int) -> None:
        self._length = length
        self.accessed = False

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> float:  # type: ignore[override]
        self.accessed = True
        raise AssertionError("Element access forbidden before lower budget admission")


def test_constant_golden_wire_json_and_csv() -> None:
    values = [7.0] * 128
    plan = BacktestPlan(lag_count=1, horizon_count=1, origins=[64], seasonal_period=3)

    result = run_backtest(values, plan)
    assert result.status == Status.COMPLETE
    assert result.reason is None
    assert result.report is not None

    report = result.report
    assert report.schema == "forecast-origin-workbench/1"
    assert report.unit == "synthetic_units"
    assert report.input.count == 128
    assert report.input.start_tick == 0
    assert report.input.step == 1
    assert (
        report.input.sha256
        == "f799155f5d476e17faee0f5519653477b0e2aebdd634ed93a18bd6c441a9a273"
    )
    assert report.work.hgb_fits == 1
    assert report.work.design_cells == 65
    assert report.work.forecast_cells == 3

    expected_prefix = "211970272cbe440db9f91626330e1b657827887aeed68ec82ac433c374186e3f"
    for f in report.forecasts:
        assert f.prediction == 7.0
        assert f.lineage.prefix_sha256 == expected_prefix
        assert f.band.status == BandStatus.UNAVAILABLE_INSUFFICIENT_HISTORY
        assert f.band.count == 0
        assert f.band.rank is None
        assert f.band.radius is None
        assert f.band.lower is None
        assert f.band.upper is None
        assert f.band.eligible_origins == ()

    json_enc = serialize_json(result)
    assert json_enc.status == Status.COMPLETE
    assert json_enc.reason is None
    assert json_enc.required_bytes == 2360
    assert json_enc.data is not None
    assert len(json_enc.data) == 2360
    assert json_enc.data.endswith(b"\n")
    assert (
        hashlib.sha256(json_enc.data).hexdigest()
        == "ad6201f9f928c180027ea528bf05fe7b4b8290f688cb67c5aff6af5c1a17f4fc"
    )

    csv_enc = serialize_csv(result)
    assert csv_enc.status == Status.COMPLETE
    assert csv_enc.reason is None
    assert csv_enc.required_bytes == 734
    assert csv_enc.data is not None
    assert len(csv_enc.data) == 734
    assert csv_enc.data.decode("utf-8") == EXPECTED_CONSTANT_CSV
    assert (
        hashlib.sha256(csv_enc.data).hexdigest()
        == "78634e2f366759c0afaeed871c9695a1361804079d1c1326b50dd933713a22b6"
    )


def test_zero_radius_available_band_vs_insufficient_history() -> None:
    values = [7.0] * 128
    plan = BacktestPlan(
        lag_count=1,
        horizon_count=1,
        origins=[64, 66, 68, 70],
        seasonal_period=3,
    )
    result = run_backtest(values, plan)
    assert result.status == Status.COMPLETE
    assert result.report is not None

    for f in result.report.forecasts:
        if f.origin == 64:
            assert f.band.status == BandStatus.UNAVAILABLE_INSUFFICIENT_HISTORY
            assert f.band.count == 0
            assert f.band.rank is None
            assert f.band.radius is None
            assert f.band.lower is None
            assert f.band.upper is None
            assert f.band.eligible_origins == ()
        elif f.origin == 70:
            assert f.band.status == BandStatus.AVAILABLE
            assert f.band.count == 3
            assert f.band.rank == 3
            assert f.band.radius == 0.0
            assert f.band.lower == 7.0
            assert f.band.upper == 7.0
            assert f.band.eligible_origins == (64, 66, 68)


def test_baseline_lineage_and_multi_origin_bands_and_decimal_metrics() -> None:
    values = _make_structured_series()
    plan = BacktestPlan(
        lag_count=3,
        horizon_count=2,
        origins=[67, 69, 71, 73],
        seasonal_period=3,
    )
    result = run_backtest(values, plan)
    assert result.status == Status.COMPLETE
    assert result.report is not None
    report = result.report

    assert report.work.hgb_fits == 8
    assert report.work.forecast_cells == 24
    assert report.work.design_cells == 1644

    by_key = {(f.origin, f.horizon, f.model): f for f in report.forecasts}

    hgb_69_2 = by_key[(69, 2, "hgb")]
    assert hgb_69_2.lineage.training_status == "APPLICABLE"
    assert hgb_69_2.lineage.anchor_start == 2
    assert hgb_69_2.lineage.anchor_end == 67
    assert hgb_69_2.lineage.training_rows == 66
    assert hgb_69_2.lineage.latest_training_target == 69
    assert hgb_69_2.lineage.observed_support == (69, 68, 67)

    pers_69_2 = by_key[(69, 2, "persistence")]
    assert pers_69_2.prediction == 18.0
    assert pers_69_2.lineage.training_status == "NOT_APPLICABLE"
    assert pers_69_2.lineage.anchor_start is None
    assert pers_69_2.lineage.anchor_end is None
    assert pers_69_2.lineage.training_rows is None
    assert pers_69_2.lineage.latest_training_target is None
    assert pers_69_2.lineage.observed_support == (69,)

    seas_69_2 = by_key[(69, 2, "seasonal")]
    assert seas_69_2.prediction == 14.0
    assert seas_69_2.lineage.training_status == "NOT_APPLICABLE"
    assert seas_69_2.lineage.anchor_start is None
    assert seas_69_2.lineage.anchor_end is None
    assert seas_69_2.lineage.training_rows is None
    assert seas_69_2.lineage.latest_training_target is None
    assert seas_69_2.lineage.observed_support == (68,)

    assert (
        hgb_69_2.lineage.prefix_sha256
        == pers_69_2.lineage.prefix_sha256
        == seas_69_2.lineage.prefix_sha256
    )

    pers_71_2 = by_key[(71, 2, "persistence")]
    assert pers_71_2.band.status == BandStatus.UNAVAILABLE_INSUFFICIENT_HISTORY
    assert pers_71_2.band.count == 2
    assert pers_71_2.band.eligible_origins == (67, 69)
    assert pers_71_2.band.rank is None
    assert pers_71_2.band.radius is None
    assert pers_71_2.band.lower is None
    assert pers_71_2.band.upper is None

    pers_73_2 = by_key[(73, 2, "persistence")]
    assert pers_73_2.prediction == 24.0
    assert pers_73_2.band.status == BandStatus.AVAILABLE
    assert pers_73_2.band.count == 3
    assert pers_73_2.band.rank == 3
    assert pers_73_2.band.radius == 3.0
    assert pers_73_2.band.lower == 21.0
    assert pers_73_2.band.upper == 27.0
    assert pers_73_2.band.eligible_origins == (67, 69, 71)

    with localcontext() as ctx:
        ctx.prec = 70
        for m_rec in report.metrics:
            matching_scores = [
                s
                for s in report.scores
                if s.model == m_rec.model
                and (m_rec.horizon is None or s.horizon == m_rec.horizon)
            ]
            assert len(matching_scores) == m_rec.count
            d_count = Decimal(m_rec.count)
            d_errors = [
                Decimal.from_float(s.actual) - Decimal.from_float(
                    by_key[(s.origin, s.horizon, s.model)].prediction
                )
                for s in matching_scores
            ]
            mae_ref = sum((abs(e) for e in d_errors), Decimal(0)) / d_count
            rmse_ref = (sum((e * e for e in d_errors), Decimal(0)) / d_count).sqrt()

            tol_mae = Decimal("1e-12") * max(Decimal(1), abs(mae_ref))
            tol_rmse = Decimal("1e-12") * max(Decimal(1), abs(rmse_ref))
            assert abs(Decimal.from_float(m_rec.mae) - mae_ref) <= tol_mae
            assert abs(Decimal.from_float(m_rec.rmse) - rmse_ref) <= tol_rmse


def test_seasonal_support_when_horizon_exceeds_period() -> None:
    values = _make_structured_series()
    plan = BacktestPlan(
        lag_count=2,
        horizon_count=5,
        origins=[70],
        seasonal_period=2,
    )
    result = run_backtest(values, plan)
    assert result.status == Status.COMPLETE
    assert result.report is not None
    seas_h5 = [
        f
        for f in result.report.forecasts
        if f.model == "seasonal" and f.origin == 70 and f.horizon == 5
    ][0]
    assert seas_h5.lineage.observed_support == (69,)
    assert seas_h5.prediction == values[69]


def test_work_budget_admission_and_no_copy_or_fit_before_lower_admission() -> None:
    values = _make_structured_series()
    plan = BacktestPlan(
        lag_count=3,
        horizon_count=2,
        origins=[67, 69, 71, 73],
        seasonal_period=3,
    )

    exact = run_backtest(
        values,
        plan,
        budget=Budget(max_rows=128, max_hgb_fits=8, max_design_cells=1644),
    )
    assert exact.status == Status.COMPLETE

    rejected_budgets = [
        (Budget(max_rows=0), "ADMISSION_BUDGET"),
        (Budget(max_hgb_fits=0), "ADMISSION_BUDGET"),
        (Budget(max_design_cells=0), "ADMISSION_BUDGET"),
        (Budget(max_rows=127), "ADMISSION_BUDGET"),
        (Budget(max_hgb_fits=7), "ADMISSION_BUDGET"),
        (Budget(max_design_cells=1643), "ADMISSION_BUDGET"),
        (Budget(max_json_bytes=0), "JSON_BUDGET"),
    ]

    for low_budget, expected_reason in rejected_budgets:
        guard_seq = _NoAccessSequence(128)
        with patch(
            "sklearn.ensemble.HistGradientBoostingRegressor.fit",
            side_effect=AssertionError("Fit must not be called"),
        ):
            res = run_backtest(guard_seq, plan, budget=low_budget)
        assert res.status == Status.UNKNOWN_BUDGET
        assert res.reason == expected_reason
        assert res.report is None
        assert not guard_seq.accessed


def test_byte_budget_caps_and_separate_format_accounting() -> None:
    values = [7.0] * 128
    plan = BacktestPlan(lag_count=1, horizon_count=1, origins=[64], seasonal_period=3)

    for cap, exp_status in [(0, Status.UNKNOWN_BUDGET), (2359, Status.UNKNOWN_BUDGET), (2360, Status.COMPLETE)]:
        run_res = run_backtest(values, plan, budget=Budget(max_json_bytes=cap))
        assert run_res.status == exp_status
        if exp_status != Status.COMPLETE:
            assert run_res.reason == "JSON_BUDGET"
            assert run_res.report is None

    complete_res = run_backtest(values, plan, budget=Budget(max_json_bytes=2360))
    assert complete_res.status == Status.COMPLETE

    for cap, exp_status in [(0, Status.UNKNOWN_BUDGET), (2359, Status.UNKNOWN_BUDGET), (2360, Status.COMPLETE)]:
        enc_j = serialize_json(complete_res, max_bytes=cap)
        assert enc_j.status == exp_status
        assert enc_j.required_bytes == 2360
        if exp_status == Status.COMPLETE:
            assert enc_j.data is not None and len(enc_j.data) == 2360
        else:
            assert enc_j.reason == "JSON_BUDGET"
            assert enc_j.data is None

    for cap, exp_status in [(0, Status.UNKNOWN_BUDGET), (733, Status.UNKNOWN_BUDGET), (734, Status.COMPLETE)]:
        enc_c = serialize_csv(complete_res, max_bytes=cap)
        assert enc_c.status == exp_status
        assert enc_c.required_bytes == 734
        if exp_status == Status.COMPLETE:
            assert enc_c.data is not None and len(enc_c.data) == 734
        else:
            assert enc_c.reason == "CSV_BUDGET"
            assert enc_c.data is None

    both_json = serialize_json(complete_res, max_bytes=2360)
    both_csv = serialize_csv(complete_res, max_bytes=2360)
    assert both_json.status == Status.COMPLETE
    assert both_csv.status == Status.COMPLETE
    assert (both_json.required_bytes or 0) + (both_csv.required_bytes or 0) == 3094

    failed_canonical = run_backtest(values, plan, budget=Budget(max_json_bytes=734))
    assert failed_canonical.status == Status.UNKNOWN_BUDGET
    assert failed_canonical.reason == "JSON_BUDGET"
    csv_from_failed = serialize_csv(failed_canonical, max_bytes=734)
    assert csv_from_failed.status == Status.NOT_COMPLETE
    assert csv_from_failed.reason == "SOURCE_NOT_COMPLETE"
    assert csv_from_failed.data is None
    assert csv_from_failed.required_bytes is None

    for bad_cap in [True, False, -1, 1_048_577, 100.0, "2360"]:
        assert serialize_json(complete_res, max_bytes=bad_cap).status == Status.INVALID_INPUT  # type: ignore[arg-type]
        assert serialize_csv(complete_res, max_bytes=bad_cap).status == Status.INVALID_INPUT  # type: ignore[arg-type]


def test_all_origin_future_mutation_invariance() -> None:
    base_values = _make_structured_series()
    plan = BacktestPlan(
        lag_count=3,
        horizon_count=2,
        origins=[67, 69, 71, 73],
        seasonal_period=3,
    )
    base_res = run_backtest(base_values, plan)
    assert base_res.status == Status.COMPLETE
    assert base_res.report is not None

    for cutoff in plan.origins:
        mutated = list(base_values)
        for idx in range(cutoff + 1, len(mutated)):
            mutated[idx] = mutated[idx] + 250.0 + float(idx)

        mut_res = run_backtest(mutated, plan)
        assert mut_res.status == Status.COMPLETE
        assert mut_res.report is not None

        base_issued = [f for f in base_res.report.forecasts if f.origin <= cutoff]
        mut_issued = [f for f in mut_res.report.forecasts if f.origin <= cutoff]
        assert base_issued == mut_issued

        base_past_scores = [s for s in base_res.report.scores if s.target <= cutoff]
        mut_past_scores = [s for s in mut_res.report.scores if s.target <= cutoff]
        assert base_past_scores == mut_past_scores

        base_future_scores = [s for s in base_res.report.scores if s.target > cutoff]
        mut_future_scores = [s for s in mut_res.report.scores if s.target > cutoff]
        assert base_future_scores != mut_future_scores
        assert base_res.report.input.sha256 != mut_res.report.input.sha256


def test_injected_exceptions_and_numerical_failures() -> None:
    values = [7.0] * 128
    plan = BacktestPlan(lag_count=1, horizon_count=1, origins=[64], seasonal_period=3)

    with patch(
        "sklearn.ensemble.HistGradientBoostingRegressor.fit",
        side_effect=ValueError("fit domain failure"),
    ):
        res = run_backtest(values, plan)
    assert res.status == Status.FIT_FAILED
    assert res.reason == "FIT_VALUE_ERROR"
    assert res.report is None

    with patch(
        "sklearn.ensemble.HistGradientBoostingRegressor.predict",
        side_effect=ValueError("predict failure"),
    ):
        res = run_backtest(values, plan)
    assert res.status == Status.INCOMPLETE
    assert res.reason == "UNEXPECTED_OPERATIONAL"
    assert res.report is None

    exception_cases = [
        (MemoryError("oom"), Status.INCOMPLETE, "MEMORY_FAILURE"),
        (ImportError("missing sdk"), Status.INCOMPLETE, "DEPENDENCY_UNAVAILABLE"),
        (TimeoutError("timed out"), Status.INCOMPLETE, "TIMEOUT"),
        (OSError("io error"), Status.INCOMPLETE, "IO_FAILURE"),
        (RuntimeError("runtime bug"), Status.INCOMPLETE, "UNEXPECTED_OPERATIONAL"),
    ]
    for exc, exp_status, exp_reason in exception_cases:
        with patch(
            "sklearn.ensemble.HistGradientBoostingRegressor.fit",
            side_effect=exc,
        ):
            res = run_backtest(values, plan)
        assert res.status == exp_status
        assert res.reason == exp_reason
        assert res.report is None

    with patch(
        "sklearn.ensemble.HistGradientBoostingRegressor.predict",
        return_value=np.array([float("nan")], dtype=np.float64),
    ):
        res = run_backtest(values, plan)
    assert res.status == Status.NUMERICALLY_INDETERMINATE
    assert res.reason == "NUMERIC_RESULT"
    assert res.report is None

    with patch(
        "sklearn.ensemble.HistGradientBoostingRegressor.predict",
        return_value=np.array([7.0, 7.0], dtype=np.float64),
    ):
        res = run_backtest(values, plan)
    assert res.status == Status.INCOMPLETE
    assert res.reason == "UNEXPECTED_OPERATIONAL"


def test_input_and_plan_validation_and_forged_serialization_rejection() -> None:
    valid_values = [7.0] * 128
    valid_plan = BacktestPlan(lag_count=1, horizon_count=1, origins=[64], seasonal_period=3)

    bad_inputs: list[tuple[object, dict[str, object]]] = [
        ([7.0] * 127, {}),
        ([7.0] * 4097, {}),
        ([True] * 128, {}),
        (["7.0"] * 128, {}),
        ([float("nan")] + [7.0] * 127, {}),
        ([float("inf")] + [7.0] * 127, {}),
        ([1_000_000.1] + [7.0] * 127, {}),
        (valid_values, {"unit": ""}),
        (valid_values, {"unit": "bad-unit!"}),
        (valid_values, {"step": 0}),
        (valid_values, {"step": True}),
        (valid_values, {"start_tick": 9_007_199_254_740_991}),
        (valid_values, {"budget": Budget(max_rows=-1)}),
        (valid_values, {"budget": Budget(max_rows=True)}),  # type: ignore[arg-type]
        (valid_values, {"budget": Budget(max_hgb_fits=65)}),
    ]
    for vals, kwargs in bad_inputs:
        res = run_backtest(vals, valid_plan, **kwargs)  # type: ignore[arg-type]
        assert res.status == Status.INVALID_INPUT
        assert res.reason == "INPUT_SCHEMA"
        assert res.report is None

    bad_plans = [
        BacktestPlan(lag_count=0, horizon_count=1, origins=[64], seasonal_period=3),
        BacktestPlan(lag_count=33, horizon_count=1, origins=[64], seasonal_period=3),
        BacktestPlan(lag_count=1, horizon_count=0, origins=[64], seasonal_period=3),
        BacktestPlan(lag_count=1, horizon_count=9, origins=[64], seasonal_period=3),
        BacktestPlan(lag_count=1, horizon_count=1, origins=[64], seasonal_period=1),
        BacktestPlan(lag_count=1, horizon_count=1, origins=[], seasonal_period=3),
        BacktestPlan(lag_count=1, horizon_count=1, origins=[64, 64], seasonal_period=3),
        BacktestPlan(lag_count=1, horizon_count=1, origins=[66, 64], seasonal_period=3),
        BacktestPlan(lag_count=1, horizon_count=1, origins=[62], seasonal_period=3),
        BacktestPlan(lag_count=1, horizon_count=1, origins=[127], seasonal_period=3),
    ]
    for bad_plan in bad_plans:
        res = run_backtest(valid_values, bad_plan)
        assert res.status == Status.INVALID_PLAN
        assert res.reason == "PLAN_HISTORY"
        assert res.report is None

    good_res = run_backtest(valid_values, valid_plan)
    assert good_res.status == Status.COMPLETE and good_res.report is not None

    forged_none_report = BacktestResult(Status.COMPLETE, None, None)
    assert serialize_json(forged_none_report).status == Status.INVALID_INPUT
    assert serialize_csv(forged_none_report).status == Status.INVALID_INPUT

    bad_pers_forecast = replace(
        good_res.report.forecasts[1],
        lineage=replace(good_res.report.forecasts[1].lineage, anchor_start=0),
    )
    forged_report = replace(
        good_res.report,
        forecasts=(good_res.report.forecasts[0], bad_pers_forecast, good_res.report.forecasts[2]),
    )
    forged_res = BacktestResult(Status.COMPLETE, None, forged_report)
    assert serialize_json(forged_res).status == Status.INVALID_INPUT
    assert serialize_csv(forged_res).status == Status.INVALID_INPUT


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_numpy_1d_input_accepted(dtype: object) -> None:
    arr = np.full(128, 7, dtype=dtype)
    plan = BacktestPlan(lag_count=1, horizon_count=1, origins=[64], seasonal_period=3)
    res = run_backtest(arr, plan)
    assert res.status == Status.COMPLETE
