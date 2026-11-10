# Forecast Origin Workbench

Development **0.1.0**. A Python SDK and local CLI for inspecting expanding-origin
backtests on one complete, regularly spaced numerical series. Maintained
scikit-learn learns a fixed histogram gradient-boosting regressor; persistence
and an explicitly configured seasonal baseline stay visible alongside it.

Every horizon trains only on targets already observed at its forecast origin.
Empirical error bands use earlier out-of-sample forecasts whose targets have
also become observable. This is an experiment tool, without a coverage guarantee
or automatic model-selection recommendation.

[Português](README.pt-BR.md) · [Exact contract](docs/contract.md)

## Install and try

The qualified tuple is Linux x86_64, CPython3.13.18, scikit-learn1.9.1,
NumPy2.5.4 and SciPy1.18.1. Other tuples have not been qualified. The exact
runtime and development wheels have separate hash locks.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-build-isolation --no-deps .
forecast-origin-workbench demo --assert
forecast-origin-workbench demo --assert --format csv
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

The synthetic demo uses 128 values and four fixed origins. `--assert` checks
literal baseline/support/band results and remains active under Python `-O`.
There is no runtime network access, model download, service, model file or
arbitrary estimator option.

```sh
forecast-origin-workbench backtest --input examples/constant.json
forecast-origin-workbench backtest --input examples/constant.json --format csv
```

`--input -` reads bounded stdin. The closed input contains schema
`forecast-origin-workbench-input/1`, `values`, and `plan` with `lag_count`,
`horizon_count`, `origins`, `seasonal_period`. Optional keys are `start_tick`,
`step`, `unit`, `budget`. [The complete example](examples/constant.json) has all
128 values. File inputs are bounded regular snapshots; leaf symlinks, FIFOs,
directories and remote URIs are rejected. Duplicate keys, JSON nonfinite values,
unknown keys and unsupported scalar types are rejected with stable codes.

## Use the SDK

```python
from forecast_origin_workbench import BacktestPlan, Status, run_backtest, serialize_json

values = [7.0] * 128
plan = BacktestPlan(1, 1, (64,), 3)
result = run_backtest(values, plan)
if result.status != Status.COMPLETE:
    raise RuntimeError(result.reason)
for metric in result.report.metrics:
    print(metric.model, metric.horizon, metric.mae, metric.rmse)
encoded = serialize_json(result)
assert encoded.status == Status.COMPLETE
print(len(encoded.data))  # 2360 bytes, including the final LF for this literal case
```

`BacktestPlan` and `Budget` are passive immutable metadata; `run_backtest`
validates their fields. Reports contain immutable records and owned values.
Models remain private/in memory and are never serialized.

## Read the evidence

Origin `o` means observations through `y[o]` already exist. For horizon `h`,
anchor `a` is eligible only when `a+h <= o`. Features are the latest `L` observed
values in descending index order. Each origin/horizon fits a new direct HGB
model with64 iterations, no early stopping, tuning, recursive predictions or
seed search. Only HGB has training-anchor receipts. Baselines have
`NOT_APPLICABLE` and null training fields, with their actual observed support
index instead. All models use identical forecast cells.

Per model/horizon, a previous prediction from `r` can contribute a residual only
when `r < o` and `r+h <= o`. Fewer than3 residuals gives
`UNAVAILABLE_INSUFFICIENT_HISTORY` and null radius/ends. Otherwise the band uses
the exact nearest rank `ceil(4m/5)` of absolute errors; enough zero residuals can
produce an **available** radius0 band. At most7 residuals exist in this bounded
experiment. The4/5 parameter is not achieved80% coverage, a conformal confidence
claim or evidence of independence. Regime changes can invalidate past-error
behavior.

MAE/RMSE retain the synthetic target units. Overall metrics weight forecast
cells equally; overlapping horizons can share target times. The learned model
can be worse than either baseline. JSON retains plan, input/prefix hashes,
lineage, raw predictions, actuals, errors and metrics. CSV contains forecast and
score rows; retain the JSON for full plan/unit/provenance context.

| Boundary | Maximum |
| --- | --- |
| Values |128–4096, finite, magnitude<=1000000, no missing/bools |
| Lags / horizons / origins |32 /8 /8; at least64 training pairs per HGB fit |
| HGB fits / forecast cells |64 /192 across three models |
| Season period |2–32 explicitly supplied sample periods |
| Input JSON |524288 bytes |
| Canonical SDK JSON / emitted JSON / emitted CSV |1048576 bytes **each**, including LF |

Canonical JSON admission occurs before SDK `COMPLETE`, even if CSV will later
be selected. `--max-output-bytes` can reduce the chosen encoding's separate
limit. Input and SDK failures take precedence even when this CLI limit is zero.
The independent SDK `budget.max_json_bytes=0` still stops before copying/fitting.
No combined JSON+CSV allowance is used. A failed encoder returns no
successful output bytes. Logical row/work limits are not universal RSS or
elapsed-time guarantees.

Normal commands emit complete JSON/CSV to stdout; errors emit closed JSON to
stderr with no input path/value or dependency exception text. Exit0 means
complete output (or help),2 invalid input/plan/usage,3 budget exhaustion,4
fit/incomplete/IO failure,5 numerical indeterminacy. `FIT_FAILED` is distinct
from operational `INCOMPLETE`. A killed/OOM process cannot promise any returned
status. Although output is staged first, stdout can fail after accepting a
prefix; discard output unless exit is0. Stderr can also fail to deliver a diagnostic; the nonzero exit code remains the outcome. Shell redirection itself can create or
truncate a file before the command runs.

## Reproduce the package gates

```sh
sh scripts/check.sh
```

Only explicit dependency preparation accesses official PyPI, with exact hashes.
SDK/test and rebuilt-wheel runtime containers use network-none, no host bind,
UID/GID1000, all capabilities dropped, no-new-privileges, PID256,1GiB and2CPU.
The fresh runtime has no checkout/compiler/build backend; it checks actual
HGB fits, temporal index receipts, mature residuals, independent Decimal
metrics, future mutations, byte goldens and the installed CLI. Evidence and
artifacts are retained under ignored `dist/`. Each gate first removes this
project's generated `dist/`, so a failed run cannot retain earlier evidence.
A symlink or non-directory at `dist/` is rejected without deletion. Local gates and exact-head hosted
CI are separate; a workflow file is not evidence that hosted CI ran.

MIT covers original Lofrano source only. Exact upstream wheel/native/base notices
are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the package's
`_notices` data. [Lofrano Tecnologia](https://lofranotecnologia.com.br).
