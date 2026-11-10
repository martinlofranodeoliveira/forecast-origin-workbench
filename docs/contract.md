# SDK and temporal contract

`run_backtest(values, plan, *, start_tick=0, step=1, unit="synthetic_units",
budget=Budget())` returns a typed `BacktestResult`. Plan fields are lag_count,
horizon_count, origins and seasonal_period. Budget fields are max_rows4096,
max_hgb_fits64, max_design_cells8388608 and max_json_bytes1048576. Lower integer
caps including zero are supported; booleans and unsupported caps are invalid.
Admission checks cheap dimensions/metadata and work counts before element copies
or model fitting. Numeric values are finite binary64 in ±1000000, no missing/bools.
The series has128..4096 regularly spaced values, with safe-integer ticks
start_tick+i*step, positive step and tick magnitude<=9007199254740991.

For origin o, all y[0..o] are observed. x[a]=[y[a],...,y[a-L+1]], training
anchors L-1..o-h, target y[a+h], query x[o]. At least64 training pairs are
required. Horizons are exactly1..H, H<=8; L<=32,1..8 increasing explicit origins,
season period2..32, and every target o+h must exist in the supplied snapshot.
No date calendar, irregular sampling or forecasting beyond that snapshot is implied.

Maintained HGB owns its trees, histogram cuts and predictions. Each direct model
uses squared_error, learning_rate0.1, max_iter64, max_leaf_nodes15, max_depth6,
min_samples_leaf10, l2_regularization1, max_bins128, categorical_features=None,
max_features1, early_stoppingFalse, warm_startFalse, random_state42. Native
threadpools are limited to one. No user estimator/parameter/seed search is accepted.
Persistence predicts y[o]. Seasonal predicts y[s] where
s=o+h-ceil(h/P)*P<=o, including h>P. Only HGB has APPLICABLE training fields;
baselines use NOT_APPLICABLE/all four training fields null and support[o]/[s].
Up to64 HGB fits yield192 cells across the three models.

Each prefix digest is SHA256 of ASCII FOWprefix1 plus NUL; struct<qQI
(start_tick,step,count); then observed little-endian float64 values, -0→+0.
count=o+1 for issued receipts and n for the separate whole-input digest.

For each(model,h), earlier r is usable iff r<o and r+h<=o. The band is frozen
before the new forecast's target is observed. m<3 means unavailable and null
rank/radius/ends; otherwise k=ceil(4m/5), q=kth sorted absolute error and ends
prediction±q. There is no interpolation/clipping. An available q0 band differs
from unavailable. At most7 errors are used; no coverage/confidence guarantee.

Scores are joined only after issuance. MAE=fsum(|error|)/count;
RMSE=sqrt(fsum(error²)/count), overall equally weighted by cells. Finite binary64
and stable sums are used. Owned metric overflow/nonfinite values are numerically
indeterminate, with no fallback. Changing only values after cutoff c preserves
issued forecasts/bands/lineage/prefixes at origins<=c, while future-target scores,
aggregate metrics and whole-input hash may change. Overlapping cells are not
independent observations.

Run statuses: COMPLETE, INVALID_INPUT, INVALID_PLAN, FIT_FAILED,
NUMERICALLY_INDETERMINATE, UNKNOWN_BUDGET, INCOMPLETE. Only COMPLETE has a report.
FIT_FAILED is a ValueError inside the fixed estimator.fit boundary. Missing
dependencies, MemoryError, OSError, timeout and unexpected SDK failures remain
INCOMPLETE; numerical exceptions are not broadly relabeled as fit failures.
A killed process may return nothing. Reports are immutable under ordinary API
access, without hostile in-process object/monkeypatch guarantees.

serialize_json/serialize_csv return EncodedReport(status,reason,data,required_bytes).
Only COMPLETE has complete owned bytes. Other statuses are INVALID_INPUT,
NOT_COMPLETE, UNKNOWN_BUDGET or INCOMPLETE. Malformed public DTO types and
lineage/band/score/count relations are rejected; this does not certify a
consistently fabricated caller-created prediction. Canonical JSON uses sorted
keys, compact separators, ASCII escaping, no NaN and one final LF. CSV has the
fixed21-column header, LF, null as empty cells and compact JSON list cells.
Each emitted format is bounded separately to1MiB including LF. SDK canonical JSON
must already fit before run COMPLETE. Logical counters do not bound all native
allocations or elapsed time.
