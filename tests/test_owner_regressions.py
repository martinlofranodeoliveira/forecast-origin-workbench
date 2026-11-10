"""Independent regressions for concrete native REDs in the preserved Argon core."""
import builtins
from dataclasses import replace
from unittest.mock import patch
import pytest
import sklearn.ensemble
from forecast_origin_workbench import BacktestPlan,BacktestResult,Status,run_backtest,serialize_json,serialize_csv

@pytest.fixture(scope='module')
def constant():
 r=run_backtest([7.0]*128,BacktestPlan(1,1,(64,),3))
 assert r.status==Status.COMPLETE
 return r

@pytest.fixture(scope='module')
def matured():
 r=run_backtest([7.0]*128,BacktestPlan(1,1,(64,66,68,70),3))
 assert r.status==Status.COMPLETE
 return r

@pytest.mark.parametrize('encode',[serialize_json,serialize_csv])
@pytest.mark.parametrize('case',['forecast_horizon_bool','score_horizon_bool','fit_count_bool','design_count_float','metric_horizon_bool','metric_disagrees','tiny_score_error','nonenum_complete','bad_failure_reason'])
def test_public_dto_scalar_and_score_relations(constant,encode,case):
 r=constant.report
 if case=='forecast_horizon_bool':r=replace(r,forecasts=(replace(r.forecasts[0],horizon=True),*r.forecasts[1:]))
 elif case=='score_horizon_bool':r=replace(r,scores=(replace(r.scores[0],horizon=True),*r.scores[1:]))
 elif case=='fit_count_bool':r=replace(r,work=replace(r.work,hgb_fits=True))
 elif case=='design_count_float':r=replace(r,work=replace(r.work,design_cells=65.0))
 elif case=='metric_horizon_bool':r=replace(r,metrics=(r.metrics[0],replace(r.metrics[1],horizon=True),*r.metrics[2:]))
 elif case=='metric_disagrees':r=replace(r,metrics=(replace(r.metrics[0],mae=123.0,rmse=123.0),*r.metrics[1:]))
 elif case=='tiny_score_error':r=replace(r,scores=(replace(r.scores[0],error=1e-10),*r.scores[1:]))
 forged=BacktestResult(Status.COMPLETE,None,r)
 if case=='nonenum_complete':forged=BacktestResult('COMPLETE',None,r)
 if case=='bad_failure_reason':forged=BacktestResult(Status.INVALID_INPUT,'arbitrary_private_detail',None)
 e=encode(forged)
 assert e.status==Status.INVALID_INPUT and e.data is None
 assert e.reason=='INPUT_SCHEMA'

@pytest.mark.parametrize('encode',[serialize_json,serialize_csv])
def test_matured_quantile_is_replayed_from_report_scores(matured,encode):
 forecasts=list(matured.report.forecasts)
 i=next(i for i,f in enumerate(forecasts) if f.origin==70 and f.model=='hgb')
 f=forecasts[i]
 forecasts[i]=replace(f,band=replace(f.band,radius=99.0,lower=f.prediction-99,upper=f.prediction+99))
 r=replace(matured.report,forecasts=tuple(forecasts))
 encoded=encode(BacktestResult(Status.COMPLETE,None,r))
 assert encoded.status==Status.INVALID_INPUT and encoded.data is None

@pytest.mark.parametrize('encode',[serialize_json,serialize_csv])
def test_same_target_cannot_have_inconsistent_actuals(constant,encode):
 r=constant.report
 score=replace(r.scores[0],actual=8.0,error=1.0)
 # Preserve that model's score/metric arithmetic, isolate the cross-target relation.
 metrics=tuple(replace(m,mae=1.0,rmse=1.0) if m.model=='hgb' else m for m in r.metrics)
 forged=replace(r,scores=(score,*r.scores[1:]),metrics=metrics)
 e=encode(BacktestResult(Status.COMPLETE,None,forged))
 assert e.status==Status.INVALID_INPUT and e.data is None

def test_missing_required_threadpoolctl_is_operational_incomplete():
 original=builtins.__import__
 def missing(name,globals=None,locals=None,fromlist=(),level=0):
  if name=='threadpoolctl' and globals and globals.get('__name__')=='forecast_origin_workbench.core':raise ImportError('controlled unavailable dependency')
  return original(name,globals,locals,fromlist,level)
 with patch('builtins.__import__',side_effect=missing):r=run_backtest([7.0]*128,BacktestPlan(1,1,(64,),3))
 assert r.status==Status.INCOMPLETE and r.reason=='DEPENDENCY_UNAVAILABLE' and r.report is None

@pytest.mark.parametrize('error,reason',[(MemoryError('controlled'),'MEMORY_FAILURE'),(OSError('controlled'),'IO_FAILURE')])
def test_ordinary_input_failure_is_not_invalid_domain(error,reason):
 class FailingSequence:
  def __len__(self):raise error
  def __getitem__(self,index):return 7.0
 result=run_backtest(FailingSequence(),BacktestPlan(1,1,(64,),3))
 assert result.status==Status.INCOMPLETE and result.reason==reason and result.report is None

@pytest.mark.parametrize('origins',[(64,),(64,66)])
def test_owned_metric_nonfinite_or_overflow_is_numerically_indeterminate(origins):
 import numpy as np
 # Maintained real .fit is still called; only its finite prediction is overridden.
 with patch.object(sklearn.ensemble.HistGradientBoostingRegressor,'predict',return_value=np.array([1e308])):
  result=run_backtest([7.0]*128,BacktestPlan(1,1,origins,3))
 assert result.status==Status.NUMERICALLY_INDETERMINATE and result.reason=='NUMERIC_RESULT'
 assert result.report is None

def test_arbitrary_sdk_overflow_remains_operational():
 with patch.object(sklearn.ensemble.HistGradientBoostingRegressor,'fit',side_effect=OverflowError('controlled SDK overflow')):
  result=run_backtest([7.0]*128,BacktestPlan(1,1,(64,),3))
 assert result.status==Status.INCOMPLETE and result.reason=='UNEXPECTED_OPERATIONAL'
 assert result.report is None
