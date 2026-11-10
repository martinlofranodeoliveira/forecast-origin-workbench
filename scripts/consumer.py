"""Outside-wheel acceptance oracle. No candidate internal helper is imported."""
from dataclasses import replace
from decimal import Decimal,localcontext
from pathlib import Path
from unittest.mock import patch
import hashlib,importlib.util,json,math,resource,shutil,subprocess,tempfile
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import forecast_origin_workbench as f

ROOT=Path(__file__).parent
assert f.__file__.startswith('/opt/forecast-runtime/')
for name in ('build','pyproject_hooks','setuptools','wheel','pytest','pandas'):
 assert importlib.util.find_spec(name) is None,name
assert shutil.which('gcc') is None and shutil.which('cc') is None
expected=json.loads((ROOT/'module-hashes.json').read_text())
package=Path(f.__file__).parent
for name,digest in expected.items():assert hashlib.sha256((package/Path(name).name).read_bytes()).hexdigest()==digest

def execute(values,plan,inspect_fit=False):
 calls=[];original=HistGradientBoostingRegressor.fit
 schedule=[(o,h) for o in plan.origins for h in range(1,plan.horizon_count+1)]
 def checked(estimator,X,y,*args,**kwargs):
  o,h=schedule[len(calls)];L=plan.lag_count
  anchors=[a for a in range(len(values)) if a>=L-1 and a+h<=o]
  target=np.asarray([values[a+h] for a in anchors],dtype=float)
  design=np.asarray([[values[a-j] for j in range(L)] for a in anchors],dtype=float)
  np.testing.assert_array_equal(y,target);np.testing.assert_array_equal(X,design)
  params=estimator.get_params()
  assert params['max_iter']==64 and params['early_stopping'] is False and params['warm_start'] is False
  assert params['max_depth']==6 and params['max_leaf_nodes']==15
  calls.append({'origin':o,'horizon':h,'latest_target':max(a+h for a in anchors),'training_rows':len(anchors)})
  fitted=original(estimator,X,y,*args,**kwargs)
  assert estimator.n_iter_==64 and estimator.n_trees_per_iteration_==1
  return fitted
 if inspect_fit:
  with patch.object(HistGradientBoostingRegressor,'fit',new=checked):r=f.run_backtest(values,plan)
 else:r=f.run_backtest(values,plan)
 assert r.status==f.Status.COMPLETE,r
 encoded=f.serialize_json(r);assert encoded.status==f.Status.COMPLETE
 wire=json.loads(encoded.data)
 verify(wire,values)
 return r,wire,calls

def verify(wire,values):
 forecasts=wire['forecasts'];scores=wire['scores'];bykey={(x['model'],x['origin'],x['horizon']):x for x in forecasts}
 assert len(forecasts)==3*wire['work']['hgb_fits']
 for cell,score in zip(forecasts,scores,strict=True):
  assert score['actual']==values[cell['target']]
  assert score['error']==score['actual']-cell['prediction']
  m,o,h=cell['model'],cell['origin'],cell['horizon'];line=cell['lineage'];P=wire['plan']['seasonal_period'];L=wire['plan']['lag_count']
  if m!='hgb':
   support=o if m=='persistence' else max(i for i in range(o+1) if i%P==(o+h)%P)
   assert line['observed_support']==[support] and cell['prediction']==values[support]
   assert line['training_status']=='NOT_APPLICABLE'
   assert all(line[k] is None for k in ('anchor_start','anchor_end','training_rows','latest_training_target'))
  else:
   assert line['training_status']=='APPLICABLE' and line['anchor_end']+h<=o
   assert line['training_rows']==o-h-L+2 and line['latest_training_target']==o
  eligible=[r for r in wire['plan']['origins'] if r<o and r+h<=o]
  errors=[abs(values[r+h]-bykey[(m,r,h)]['prediction']) for r in eligible]
  b=cell['band'];assert b['eligible_origins']==eligible and b['count']==len(eligible)
  if len(errors)<3:
   assert b['status']=='UNAVAILABLE_INSUFFICIENT_HISTORY'
   assert all(b[k] is None for k in ('radius','lower','upper','rank'))
  else:
   k=min(k for k in range(1,len(errors)+1) if 5*k>=4*len(errors));radius=sorted(errors)[k-1]
   assert b['status']=='AVAILABLE' and b['rank']==k and b['radius']==radius
   assert b['lower']==cell['prediction']-radius and b['upper']==cell['prediction']+radius
 with localcontext() as ctx:
  ctx.prec=70
  for metric in wire['metrics']:
   chosen=[(cell,score) for cell,score in zip(forecasts,scores) if cell['model']==metric['model'] and (metric['horizon'] is None or cell['horizon']==metric['horizon'])]
   errors=[Decimal.from_float(score['actual'])-Decimal.from_float(cell['prediction']) for cell,score in chosen]
   assert len(errors)==metric['count']
   mae=sum(map(abs,errors))/len(errors);rmse=(sum(e*e for e in errors)/len(errors)).sqrt()
   for key,reference in [('mae',mae),('rmse',rmse)]:assert abs(Decimal.from_float(metric[key])-reference)<=Decimal('1e-12')*max(Decimal(1),abs(reference))

constant=[7.0]*128;plan=f.BacktestPlan(1,1,(64,),3)
r,wire,_=execute(constant,plan,True)
j=f.serialize_json(r);c=f.serialize_csv(r)
assert j.data==(ROOT/'oracles/constant-complete.json').read_bytes()
assert c.data==(ROOT/'oracles/constant-complete.csv').read_bytes()
for encode,n in [(f.serialize_json,2360),(f.serialize_csv,734)]:
 for cap in (0,n-1):
  e=encode(r,max_bytes=cap);assert e.status==f.Status.UNKNOWN_BUDGET and e.data is None and e.required_bytes==n
 assert encode(r,max_bytes=n).status==f.Status.COMPLETE
 assert encode(r,max_bytes=True).status==f.Status.INVALID_INPUT
for kw in [{'max_rows':0},{'max_hgb_fits':0},{'max_design_cells':0},{'max_json_bytes':0}]:
 assert f.run_backtest(constant,plan,budget=f.Budget(**kw)).status==f.Status.UNKNOWN_BUDGET

values=[0.0]*64+[10.,12.,11.,15.,14.,18.,17.,21.,20.,24.,23.,27.]+[27.]*52
plan=f.BacktestPlan(3,2,(67,69,71,73),3)
base,base_wire,calls=execute(values,plan,True);assert len(calls)==8
for cutoff in plan.origins:
 changed=[v if i<=cutoff else v+1000.0+i for i,v in enumerate(values)]
 changed_r,changed_wire,_=execute(changed,plan)
 assert [x for x in base_wire['forecasts'] if x['origin']<=cutoff]==[x for x in changed_wire['forecasts'] if x['origin']<=cutoff]
 assert base_wire['input']['sha256']!=changed_wire['input']['sha256']
 assert base_wire['scores']!=changed_wire['scores']
# Adjacent origins force a prior prediction whose target has not matured yet.
execute(values,f.BacktestPlan(3,2,(67,68,69,70),3),True)
# Native maximum logical shape; the spy still calls real maintained .fit.
t=np.arange(4096,dtype=float);large=(0.005*t+2*np.sin(2*np.pi*t/8)+0.3*np.cos(t/3)).tolist()
maximum,maximum_wire,max_calls=execute(large,f.BacktestPlan(32,8,(3583,3647,3711,3775,3839,3903,3967,4087),8),True)
assert len(max_calls)==64 and len(maximum_wire['forecasts'])==192
for name,digest in expected.items():assert hashlib.sha256((package/Path(name).name).read_bytes()).hexdigest()==digest
# Exercise the real console script from the rebuilt wheel with no checkout.
with tempfile.TemporaryDirectory() as temporary:
    demo=subprocess.run(['forecast-origin-workbench','demo','--assert'],cwd=temporary,capture_output=True,timeout=30)
    assert demo.returncode==0 and demo.stderr==b'' and json.loads(demo.stdout)['status']=='COMPLETE'
    payload={'schema':'forecast-origin-workbench-input/1','values':[7.0]*128,'plan':{'lag_count':1,'horizon_count':1,'origins':[64],'seasonal_period':3}}
    encoded_input=json.dumps(payload).encode()
    exact=subprocess.run(['forecast-origin-workbench','backtest','--input','-','--format','csv'],input=encoded_input,cwd=temporary,capture_output=True,timeout=30)
    assert exact.returncode==0 and exact.stdout==(ROOT/'oracles/constant-complete.csv').read_bytes()
    failed=subprocess.run(['forecast-origin-workbench','backtest','--input','-','--format','csv','--max-output-bytes','733'],input=encoded_input,cwd=temporary,capture_output=True,timeout=30)
    assert failed.returncode==3 and failed.stdout==b'' and json.loads(failed.stderr)['status']=='UNKNOWN_BUDGET'
print(json.dumps({'status':'PASS' ,'scope':'actual rebuilt wheel outside checkout; independent temporal/design/Decimal/bytes consumer, not quality benchmark',
 'module':f.__file__,'native_max_hgb_fits':64,'max_forecast_cells':192,'max_shape':{'rows':4096,'lags':32,'horizons':8,'origins':8},
 'all_origin_future_mutations':4,'exact_json_bytes':2360,'exact_csv_bytes':734,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'module_hashes_unchanged':True,'installed_cli_assert_demo_and_exact_csv':True}))
