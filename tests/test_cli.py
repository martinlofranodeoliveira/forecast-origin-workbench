"""Installed isolated CLI: framing, bytes, local files and error boundaries."""
import io,json,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import pytest
from forecast_origin_workbench import cli


def command(arguments,*,cwd,input=None):
    return subprocess.run([sys.executable,'-I','-m','forecast_origin_workbench',*arguments],input=input,capture_output=True,cwd=cwd,timeout=20)


def body():
    return {'schema':cli.INPUT_SCHEMA,'values':[7.0]*128,'plan':{'lag_count':1,'horizon_count':1,'origins':[64],'seasonal_period':3}}


@pytest.mark.parametrize('format',['json','csv'])
def test_real_installed_demo(format,tmp_path):
    result=command(['demo','--assert','--format',format],cwd=tmp_path)
    assert result.returncode==0 and result.stderr==b'' and result.stdout.endswith(b'\n')
    if format=='json':
        report=json.loads(result.stdout)
        assert report['status']=='COMPLETE' and report['work']['hgb_fits']==8
    else:assert result.stdout.startswith(b'model,origin,horizon,')


@pytest.mark.parametrize('format,size',[('json',2360),('csv',734)])
def test_literal_input_and_exact_output_limits(format,size,tmp_path):
    path=tmp_path/'series.json';path.write_text(json.dumps(body()))
    exact=command(['backtest','--input',str(path),'--format',format,'--max-output-bytes',str(size)],cwd=tmp_path)
    assert exact.returncode==0 and len(exact.stdout)==size and exact.stderr==b''
    for cap in (0,size-1):
        failed=command(['backtest','--input',str(path),'--format',format,'--max-output-bytes',str(cap)],cwd=tmp_path)
        assert failed.returncode==3 and failed.stdout==b''
        error=json.loads(failed.stderr);assert error['status']=='UNKNOWN_BUDGET' and error['complete'] is False


def test_stdin_and_json_admission_before_csv(tmp_path):
    payload=body()
    accepted=command(['backtest','--input','-'],input=json.dumps(payload).encode(),cwd=tmp_path)
    assert accepted.returncode==0 and len(accepted.stdout)==2360
    payload['budget']={'max_json_bytes':734}
    failed=command(['backtest','--input','-','--format','csv'],input=json.dumps(payload).encode(),cwd=tmp_path)
    assert failed.returncode==3 and failed.stdout==b'' and json.loads(failed.stderr)['reason']=='JSON_BUDGET'


@pytest.mark.parametrize('raw',[b'{',b'{"schema":"x","schema":"x"}',b'{"value":1e999}',b'{"value":NaN}',b'\xff',b'{"value":Infinity}'])
def test_invalid_json_never_emits_success(raw,tmp_path):
    result=command(['backtest','--input','-'],input=raw,cwd=tmp_path)
    assert result.returncode==2 and result.stdout==b''
    assert json.loads(result.stderr)=={'status':'INVALID_INPUT','reason':'INPUT_JSON','complete':False}


@pytest.mark.parametrize('kind',['unknown','extra_plan','long_origins','extra_budget','bool_budget'])
def test_closed_schema_and_typed_sdk_failures(kind,tmp_path):
    payload=body()
    if kind=='unknown':payload['private_secret_sentinel']=123
    elif kind=='extra_plan':payload['plan']['expression']='do_not_execute'
    elif kind=='long_origins':payload['plan']['origins']=list(range(64,73))
    elif kind=='extra_budget':payload['budget']={'extra':5}
    else:payload['budget']={'max_rows':True}
    result=command(['backtest','--input','-'],input=json.dumps(payload).encode(),cwd=tmp_path)
    assert result.returncode==2 and result.stdout==b''
    assert b'private_secret_sentinel' not in result.stderr and b'do_not_execute' not in result.stderr


@pytest.mark.parametrize('kind',['symlink','fifo','directory','missing'])
def test_file_boundaries_do_not_block_or_echo_paths(kind,tmp_path):
    path=tmp_path/'private_path_sentinel'
    if kind=='symlink':
        target=tmp_path/'real.json';target.write_text(json.dumps(body()));path.symlink_to(target)
    elif kind=='fifo':os.mkfifo(path)
    elif kind=='directory':path.mkdir()
    result=command(['backtest','--input',str(path)],cwd=tmp_path)
    assert result.returncode in (2,4) and result.stdout==b'' and b'private_path_sentinel' not in result.stderr
    assert json.loads(result.stderr)['complete'] is False


def test_oversized_regular_input_and_zero_sdk_json_no_fit(tmp_path):
    path=tmp_path/'large';path.write_bytes(b' '*(cli.MAX_INPUT_BYTES+1))
    result=command(['backtest','--input',str(path)],cwd=tmp_path)
    assert result.returncode==3 and result.stdout==b'' and json.loads(result.stderr)['reason']=='INPUT_BYTES'
    # The SDK JSON admission cap is distinct from the CLI encoding cap.
    from forecast_origin_workbench import run_backtest,BacktestPlan,Budget,Status
    with patch('sklearn.ensemble.HistGradientBoostingRegressor.fit',side_effect=AssertionError('must not fit')) as fit:
        failed=run_backtest([7.0]*128,BacktestPlan(1,1,(64,),3),budget=Budget(max_json_bytes=0))
        assert failed.status==Status.UNKNOWN_BUDGET and failed.reason=='JSON_BUDGET' and failed.report is None
        fit.assert_not_called()


def test_short_and_failed_stream_writes():
    class Short:
        def __init__(self):self.data=bytearray()
        def write(self,data):self.data.extend(data[:2]);return min(2,len(data))
        def flush(self):pass
    stream=Short();cli._write(stream,b'abcdef\n');assert stream.data==b'abcdef\n'
    class Broken:
        def write(self,data):return 0
        def flush(self):pass
    with pytest.raises(OSError):cli._write(Broken(),b'ab')


def test_demo_assertions_work_with_python_optimization(tmp_path):
    result=subprocess.run([sys.executable,'-I','-O','-m','forecast_origin_workbench','demo','--assert'],capture_output=True,cwd=tmp_path,timeout=20)
    assert result.returncode==0 and json.loads(result.stdout)['status']=='COMPLETE'


@pytest.mark.parametrize('args',[[],['backtest'],['demo','--format','private_usage_sentinel'],['demo','--max-output-bytes','01'],['demo','--max-output-bytes','1048577']])
def test_cli_usage_errors_are_stable(args,tmp_path):
    result=command(args,cwd=tmp_path)
    assert result.returncode==2 and result.stdout==b''
    assert json.loads(result.stderr)=={'status':'INVALID_INPUT','reason':'CLI_USAGE','complete':False}
    assert b'private_usage_sentinel' not in result.stderr


@pytest.mark.parametrize('case,expected',[
    ('small_json_stdout_full',4),('invalid_json_stderr_full',2),
    ('both_full',4),('help_stdout_full',4),
])
def test_real_failed_standard_descriptors_preserve_exit_codes(case,expected,tmp_path):
    payload=json.dumps(body()).encode()
    args=['backtest','--input','-']
    if case=='invalid_json_stderr_full':payload=b'{'
    if case=='help_stdout_full':args=['--help'];payload=None
    with open('/dev/full','wb') as full:
        stdout=full if case in ('small_json_stdout_full','both_full','help_stdout_full') else subprocess.PIPE
        stderr=full if case in ('invalid_json_stderr_full','both_full') else subprocess.PIPE
        result=subprocess.run([sys.executable,'-I','-m','forecast_origin_workbench',*args],input=payload,stdout=stdout,stderr=stderr,cwd=tmp_path,timeout=20)
    assert result.returncode==expected
    if result.stderr:
        error=json.loads(result.stderr)
        assert error=={'status':'INCOMPLETE','reason':'OUTPUT_IO','complete':False}
        assert b'Exception ignored' not in result.stderr
