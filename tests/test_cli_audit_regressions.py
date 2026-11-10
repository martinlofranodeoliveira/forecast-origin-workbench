"""Real installed CLI input failures and output-cap precedence regressions."""
import io,json,os,subprocess,sys
from unittest.mock import patch
import pytest
from forecast_origin_workbench import cli,BacktestResult,Status


def payload():
    return {'schema':cli.INPUT_SCHEMA,'values':[7.0]*128,'plan':{'lag_count':1,'horizon_count':1,'origins':[64],'seasonal_period':3}}


def invoke(args,**kwargs):
    return subprocess.run([sys.executable,'-I','-m','forecast_origin_workbench','backtest',*args],capture_output=True,timeout=20,**kwargs)


def expect_failure(result,status,reason,code):
    assert result.returncode==code and result.stdout==b''
    assert json.loads(result.stderr)=={'status':status,'reason':reason,'complete':False}


def test_directory_stdin_is_input_io(tmp_path):
    # CPython cannot initialize sys.stdin from a directory descriptor. Switch
    # to the real directory only after startup so the adapter owns the read.
    program="import os,sys; from forecast_origin_workbench import cli; fd=os.open(sys.argv[1],os.O_RDONLY|os.O_DIRECTORY); os.dup2(fd,0); os.close(fd); raise SystemExit(cli.main(['backtest','--input','-']))"
    result=subprocess.run([sys.executable,'-I','-c',program,str(tmp_path)],stdin=subprocess.DEVNULL,capture_output=True,cwd=tmp_path,timeout=20)
    expect_failure(result,'INCOMPLETE','INPUT_IO',4)


def test_closed_stdin_after_startup_is_input_io(tmp_path):
    program="import sys; from forecast_origin_workbench import cli; sys.stdin.buffer.close(); raise SystemExit(cli.main(['backtest','--input','-']))"
    result=subprocess.run([sys.executable,'-I','-c',program],stdin=subprocess.DEVNULL,capture_output=True,cwd=tmp_path,timeout=20)
    expect_failure(result,'INCOMPLETE','INPUT_IO',4)


def test_nonblocking_stdin_without_data_is_input_io(tmp_path):
    reader,writer=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        # A live writer with no data makes the real nonblocking read return
        # None, rather than EOF. No simulated producer is involved.
        result=invoke(['--input','-'],stdin=reader,cwd=tmp_path)
    finally:os.close(reader);os.close(writer)
    expect_failure(result,'INCOMPLETE','INPUT_IO',4)


@pytest.mark.parametrize('format',['json','csv'])
@pytest.mark.parametrize('case,status,reason,code',[
    ('json','INVALID_INPUT','INPUT_JSON',2),
    ('scalar','INVALID_INPUT','INPUT_SCHEMA',2),
    ('plan','INVALID_PLAN','PLAN_HISTORY',2),
    ('missing','INCOMPLETE','INPUT_IO',4),
    ('budget','UNKNOWN_BUDGET','ADMISSION_BUDGET',3),
])
def test_zero_output_cap_preserves_input_and_sdk_failures(case,status,reason,code,format,tmp_path):
    body=payload();args=['--input','-','--format',format,'--max-output-bytes','0']
    if case=='json':raw=b'{'
    else:
        if case=='scalar':body['values'][0]=True
        elif case=='plan':body['plan']['lag_count']=0
        elif case=='budget':body['budget']={'max_rows':0}
        elif case=='missing':args[1]=str(tmp_path/'absent-private-sentinel')
        raw=json.dumps(body).encode()
    result=invoke(args,input=raw,cwd=tmp_path)
    expect_failure(result,status,reason,code)
    assert b'absent-private-sentinel' not in result.stderr


@pytest.mark.parametrize('format',['json','csv'])
@pytest.mark.parametrize('status,reason,code',[
    (Status.FIT_FAILED,'FIT_VALUE_ERROR',4),
    (Status.NUMERICALLY_INDETERMINATE,'NUMERIC_RESULT',5),
    (Status.INCOMPLETE,'IO_FAILURE',4),
])
def test_zero_output_cap_preserves_failed_run_dto(status,reason,code,format,capfd):
    # Public DTO seam: this is not a claim that native fitting actually failed
    # numerically on the constant input.
    failed=BacktestResult(status,reason,None)
    stdin=io.TextIOWrapper(io.BytesIO(json.dumps(payload()).encode()))
    with patch.object(cli.sys,'stdin',stdin),patch.object(cli,'run_backtest',return_value=failed) as run:
        assert cli.main(['backtest','--input','-','--format',format,'--max-output-bytes','0'])==code
        run.assert_called_once()
    out,err=capfd.readouterr()
    assert out=='' and json.loads(err)=={'status':str(status),'reason':reason,'complete':False}
