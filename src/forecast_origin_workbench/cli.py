"""Bounded local JSON/CSV adapters for the fixed forecasting SDK."""
from __future__ import annotations
import argparse,errno,json,math,os,re,stat,sys
from . import BacktestPlan,Budget,Status,run_backtest,serialize_json,serialize_csv

MAX_INPUT_BYTES=524288
MAX_OUTPUT_BYTES=1048576
INPUT_SCHEMA='forecast-origin-workbench-input/1'
class InputFailure(Exception):
    def __init__(self,status,reason):self.status=status;self.reason=reason
class UsageFailure(Exception):pass
class ParserExit(Exception):
    def __init__(self,status):self.status=status
class Parser(argparse.ArgumentParser):
    def error(self,message):raise UsageFailure()
    def _print_message(self,message,file=None):
        if message:
            stream=file if file is not None else sys.stderr
            try:
                stream.write(message)
                stream.flush()
            except (OSError,ValueError):
                _quiet_failed_stream(stream)
                raise
    def exit(self,status=0,message=None):
        if message:self._print_message(message,sys.stderr)
        raise ParserExit(status)

def _read_input(name):
    if name=='-':
        try:raw=sys.stdin.buffer.read(MAX_INPUT_BYTES+1)
        except (OSError,ValueError):raise InputFailure('INCOMPLETE','INPUT_IO') from None
        if raw is None:raise InputFailure('INCOMPLETE','INPUT_IO')
        if len(raw)>MAX_INPUT_BYTES:raise InputFailure('UNKNOWN_BUDGET','INPUT_BYTES')
        return raw
    if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',name):raise InputFailure('INVALID_INPUT','REMOTE_INPUT')
    try:fd=os.open(name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC)
    except OSError as error:
        if error.errno==errno.ELOOP:raise InputFailure('INVALID_INPUT','SYMLINK_INPUT') from None
        raise InputFailure('INCOMPLETE','INPUT_IO') from None
    except ValueError:raise InputFailure('INCOMPLETE','INPUT_IO') from None
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):raise InputFailure('INVALID_INPUT','NON_REGULAR_INPUT')
        if before.st_size>MAX_INPUT_BYTES:raise InputFailure('UNKNOWN_BUDGET','INPUT_BYTES')
        raw=bytearray()
        while len(raw)<=MAX_INPUT_BYTES:
            part=os.read(fd,min(65536,MAX_INPUT_BYTES+1-len(raw)))
            if not part:break
            raw.extend(part)
        if len(raw)>MAX_INPUT_BYTES:raise InputFailure('UNKNOWN_BUDGET','INPUT_BYTES')
        after=os.fstat(fd)
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise InputFailure('INCOMPLETE','INPUT_CHANGED')
        return bytes(raw)
    except OSError:raise InputFailure('INCOMPLETE','INPUT_IO') from None
    finally:os.close(fd)

def _decode(raw):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('duplicate')
            out[key]=value
        return out
    def number(token):
        value=float(token)
        if not math.isfinite(value):raise ValueError('nonfinite')
        return value
    def constant(token):raise ValueError('nonfinite')
    try:data=json.loads(raw.decode('utf-8'),object_pairs_hook=unique,parse_float=number,parse_constant=constant)
    except (UnicodeError,ValueError,RecursionError):raise InputFailure('INVALID_INPUT','INPUT_JSON') from None
    if type(data) is not dict or not {'schema','values','plan'}<=data.keys() or data.keys()-{'schema','values','plan','start_tick','step','unit','budget'}:
        raise InputFailure('INVALID_INPUT','INPUT_SCHEMA')
    if data['schema']!=INPUT_SCHEMA or type(data['values']) is not list:raise InputFailure('INVALID_INPUT','INPUT_SCHEMA')
    plan=data['plan']
    if type(plan) is not dict or set(plan)!={'lag_count','horizon_count','origins','seasonal_period'} or type(plan['origins']) is not list:
        raise InputFailure('INVALID_PLAN','PLAN_HISTORY')
    if not 1<=len(plan['origins'])<=8:raise InputFailure('INVALID_PLAN','PLAN_HISTORY')
    budget=data.get('budget',{})
    if type(budget) is not dict or budget.keys()-{'max_rows','max_hgb_fits','max_design_cells','max_json_bytes'}:raise InputFailure('INVALID_INPUT','INPUT_SCHEMA')
    return data['values'],BacktestPlan(**plan),Budget(**budget),{k:data[k] for k in ('start_tick','step','unit') if k in data}

def _demo():
    values=[0.0]*64+[10.,12.,11.,15.,14.,18.,17.,21.,20.,24.,23.,27.]+[27.]*52
    return values,BacktestPlan(3,2,(67,69,71,73),3),Budget(),{}

def _cap(value):
    if not re.fullmatch(r'0|[1-9][0-9]{0,6}',value):raise UsageFailure()
    result=int(value)
    if result>MAX_OUTPUT_BYTES:raise UsageFailure()
    return result

def _quiet_failed_stream(stream):
    # Only an already-failed standard descriptor is silenced. Preserve the other.
    try:
        failed=stream.fileno()
        if type(failed) is not int or failed not in (1,2):return
        descriptor=os.open(os.devnull,os.O_WRONLY|os.O_CLOEXEC)
        try:os.dup2(descriptor,failed)
        finally:
            if descriptor!=failed:os.close(descriptor)
    except (AttributeError,OSError,ValueError):pass


def _write(stream,payload):
    try:
        view=memoryview(payload)
        while view:
            count=stream.write(view)
            if type(count) is not int or not 0<count<=len(view):raise OSError('short write')
            view=view[count:]
        stream.flush()
    except (OSError,ValueError):
        _quiet_failed_stream(stream)
        raise

def _error(status,reason):
    payload=(json.dumps({'status':str(status),'reason':str(reason),'complete':False},sort_keys=True,separators=(',',':'))+'\n').encode()
    try:_write(sys.stderr.buffer,payload)
    except (OSError,ValueError):pass
    return {'INVALID_INPUT':2,'INVALID_PLAN':2,'UNKNOWN_BUDGET':3,'NUMERICALLY_INDETERMINATE':5}.get(str(status),4)

def main(argv=None):
    parser=Parser(prog='forecast-origin-workbench',description='Bounded local expanding-origin backtests; empirical bands do not promise coverage.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('demo','backtest'):
        child=commands.add_parser(name)
        child.add_argument('--format',choices=('json','csv'),default='json')
        child.add_argument('--max-output-bytes',default='1048576')
        if name=='backtest':child.add_argument('--input',required=True)
        else:child.add_argument('--assert',dest='assert_demo',action='store_true')
    try:
        args=parser.parse_args(argv);cap=_cap(args.max_output_bytes)
        values,plan,budget,metadata=_demo() if args.command=='demo' else _decode(_read_input(args.input))
        result=run_backtest(values,plan,budget=budget,**metadata)
        if result.status!=Status.COMPLETE:return _error(result.status,result.reason)
        if args.command=='demo' and args.assert_demo:
            cells={(x.origin,x.horizon,x.model):x for x in result.report.forecasts}
            if not (cells[(69,2,'persistence')].prediction==18.0 and cells[(69,2,'seasonal')].lineage.observed_support==(68,) and cells[(73,2,'persistence')].band.radius==3.0):
                return _error('INCOMPLETE','DEMO_ASSERTION')
        encoded=(serialize_json if args.format=='json' else serialize_csv)(result,max_bytes=cap)
        if encoded.status!=Status.COMPLETE:return _error(encoded.status,encoded.reason)
        _write(sys.stdout.buffer,encoded.data)
        return 0
    except ParserExit as stopped:return stopped.status
    except UsageFailure:return _error('INVALID_INPUT','CLI_USAGE')
    except InputFailure as error:return _error(error.status,error.reason)
    except MemoryError:return _error('INCOMPLETE','MEMORY_FAILURE')
    except ImportError:return _error('INCOMPLETE','DEPENDENCY_UNAVAILABLE')
    except TimeoutError:return _error('INCOMPLETE','TIMEOUT')
    except (OSError,ValueError):return _error('INCOMPLETE','OUTPUT_IO')
    except Exception:return _error('INCOMPLETE','UNEXPECTED_OPERATIONAL')

if __name__=='__main__':raise SystemExit(main())
