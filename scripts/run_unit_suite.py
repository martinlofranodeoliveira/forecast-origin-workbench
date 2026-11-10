"""Run the actual generated pytest suite, retaining real counts and no empty pass."""
import json,os,sys,xml.etree.ElementTree as ET
from pathlib import Path
import pytest
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
xml=Path(sys.argv[2]).with_suffix('.xml')
code=pytest.main([sys.argv[1],'-q','--strict-config','--strict-markers','-p','no:cacheprovider','--junitxml='+str(xml)])
root=ET.parse(xml).getroot();suites=[root] if root.tag=='testsuite' else list(root.findall('testsuite'))
record={k:sum(int(x.attrib.get(k,'0')) for x in suites) for k in ('tests','failures','errors','skipped')}
record.update({'pytest_exit':int(code),'framework_version':pytest.__version__})
Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(0 if code==0 and record['tests']>0 and record['skipped']==0 else 1)
