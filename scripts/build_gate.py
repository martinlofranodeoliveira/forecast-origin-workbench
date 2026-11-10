"""Run only after the owner has read/admitted the exact six source files."""
import hashlib,json,os,subprocess,sys,tarfile,zipfile
from pathlib import Path

SOURCE=Path('/work/source');OUT=Path('/work/distributions');EVIDENCE=Path('/work/evidence')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def run(argv,**kwargs):subprocess.run(argv,check=True,**kwargs)
files={str(p.relative_to(SOURCE)):p for p in SOURCE.rglob('*') if p.is_file()}
assert files and 'pyproject.toml' in files
inventory={name:{'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size} for name,p in sorted(files.items())}
for item in json.loads((SOURCE/'qualified-wheel-inventory.json').read_text()):
 assert digest((Path('/opt/wheelhouse')/item['file']).read_bytes())==item['sha256']
run([sys.executable,'-m','venv','/tmp/forecast-sdk'])
python='/tmp/forecast-sdk/bin/python'
run([python,'-m','pip','install','--no-cache-dir','--no-index','--find-links','/opt/wheelhouse','--require-hashes','-r','/opt/requirements-dev.lock'])
run([python,'-m','build','--no-isolation','--wheel','--sdist','--outdir',str(OUT)],cwd=SOURCE)
wheel,=OUT.glob('*.whl');sdist,=OUT.glob('*.tar.gz')
package_files={name.removeprefix('src/'):item['sha256'] for name,item in inventory.items() if name.startswith('src/')}
module_hashes={name:sha for name,sha in package_files.items() if name.count('/')==1 and name.endswith('.py')}
def check_wheel(path):
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  assert len(names)==len(set(names))
  for member,expected in package_files.items():assert digest(z.read(member))==expected
  unexpected=[n for n in names if n not in package_files and '.dist-info/' not in n]
  assert not unexpected,unexpected
  assert all(not n.startswith('/') and '..' not in Path(n).parts and '\\' not in n for n in names)
check_wheel(wheel)
with tarfile.open(sdist) as t:
 members=t.getmembers();tops={m.name.split('/')[0] for m in members};assert len(tops)==1
 top=next(iter(tops));assert top and top not in ('.','..')
 for m in members:
  assert not m.name.startswith('/') and '..' not in Path(m.name).parts and '\\' not in m.name
  assert m.isfile() or m.isdir(),m.name
 for name,item in inventory.items():assert digest(t.extractfile(top+'/'+name).read())==item['sha256'],name
 unpack=Path('/tmp/forecast-sdist');unpack.mkdir()
 t.extractall(unpack,filter='data')
rebuilt=OUT/'rebuilt';rebuilt.mkdir()
run([python,'-m','build','--no-isolation','--wheel','--outdir',str(rebuilt)],cwd=unpack/top)
rebuilt_wheel,=rebuilt.glob('*.whl');check_wheel(rebuilt_wheel)
run([python,'-m','pip','install','--no-index','--no-deps',str(wheel)])
run([python,'-m','pip','check'])
run([python,'-c',"import forecast_origin_workbench as f; assert f.__file__.startswith('/tmp/forecast-sdk/'); print(f.__file__)"],cwd='/tmp')
# The owner reads tests first; collect a real count, reject skipped/empty suites.
runner=SOURCE/'scripts/run_unit_suite.py'
run([python,str(runner),str(SOURCE/'tests'),str(EVIDENCE/'unit-tests.json')],cwd=SOURCE)
for name, item in inventory.items():
 assert digest((SOURCE/name).read_bytes()) == item['sha256'], name
proof={'status':'PASS','scope':'exact full project, installed SDK/CLI tests and source/sdist/rebuilt-wheel equality; hosted CI remains separate',
 'source_inventory':inventory,'module_hashes':module_hashes,'source_sdist_all_admitted_files_equal':True,
 'source_original_and_rebuilt_wheel_module_hashes_equal':True,
 'artifacts':{str(p.relative_to(OUT)):digest(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}}
(EVIDENCE/'package-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'status':'PASS','gate':'SDK_TESTS_REAL_SDIST_WHEEL_AND_REBUILD'}))
