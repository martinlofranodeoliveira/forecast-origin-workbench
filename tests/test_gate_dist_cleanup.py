"""Invalidate generated dist before Docker stages; do not fake Docker."""
import os,shutil,subprocess
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/check.sh'


def setup(tmp_path):
    project=tmp_path/'owned-project';(project/'scripts').mkdir(parents=True)
    shutil.copyfile(SCRIPT,project/'scripts/check.sh')
    env=os.environ.copy()
    # Where Docker exists it fails connecting to a private nonexistent socket.
    # SDK images have no Docker client, and instead fail first-stage lookup.
    # Neither mode is a mocked Docker success or a native consumer claim.
    env['DOCKER_HOST']='unix://'+str(tmp_path/'no-docker.sock')
    return project,env


def test_old_dist_cannot_survive_first_stage_failure(tmp_path):
    project,env=setup(tmp_path);(project/'dist/evidence').mkdir(parents=True)
    for name in ('evidence/package-proof.json','evidence/unit-tests.json','runtime-container.json','sdk-container.json','old.whl','old.tar.gz'):
        (project/'dist'/name).write_text('OLD_PASS_SENTINEL')
    outside=tmp_path/'outside';outside.write_text('KEEP')
    result=subprocess.run(['sh','scripts/check.sh'],cwd=project,env=env,capture_output=True,timeout=20)
    assert result.returncode!=0 and not (project/'dist').exists()
    assert outside.read_text()=='KEEP'


def test_symlink_dist_is_rejected_without_following_or_removing_target(tmp_path):
    project,env=setup(tmp_path)
    outside=tmp_path/'outside';outside.mkdir();(outside/'keep').write_text('KEEP')
    (project/'dist').symlink_to(outside,target_is_directory=True)
    result=subprocess.run(['sh','scripts/check.sh'],cwd=project,env=env,capture_output=True,timeout=20)
    assert result.returncode!=0 and (project/'dist').is_symlink()
    assert (outside/'keep').read_text()=='KEEP'
    assert b'dist must be a real generated directory' in result.stderr


def test_non_directory_dist_is_rejected_without_deletion(tmp_path):
    project,env=setup(tmp_path);(project/'dist').write_text('KEEP')
    result=subprocess.run(['sh','scripts/check.sh'],cwd=project,env=env,capture_output=True,timeout=20)
    assert result.returncode!=0 and (project/'dist').read_text()=='KEEP'
    assert b'dist must be a real generated directory' in result.stderr
