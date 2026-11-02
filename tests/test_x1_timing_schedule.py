"""Exercise the checkpoint handoff with tiny subprocesses, not protein data."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'linux', reason='Scheduler uses Linux /proc and signals')
@pytest.mark.parametrize('timing_exit', [0, 1])
def test_checkpoint_then_timing_then_resume(tmp_path, timing_exit):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / 'scripts/schedule_x1_timing.py'
    shutil.copyfile(source, scripts / source.name)
    l1 = tmp_path / 'l1'
    l1.mkdir()
    (l1 / 'primary-functional_l1').mkdir()
    timing = tmp_path / 'timing'
    (scripts / 'run_x1_l1.py').write_text('''
from pathlib import Path
import sys,time
p=Path(sys.argv[-1])
checkpoint=p/'primary-functional_l1/run-000.json'
if checkpoint.exists():
    assert (p.parent/'timing/timing_attempted').exists()
    (p/'verified_results.json').write_text('{}')
else:
    while not (p.parent/'timing/schedule_status.json').exists():
        time.sleep(.05)
    checkpoint.write_text('{}')
    time.sleep(60)
''')
    (scripts / 'time_x1_vectors.py').write_text(f'''
from pathlib import Path
import sys
p=Path(sys.argv[-1])
assert (p.parent/'l1/primary-functional_l1/run-000.json').exists()
(p/'timing_attempted').write_text('yes')
sys.exit({timing_exit})
''')
    worker = subprocess.Popen([sys.executable, 'scripts/run_x1_l1.py', '--output-dir', str(l1)], cwd=tmp_path)
    try:
        result = subprocess.run([sys.executable, 'scripts/schedule_x1_timing.py', '--l1-pid', str(worker.pid),
                                 '--l1-dir', str(l1), '--timing-dir', str(timing)], cwd=tmp_path,
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert f'"exit_code": {timing_exit}' in result.stdout
        assert (l1 / 'verified_results.json').exists()
        assert (l1 / 'exit_code.txt').read_text().strip() == '0'
        assert worker.wait(timeout=5) == -15
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
