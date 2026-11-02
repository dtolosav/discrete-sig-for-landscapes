"""Wait for an L1 checkpoint, run serial vector timings, then resume L1."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def command(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0', b' ').decode() if path.exists() else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l1-pid', required=True, type=int)
    parser.add_argument('--l1-dir', required=True, type=Path)
    parser.add_argument('--timing-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.timing_dir
    out.mkdir(parents=True, exist_ok=True)
    initial_command = command(args.l1_pid)
    assert 'scripts/run_x1_l1.py' in initial_command and str(args.l1_dir) in initial_command
    initial_start = Path(f'/proc/{args.l1_pid}/stat').read_text().split()[21]
    baseline = len(list((args.l1_dir / 'primary-functional_l1').glob('run-*.json')))

    def record(state, **details):
        value = dict(state=state, utc=datetime.now(timezone.utc).isoformat(), **details)
        (out / 'schedule_status.json').write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    def active():
        stat = Path(f'/proc/{args.l1_pid}/stat')
        return (command(args.l1_pid) == initial_command and stat.exists()
                and stat.read_text().split()[21] == initial_start)

    record('waiting_for_l1_checkpoint', completed_runs=baseline, l1_pid=args.l1_pid)
    while active():
        files = list((args.l1_dir / 'primary-functional_l1').glob('run-*.json'))
        if len(files) > baseline:
            break
        time.sleep(1)
    stopped = False
    try:
        if active():
            os.kill(args.l1_pid, signal.SIGTERM)
            stopped = True
            while active():
                time.sleep(.2)
        elif not (args.l1_dir / 'verified_results.json').exists():
            raise RuntimeError('L1 exited before a new checkpoint; investigate before timing.')
        record('running_vector_timings', completed_l1_runs=len(list((args.l1_dir / 'primary-functional_l1').glob('run-*.json'))),
               note='Unfinished L1 run will be recomputed; no paused interval enters a run timing.')
        with (out / 'computation.log').open('a') as log:
            code = subprocess.call([sys.executable, '-u', 'scripts/time_x1_vectors.py', '--output-dir', str(out)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record('vector_timings_finished', exit_code=code)
    finally:
        if stopped and not (args.l1_dir / 'verified_results.json').exists():
            with (args.l1_dir / 'computation.log').open('a') as log, (args.l1_dir / 'errors.log').open('a') as errors:
                worker = subprocess.Popen([sys.executable, '-u', 'scripts/run_x1_l1.py', '--output-dir', str(args.l1_dir)],
                                          cwd=ROOT, stdout=log, stderr=errors)
                record('l1_resumed', l1_pid=worker.pid)
                (out / 'resumed_l1_pid.txt').write_text(str(worker.pid) + '\n')
                result = worker.wait()
            (args.l1_dir / 'exit_code.txt').write_text(str(result) + '\n')
            record('resumed_l1_finished', exit_code=result)


if __name__ == '__main__':
    main()
