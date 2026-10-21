#!/usr/bin/env python3
"""Replay in a fresh directory; never replace the delivered evidence."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
results=root/'results'
if results.exists() and any(results.iterdir()):
    raise SystemExit('Preserve existing results first: use a new copy and rename results to reference-results.')
def run(script,*args):
    return subprocess.run([sys.executable,str(root/script),*args],cwd=root,check=True)

run('build.py')
pilot=subprocess.run([sys.executable,str(root/'run_campaign.py'),'--phase','calibration'],cwd=root)
if pilot.returncode:
    record=json.loads((results/'execution.json').read_text())
    if record['completed'] or not (results/'calibration.csv').exists():
        raise SystemExit('Unexpected pilot failure; inspect logs.')
    run('calibrate_extension.py')
else:
    raise SystemExit('Pilot differs from archived failed gate. Preserve new evidence and inspect before proceeding.')
for phase in ['compare','lanes','timing']:
    run('run_campaign.py','--phase',phase)
run('analyze.py')
run('verify_replay.py')
