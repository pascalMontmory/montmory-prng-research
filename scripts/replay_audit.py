#!/usr/bin/env python3
"""Run frozen analysis in a disposable copy; never rewrite committed evidence."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {'tpu': 'tpu-opt-2026-10-21', 'tpu-initial': 'jax-tpu-2026-10-21',
           'metal': 'metal-opt-2026-10-21', 'tacm': 'tacm-comparison-2026-10-21'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', choices=STUDIES)
    parser.add_argument('--out', type=Path, help='New output directory; must not exist')
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.out or ROOT / 'build' / f'{stamp}-{args.audit}-reanalysis').resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               JAX_PLATFORMS='cpu', MPLCONFIGDIR=str(out / 'matplotlib-cache'))
    # Legacy audits use `python3` for a child process: select the current venv too.
    env['PATH'] = str(Path(sys.executable).absolute().parent) + os.pathsep + env.get('PATH', '')
    with tempfile.TemporaryDirectory(prefix='montmory-reanalysis-') as tmp:
        name = STUDIES[args.audit]
        work = Path(tmp) / name
        shutil.copytree(ROOT / 'studies' / name, work)
        if args.audit == 'tpu':
            commands = [['audit_results.py', '20261021T185102Z-tpu-opt']]
            result_dir = work / 'results/audit-20261021T185102Z-tpu-opt'
        elif args.audit == 'tpu-initial':
            result_dir = work / 'results/reanalysis'
            commands = [['audit_colab.py', '20261021T180635Z-colab-tpu', '--out', str(result_dir)]]
        elif args.audit == 'metal':
            result_dir = work / 'results/mac-analysis-20261021T162407Z-22542'
            commands = [['analyze_mac.py']]
            archive = ROOT / 'evidence/archives/montmory-metal-opt-results-20261021T162407Z-22542.tar.gz'
            shutil.copy2(archive, work.parent / archive.name)
        else:
            result_dir = work / 'results'
            commands = [['build.py'], ['analyze.py'], ['verify_replay.py']]
        with (out / 'reanalysis.log').open('w') as log:
            for command in commands:
                print('Reanalysis:', ' '.join(command), flush=True)
                process = subprocess.Popen([sys.executable, '-u', *command], cwd=work, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                if process.wait():
                    raise RuntimeError(f'{command[0]} failed; see {out / "reanalysis.log"}')
        shutil.copytree(result_dir, out / 'analysis', ignore=shutil.ignore_patterns('*.so', '__pycache__'))
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_repository.py'),
                    '--out', str(out / 'post-replay-integrity.json')], check=True, env=env)
    result = {'status': 'pass', 'audit': args.audit, 'hardware_rebenchmarked': False,
              'frozen_imports_unchanged': True, 'python': sys.version.split()[0]}
    (out / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Reanalysis saved:', out)


if __name__ == '__main__':
    main()
