#!/usr/bin/env python3
"""Control-flow checks using fake Apple/build tools; no native Metal validation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / 'mac-metal-testu01-repair-v1'
FAKE = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
base = Path(os.environ['CHECK_BASE'])
case = os.environ['CHECK_CASE']
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
def executable(path, body):
    path.write_text('#!/usr/bin/env python3\n' + body)
    path.chmod(0o755)
if tool == 'uname':
    print('Darwin')
elif tool == 'sw_vers':
    print('Mock macOS')
elif tool == 'xcrun':
    assert args[:2] == ['--sdk', 'macosx']
    args = args[2:]
    if args == ['--show-sdk-path']:
        assert 'SDKROOT' not in os.environ
        print(base / 'sdk')
    elif args[0] == '--find':
        print(base / 'fake-tools' / 'fake-cc')
    elif args[0] == 'clang++':
        os.execv(str(base / 'fake-tools/fake-cc'), ['fake-cc'] + args[1:])
    else:
        raise AssertionError(args)
elif tool == 'fake-cc':
    assert os.environ['SDKROOT'] == str(base / 'sdk')
    assert 'LDFLAGS' not in os.environ and 'CPATH' not in os.environ
    if args == ['--version']:
        print('Mock compiler')
        sys.exit(0)
    target = Path(args[args.index('-o') + 1])
    if target.name == 'compiler-check':
        if case == 'probe_fail':
            print('MOCK compiler link error')
            sys.exit(1)
        executable(target, "print('Mock ARM64 probe PASS')\n")
    elif target.name == 'testu01_metal':
        if case == 'wrapper_fail':
            print('MOCK wrapper compile error')
            sys.exit(2)
        executable(target, """import os, sys
from pathlib import Path
assert sys.argv[1] == 'montmory_mulhi_pair'
with (Path(os.environ['CHECK_BASE']) / 'batteries').open('a') as f:
    f.write(sys.argv[2] + '\\n')
if os.environ['CHECK_CASE'] == 'battery_fail' and sys.argv[2] == 'crush':
    print('Suspicious test result')
else:
    print('All tests were passed')
""")
    else:
        raise AssertionError(target)
elif tool == 'unzip':
    assert args[0] == '-q' and args[2] == '-d'
    folder = Path(args[3]) / 'TestU01-1.2.3'
    folder.mkdir()
    executable(folder / 'configure', """import os, sys
from pathlib import Path
assert os.environ['SDKROOT'] == str(Path(os.environ['CHECK_BASE']) / 'sdk')
Path('config.log').write_text('checking for C compiler default output file name\\nMOCK linker diagnostic in config.log\\n')
if os.environ['CHECK_CASE'] == 'configure_fail':
    print('MOCK configure error')
    sys.exit(77)
print('Mock configuration succeeded')
""")
elif tool == 'make':
    print('Mock make', args)
else:
    raise AssertionError(tool)
'''

def checksums(folder, names):
    (folder / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256((folder / name).read_bytes()).hexdigest() + '  ' + name + '\n'
        for name in names))

def one(case, expected):
    with tempfile.TemporaryDirectory(prefix='testu01-repair-check-') as temp:
        base = Path(temp)
        kit = base / 'kit'
        kit.mkdir()
        names = ['src/testu01_metal.mm', 'vendor/TestU01.zip', 'generated/provenance.json']
        for name in names:
            p = kit / name
            p.parent.mkdir(exist_ok=True)
            p.write_text('{}\n')
        checksums(kit, names)
        repair = kit / 'testu01-repair-v1'
        repair.mkdir()
        shutil.copy2(REPAIR / 'run.sh', repair / 'run.sh')
        checksums(repair, ['run.sh'])
        fake = base / 'fake-tools'
        fake.mkdir()
        (base / 'sdk').mkdir()
        for name in ['uname', 'sw_vers', 'xcrun', 'fake-cc', 'unzip', 'make']:
            p = fake / name
            p.write_text(FAKE)
            p.chmod(0o755)
        env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                   CHECK_BASE=str(base), CHECK_CASE=case, SDKROOT='/wrong-sdk',
                   LDFLAGS='invalid inherited flags', CPATH='/wrong-headers')
        result = subprocess.run(['bash', str(repair / 'run.sh'), 'montmory_mulhi_pair', 'all'],
                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == expected, (case, result.returncode, result.stdout)
        directories = list((kit / 'results').iterdir())
        assert len(directories) == 1
        logs = directories[0]
        assert int((logs / 'exit-code.txt').read_text()) == expected
        archives = list(kit.glob('montmory-metal-v3-testu01-repair1-*.tar.gz'))
        assert len(archives) == 1
        with tarfile.open(archives[0]) as tf:
            assert any(m.name.endswith('/exit-code.txt') for m in tf.getmembers())
        batteries = (base / 'batteries').read_text().splitlines() if (base / 'batteries').exists() else []
        expected_batteries = ['small', 'crush', 'big'] if case == 'success' else (
            ['small', 'crush'] if case == 'battery_fail' else [])
        assert batteries == expected_batteries, (case, batteries)
        assert ('PASS : toutes les batteries' in result.stdout) == (case == 'success')
        if case == 'configure_fail':
            assert 'MOCK linker diagnostic in config.log' in result.stdout
            assert (logs / 'config.log').is_file()
        return {'case': case, 'exit_code': expected, 'battery_order': batteries,
                'archive_created': True, 'last_stage': (logs / 'last-stage.txt').read_text().strip()}

if __name__ == '__main__':
    subprocess.run(['bash', '-n', str(REPAIR / 'run.sh')], check=True)
    report = {'status': 'pass',
              'scope': 'Shell orchestration with fake Apple/build tools on Linux; not native compilation or TestU01 execution',
              'source_sha256': hashlib.sha256((REPAIR / 'run.sh').read_bytes()).hexdigest(),
              'cases': [one(case, code) for case, code in [
                  ('success', 0), ('probe_fail', 1), ('configure_fail', 77),
                  ('wrapper_fail', 2), ('battery_fail', 3)]]}
    (ROOT / 'evidence/testu01-repair-v1-shell-checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
