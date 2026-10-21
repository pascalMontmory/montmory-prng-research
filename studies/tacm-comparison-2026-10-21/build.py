#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parent
audit=root.parent/'audit-2026-10-21'
for name in ['src','deps','build','results']:
    (root/name).mkdir(exist_ok=True)
if not (root/'deps/random123').exists():
    shutil.copytree(audit/'deps/random123',root/'deps/random123')
(root/'deps/montmory').mkdir(exist_ok=True)
for name in ['mmc_core.c','mmc_core.h']:
    if not (root/'deps/montmory'/name).exists():
        shutil.copy2(audit/'snapshot/montmory_gpu_test/montmory_gpu_test/src'/name,root/'deps/montmory'/name)
xoshiro=(root/'deps/xoshiro256starstar.c').read_text()
prefix=xoshiro.split('/* The following arbitrary-jump function',1)[0]
assert 'uint64_t next(void)' in prefix and 'void long_jump' in prefix and '#include "f2x.c"' not in prefix
(root/'build/xoshiro_reference_prefix.h').write_text(prefix)
common=['g++','-std=c++17','-I'+str(root/'deps/random123/include'),
        '-I'+str(root/'deps/montmory'),'-I'+str(root/'build'),str(root/'src/engines.cpp')]
commands=[common+['-O2','-fsanitize=undefined','-fno-sanitize-recover=undefined','-DTACM_CHECK_MAIN','-o',str(root/'build/check_engines')],
          common+['-O3','-march=native','-fPIC','-shared','-o',str(root/'build/libtacm.so')]]
for cmd in commands:
    subprocess.run(cmd,check=True)
check=subprocess.check_output([str(root/'build/check_engines'),str(root/'deps/random123/tests/kat_vectors')],text=True)
result=json.loads(check)
assert result['native_conformance_status']==0
result['compiler']=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
result['build_commands']=commands
result['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [root/'src/engines.cpp',root/'build/xoshiro_reference_prefix.h',root/'deps/xoshiro256starstar.c',root/'deps/montmory/mmc_core.c',root/'deps/montmory/mmc_core.h']}
(root/'results/conformance.json').write_text(json.dumps(result,indent=2)+'\n')
print(check,end='')
