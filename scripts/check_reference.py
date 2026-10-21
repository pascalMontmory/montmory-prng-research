#!/usr/bin/env python3
"""Rebuild the unchanged C reference and replay finite/exact certificates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'build/reference-check.json')
    args = p.parse_args()
    ref = ROOT / 'src/reference'
    cc = shlex.split(os.environ.get('CC', 'cc'))
    with tempfile.TemporaryDirectory(prefix='montmory-reference-') as tmp:
        exe = Path(tmp) / 'vectors'
        command = [*cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                   '-fsanitize=undefined', '-fno-sanitize-recover=undefined',
                   '-I' + str(ref), str(ref / 'conformance_dump.c'), '-o', str(exe)]
        subprocess.run(command, check=True)
        actual = subprocess.check_output([str(exe)])
    expected = (ref / 'golden-vectors.txt').read_bytes()
    assert actual == expected, 'C output differs from frozen golden vectors'
    model = runpy.run_path(str(ref / 'reference.py'))
    lines = expected.decode().splitlines()
    assert len(lines) == 1600
    for line in lines:
        seed, lane, base, word, mantissa = (int(x, 16) for x in line.split())
        assert model['at'](seed, lane, base + 1) == word
        assert word >> 11 == mantissa
    cert = json.loads((ROOT / 'studies/audit-2026-10-21/evidence/exact.json').read_text())
    expected_inputs = [int(v['mixer_input'], 16) for v in cert['noninjective_mixer_certificate']]
    assert model['preimages'](7) == expected_inputs
    assert all(model['mixer'](x) == 7 for x in expected_inputs)
    for entry in cert['noninjective_mixer_certificate']:
        assert model['at'](42, 0, entry['draw_index_mod_2_64']) == 7
    first_bin_count = sum(len(model['preimages'](y)) for y in range(2048))
    assert first_bin_count == cert['first_mantissa_bin']['exact_state_count'] == 2066
    assert model['state'](42, 0, 2**63) == model['state'](42, 1, 2**63)
    assert model['metal_mantissa'](42, 2, 4, 2) != model['metal_mantissa'](42, 4, 2, 2)
    result = {
        'status': 'pass', 'scope': 'CPU reference and exact integer certificates; no GPU/TPU or statistical battery',
        'c_and_python_golden_vectors': len(lines), 'ubsan': True,
        'golden_sha256': hashlib.sha256(actual).hexdigest(),
        'noninjective_preimages_of_7': len(expected_inputs),
        'mantissa_zero_exact_preimage_count': first_bin_count,
        'lane_same_state_at_2_to_63': True, 'logical_geometry_changes_stream': True,
        'compiler': subprocess.check_output([*cc, '--version'], text=True).splitlines()[0],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
