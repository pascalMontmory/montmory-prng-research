#!/usr/bin/env python3
"""Compile frozen recovered source and check exact, reproducible certificates."""

import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import reference as ref

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'snapshot/montmory_gpu_test/montmory_gpu_test'
BUILD = ROOT / 'build'
EVIDENCE = ROOT / 'evidence'


def command(args):
    p = subprocess.run(args, check=True, text=True, capture_output=True)
    return p.stdout


def native_library():
    BUILD.mkdir(exist_ok=True)
    bridge = BUILD / 'bridge.c'
    bridge.write_text('''#include "mmc_core.c"
uint64_t audit_mix(uint64_t x) { return pcg_dxsm(x); }
uint64_t audit_perm(uint64_t seed, uint64_t x) {
    mmc_state_t s; mmc_init(&s, seed, 0); return perm3(x, &s);
}
uint64_t audit_at(uint64_t seed, uint64_t lane, uint64_t n) {
    mmc_state_t s; mmc_init(&s, seed, lane);
    s.s += (n - UINT64_C(1)) * s.gamma; return mmc_next_u64(&s);
}
''')
    libpath = BUILD / 'legacy.so'
    command(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
             '-fPIC', '-shared', '-I', str(PROJECT / 'src'), str(bridge),
             '-o', str(libpath)])
    lib = ctypes.CDLL(str(libpath))
    for name, argc in [('audit_mix', 1), ('audit_perm', 2), ('audit_at', 3)]:
        f = getattr(lib, name)
        f.argtypes = [ctypes.c_uint64] * argc
        f.restype = ctypes.c_uint64
    return lib


def metal_library():
    # Mechanical syntax adaptation only; the complete kernel body is retained.
    source = (PROJECT / 'metal/mmc_kernel.metal').read_text()
    source = source.replace('#include <metal_stdlib>', '#include <stdint.h>')
    source = source.replace('using namespace metal;', '')
    source = re.sub(r'\[\[.*?\]\]', '', source)
    for old, new in [('ulong', 'uint64_t'), ('uint', 'uint32_t'),
                     ('constant', 'const'), ('thread', ''), ('device', ''),
                     ('kernel', '')]:
        source = re.sub(r'\b' + old + r'\b', new, source)
    source += '''\nextern "C" void audit_fill(uint64_t seed, uint64_t base,
    uint64_t draws, uint32_t threads, uint64_t *out) {
    Params p = {seed, base, draws};
    for (uint32_t tid=0; tid<threads; ++tid) mmc_fill(out, p, tid);
}\n'''
    cpp = BUILD / 'metal_emulated.cpp'
    cpp.write_text(source)
    libpath = BUILD / 'metal_emulated.so'
    command(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
             '-fPIC', '-shared', str(cpp), '-o', str(libpath)])
    lib = ctypes.CDLL(str(libpath))
    lib.audit_fill.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                              ctypes.c_uint64, ctypes.c_uint32,
                              ctypes.POINTER(ctypes.c_uint64)]
    lib.audit_fill.restype = None
    return lib


def main():
    lib = native_library()
    metal = metal_library()
    results = {'scope': 'CPU x86 execution; Metal integer kernel emulated in C++; no GPU execution'}
    seeds = [0, 1, 42, ref.MASK, 1 << 63]
    x = 42
    for _ in range(59):
        x = ref.splitmix(x)
        seeds.append(x)
    nchecks = 0
    vectors = []
    for seed in seeds:
        for lane in [0, 1, 2, 8191, (1 << 32) - 1, (1 << 63)]:
            for n in [0, 1, 2, 3, 17, (1 << 32) - 1, 1 << 63, ref.MASK]:
                want = ref.at(seed, lane, n)
                assert lib.audit_at(seed, lane, n) == want
                s = ref.state(seed, lane, n)
                p = ref.feistel(s, seed)
                assert ref.feistel(p, seed, inverse=True) == s
                assert lib.audit_perm(seed, s) == p
                nchecks += 1
                if seed == 42 and lane in (0, 1) and n in (1, 2, 3, 17):
                    vectors.append({'seed':seed, 'lane':lane, 'n':n, 'u64':hex(want), 'mantissa':want >> 11})
    results['cpu_reference_cases'] = nchecks
    metal_checks = 0
    for seed in [0, 1, 42, ref.MASK]:
        for threads, draws in [(1, 1), (2, 3), (7, 11), (32, 16)]:
            size = threads * draws
            for base in [0, size, 2 * size, ref.MASK - 3]:
                out = (ctypes.c_uint64 * size)()
                metal.audit_fill(seed, base, draws, threads, out)
                for i, got in enumerate(out):
                    lane = i // draws
                    assert got == ref.at(seed, lane, base + i + 1) >> 11
                    metal_checks += 1
    results['metal_emulation_values'] = metal_checks
    xs = ref.preimages(7)
    assert len(xs) == 4 and len(set(xs)) == 4
    cert = []
    for x in xs:
        assert ref.mixer(x) == 7 and lib.audit_mix(x) == 7
        s = ref.feistel(x, 42, inverse=True)
        n = ((s - 42) * pow(ref.GAMMA, -1, ref.MOD)) & ref.MASK
        assert lib.audit_at(42, 0, n) == 7
        cert.append({'mixer_input':hex(x), 'counter_state':hex(s), 'draw_index_mod_2_64':n, 'output':7})
    results['noninjective_mixer_certificate'] = cert
    # All outputs 0..2047 map to the first 53-bit mantissa bin.
    first_bin_preimages = sum(len(ref.preimages(y)) for y in range(2048))
    assert first_bin_preimages != 2048
    results['first_mantissa_bin'] = {'exact_state_count':first_bin_preimages,
                                    'uniform_state_count':2048, 'total_states':ref.MOD}
    first = [ref.metal_mantissa(42, 2, 4, i) for i in range(16)]
    second = [ref.metal_mantissa(42, 4, 2, i) for i in range(16)]
    i = next(i for i in range(16) if first[i] != second[i])
    results['geometry_dependency'] = {'first_differing_zero_based_index':i,
                                      'threads_2_draws_4':first[i], 'threads_4_draws_2':second[i]}
    for n in [1, 2, 3, 17, 12345, ref.MASK]:
        m = n * ref.GAMMA * pow((ref.GAMMA + 2) & ref.MASK, -1, ref.MOD) & ref.MASK
        assert ref.state(42, 0, n) == ref.state(42, 1, m)
        assert lib.audit_at(42, 0, n) == lib.audit_at(42, 1, m)
    results['lane_reindexing'] = {'lane_0_n':1,
        'lane_1_n':ref.GAMMA * pow((ref.GAMMA+2)&ref.MASK, -1, ref.MOD) & ref.MASK,
        'same_step_collision_n':1 << 63}
    assert lib.audit_at(42, 0, 1 << 63) == lib.audit_at(42, 1, 1 << 63)
    # Manifest proves identity of these bytes, not which executable ran a test.
    manifest = []
    archive = PROJECT / 'Montmory_GPU3R_Certified'
    for line in (archive / 'SHA256SUMS.txt').read_text().splitlines():
        want, name = line.split()
        got = hashlib.sha256((archive / name).read_bytes()).hexdigest()
        assert want == got
        manifest.append({'file':name,'sha256':got})
    results['archive_manifest_verified'] = manifest
    results['golden_vectors'] = vectors
    results['status'] = 'pass: exact identities and counterexamples verified'
    (EVIDENCE / 'exact.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k:v for k,v in results.items() if k not in ('golden_vectors','archive_manifest_verified','noninjective_mixer_certificate')},indent=2))


if __name__ == '__main__':
    main()
