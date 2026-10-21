#!/usr/bin/env python3
"""Audit recovered CUDA arithmetic on CPU, without claiming a CUDA execution."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
import reference as ref
from verify_exact import metal_library

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / 'snapshot/cuda-ovh'
BUILD = ROOT / 'build'
U64 = C.c_ulonglong
PREFIX = '''#include <stdint.h>
#include <cstdlib>
#define __host__
#define __device__
#define __global__
struct Dim { unsigned x; };
static Dim blockIdx{}, blockDim{256}, threadIdx{};
'''


def compile_library(source, name):
    cpp = BUILD / (name + '.cpp')
    so = BUILD / (name + '.so')
    cpp.write_text(source)
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unknown-pragmas', '-fPIC', '-shared',
                    '-I', str(ROOT / 'deps/random123/include'),
                    '-I', str(ROOT / 'reviewed-source/cuda/src'),
                    str(cpp), '-o', str(so)], check=True)
    return C.CDLL(str(so))


def kernel_library(corrected=False):
    directory = ROOT / 'reviewed-source/cuda' if corrected else CUDA
    source = PREFIX + (directory / 'src/mmc_kernel.cu').read_text()
    source += '''
extern "C" void audit_cuda_fill(unsigned long long seed, unsigned long long base,
 unsigned long long draws, unsigned threads, unsigned long long* out) {
 Params p{seed,base,draws};
 unsigned launched = ((threads+255)/256)*256;
 for(unsigned tid=0;tid<launched;++tid) {
  blockIdx.x=tid/256; threadIdx.x=tid%256;
  mmc_fill_kernel(out,p);
 }
}
'''
    if corrected:
        source = source.replace('Params p{seed,base,draws};', 'Params p{seed,base,draws,threads};')
        source = source.replace('mmc_fill_kernel(out,p);','mmc_fill_kernel(reinterpret_cast<uint64_t*>(out),p);')
    lib = compile_library(source, 'cuda_corrected_emulated' if corrected else 'cuda_emulated')
    lib.audit_cuda_fill.argtypes = [U64, U64, U64, C.c_uint, C.POINTER(U64)]
    lib.audit_cuda_fill.restype = None
    return lib


def check_benchmark(kind):
    path = CUDA / f'src/bench_montmory_vs_philox_{kind}.cu'
    source = path.read_text().split('static float run_montmory')[0]
    source = source.replace('#include <cuda_runtime.h>', '#include <Random123/philox.h>')
    typ = 'double' if kind == 'double' else 'unsigned long long'
    function = 'philox_generate2' + ('' if kind == 'raw64' else '_' + kind)
    transform = {'raw64':'raw', 'mant53':'(raw >> 11)',
                 'double':'double(raw >> 11) * 0x1p-53'}[kind]
    source = PREFIX + source + f'''
extern "C" void audit_compare(unsigned long long seed, unsigned long long lane,
 unsigned calls, {typ}* old, {typ}* standard) {{
 philox_state_t st; philox_init_device(st,seed,lane);
 r123::Philox4x32::ctr_type ctr{{{{uint32_t(lane),uint32_t(lane>>32),0,0}}}};
 r123::Philox4x32::key_type key{{{{uint32_t(seed),uint32_t(seed>>32)}}}};
 for(unsigned i=0;i<calls;++i) {{
  {function}(st,old[2*i],old[2*i+1]);
  auto result=r123::Philox4x32{{}}(ctr,key);
  for(unsigned j=0;j<2;++j) {{
   unsigned long long raw=(uint64_t(result.v[2*j])<<32)|result.v[2*j+1];
   standard[2*i+j]={transform};
  }}
  for(unsigned j=0;j<4;++j) {{ if(++ctr.v[j]) break; }}
 }}
}}
extern "C" void audit_montmory(unsigned long long seed, unsigned lanes,
 unsigned long long draws, {typ}* out) {{
 for(unsigned tid=0;tid<lanes;++tid) {{
  blockIdx.x=tid/256; threadIdx.x=tid%256;
  montmory_{kind}_kernel(out,seed,draws);
 }}
}}
'''
    lib = compile_library(source, 'cuda_benchmark_' + kind)
    ctype = C.c_double if kind == 'double' else U64
    lib.audit_compare.argtypes = [U64, U64, C.c_uint, C.POINTER(ctype), C.POINTER(ctype)]
    lib.audit_compare.restype = None
    lib.audit_montmory.argtypes = [U64, C.c_uint, U64, C.POINTER(ctype)]
    lib.audit_montmory.restype = None
    divergences = []
    for seed in (0,42,0x123456789abcdef0,ref.MASK):
        for lane in (0,1,65535,1<<32):
            old, standard = (ctype*8)(), (ctype*8)()
            lib.audit_compare(seed,lane,4,old,standard)
            assert list(old[:2]) == list(standard[:2])
            assert list(old[2:4]) != list(standard[2:4])
            divergences.append({'seed':seed,'lane':lane,
                                'first_block_equal':True,
                                'second_block_archived':list(old[2:4]),
                                'second_block_standard':list(standard[2:4])})
    nchecked = 0
    for seed in (0,42,ref.MASK):
        lanes, draws = 17, 31
        out = (ctype*(lanes*draws))()
        lib.audit_montmory(seed,lanes,draws,out)
        for i, got in enumerate(out):
            want = ref.at(seed,i//draws,i+1)
            if kind != 'raw64': want >>= 11
            if kind == 'double': want *= 2.0**-53
            assert got == want, (kind,seed,i,got,want)
            nchecked += 1
    return {'file':str(path.relative_to(ROOT)),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'montmory_values_verified':nchecked,
            'philox_divergence_cases':divergences}


def check_logs():
    logs = []
    metal = ROOT / 'snapshot/montmory_gpu_test/montmory_gpu_test/logs'
    for name in ('smallcrush','crush','bigcrush'):
        p = CUDA / f'logs/{name}_gpu.log'
        cuda_text = p.read_text()
        metal_text = (metal / p.name).read_text()
        def pvalues(t):
            return re.findall(r'^.*p-value.*?:\s*(.*?)\s*$',t,re.M|re.I)
        cp,mp = pvalues(cuda_text),pvalues(metal_text)
        assert cp and cp==mp
        assert 'All tests were passed' in cuda_text
        logs.append({'battery':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                     'statistics':int(re.search(r'Number of statistics:\s*(\d+)',cuda_text)[1]),
                     'printed_p_values':len(cp),'same_printed_p_values_as_metal':True,
                     'total_CPU_time':re.search(r'Total CPU time:\s*(\S+)',cuda_text)[1],
                     'header':cuda_text.splitlines()[0],
                     'status':'historical log: all tests passed'})
    return logs


def main():
    BUILD.mkdir(exist_ok=True)
    cuda,metal,fixed = kernel_library(),metal_library(),kernel_library(corrected=True)
    checked = 0
    for seed in (0,1,42,ref.MASK):
        for lanes,draws in ((256,1),(256,3),(512,7),(8192,2)):
            n = lanes*draws
            for base in (0,n,ref.MASK-3):
                cu,mt,fx = (U64*n)(), (C.c_uint64*n)(), (U64*n)()
                cuda.audit_cuda_fill(seed,base,draws,lanes,cu)
                metal.audit_fill(seed,base,draws,lanes,mt)
                fixed.audit_cuda_fill(seed,base,draws,lanes,fx)
                for i,(a,b,c) in enumerate(zip(cu,mt,fx)):
                    assert a==b==c==ref.at(seed,i//draws,base+i+1)>>11
                    checked += 1
    # Allocate the padded extent deliberately; detect writes beyond the user's
    # intended allocation without causing undefined behavior in this audit.
    violations = []
    for lanes in (1,7,255,257,511):
        draws=3; launched=((lanes+255)//256)*256; size=launched*draws
        out=(U64*size)(*([ref.MASK]*size))
        cuda.audit_cuda_fill(42,0,draws,lanes,out)
        extra=sum(x!=ref.MASK for x in out[lanes*draws:])
        assert extra==(launched-lanes)*draws
        fx=(U64*size)(*([ref.MASK]*size))
        fixed.audit_cuda_fill(42,0,draws,lanes,fx)
        assert list(fx[:lanes*draws])==list(out[:lanes*draws])
        assert all(x==ref.MASK for x in fx[lanes*draws:])
        violations.append({'requested_lanes':lanes,'launched_threads':launched,
                           'draws':draws,'writes_beyond_requested_allocation':extra,
                           'corrected_extra_writes':0,'corrected_preserves_valid_outputs':True})
    result={'scope':'Original CUDA functions mechanically compiled on x86 C++; no CUDA compiler or GPU execution',
            'status':'checks passed; defects reproduced',
            'cuda_metal_reference_equal_mantissas':checked,
            'C_source_byte_identical_to_Metal_archive':(CUDA/'src/mmc_core.c').read_bytes()==(ROOT/'snapshot/montmory_gpu_test/montmory_gpu_test/src/mmc_core.c').read_bytes(),
            'missing_thread_guard_witnesses':violations,
            'corrected_kernel_preserves_all_compared_valid_outputs':True,
            'corrected_wrapper_note':'Wrapper reviewed statically only; no nvcc or GPU on this VPS',
            'historical_logs':check_logs(),
            'benchmarks':[check_benchmark(k) for k in ('raw64','mant53','double')]}
    (ROOT/'evidence/cuda-exact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'mantissas':checked,'guard_witnesses':len(violations),
                      'benchmark_Philox_divergence_cases':48,'historical_logs':result['historical_logs']},indent=2))


if __name__=='__main__':
    main()
