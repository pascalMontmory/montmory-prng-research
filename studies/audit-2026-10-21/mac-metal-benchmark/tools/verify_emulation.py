#!/usr/bin/env python3
"""Execute generated Metal integer bodies on CPU before shipping the Mac kit.

This verifies arithmetic and output assignment, not Apple's compiler or device.
"""
import ctypes as C
import hashlib,json
from pathlib import Path
import re,subprocess,tempfile

ROOT=Path(__file__).resolve().parents[1]
class Params(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('seed','base','words','k0','k1','k2')]+[
        (n,C.c_uint32) for n in ('r0','r1','r2','key0','key1','logical_lanes','draws_shift','physical_threads','output_shift','reserved')]
assert C.sizeof(Params)==88

def compile_library(code,path):
    cpp=path.with_suffix('.cpp'); cpp.write_text(code)
    subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared',
                    '-fsanitize=undefined','-fno-sanitize-recover=all',
                    '-I',str(ROOT/'src'),'-I',str(ROOT/'vendor/include'),str(cpp),'-o',str(path)],check=True)
    return C.CDLL(str(path))

def main():
    source=(ROOT/'generated/all.metal').read_text()
    source=re.sub(r'#include <metal_\w+>','',source)
    source=re.sub(r'\[\[.*?\]\]','',source)
    for old,new in [('kernel',''),('device',''),('thread',''),('constant','const')]:
        source=re.sub(r'\b'+old+r'\b',new,source)
    source=re.sub(r'\bconst\s+const\b','const',source)
    source='''#include <stdint.h>
using uint=uint32_t; using ulong=uint64_t; using uchar=uint8_t;
namespace metal { inline uint32_t mulhi(uint32_t a,uint32_t b) {return (uint64_t(a)*b)>>32;} }
'''+source+'''
extern "C" void fill(unsigned engine, const BenchParams* p, uint64_t* out) {
 if(engine==3) { Params legacy{p->seed,p->base,uint64_t(1)<<p->draws_shift};
  for(uint32_t tid=0;tid<p->logical_lanes;++tid) mmc_fill(out,legacy,tid);
 } else for(uint32_t tid=0;tid<p->physical_threads;++tid) {
  if(engine==0) montmory_indexed(out,*p,tid);
  else if(engine==1) threefry_bulk(out,*p,tid);
  else philox_bulk(out,*p,tid);
 }
}
extern "C" void kats(const uint32_t* in,uint32_t* out) {
 for(unsigned i=0;i<6;++i) reference_kat(out,in,i);
}
'''
    reference='''#include "reference.hpp"
extern "C" void params(uint64_t seed,uint64_t base,uint64_t n,uint32_t lanes,uint32_t workers,uint32_t shift,BenchParams* p) {
 *p=parameters(seed,base,n,lanes,workers,shift);
}
extern "C" void expected(unsigned engine,const BenchParams* p,uint64_t* out) {
 for(uint64_t i=0;i<p->words;++i) out[i]=referenceWord(engine,*p,i);
}
extern "C" void get_kats(const char* filename,uint32_t* in,uint32_t* out) {
 auto cases=knownAnswers(filename);
 for(unsigned i=0;i<6;++i) {
  std::copy(cases[i].input.begin(),cases[i].input.end(),in+6*i);
  std::copy(cases[i].expected.begin(),cases[i].expected.end(),out+4*i);
 }
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp); gpu=compile_library(source,tmp/'metal.so'); cpu=compile_library(reference,tmp/'reference.so')
        gpu.fill.argtypes=[C.c_uint,C.POINTER(Params),C.POINTER(C.c_uint64)]
        gpu.kats.argtypes=[C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)]
        cpu.params.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(Params)]
        cpu.expected.argtypes=[C.c_uint,C.POINTER(Params),C.POINTER(C.c_uint64)]
        cpu.get_kats.argtypes=[C.c_char_p,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)]
        for lib,names in ((gpu,['fill','kats']),(cpu,['params','expected','get_kats'])):
            for name in names: getattr(lib,name).restype=None
        inputs=(C.c_uint32*36)(); wanted=(C.c_uint32*24)(); got=(C.c_uint32*24)()
        cpu.get_kats(str(ROOT/'reference/kat_vectors').encode(),inputs,wanted); gpu.kats(inputs,got)
        assert list(got)==list(wanted)
        checks=0; maximum=(1<<64)-1
        for n,lanes in ((64,8),(2048,32),(65536,8192)):
            for seed in (0,1,42,maximum):
                for base in (0,n,maximum-3):
                    for workers in (7,64,257):
                        for shift in (0,11):
                            p=Params(); cpu.params(seed,base,n,lanes,workers,shift,C.byref(p))
                            for engine in range(4 if shift==11 else 3):
                                result=(C.c_uint64*(n+16))(*([maximum]*(n+16)))
                                expected=(C.c_uint64*n)()
                                gpu.fill(engine,C.byref(p),result); cpu.expected(engine,C.byref(p),expected)
                                assert list(result[:n])==list(expected),(n,seed,base,workers,shift,engine)
                                assert all(x==maximum for x in result[n:])
                                checks+=n
        report={'status':'pass','scope':'Generated Metal arithmetic emulated on x86 C++ with UBSan; not a native Metal run',
                'generated_source_sha256':hashlib.sha256((ROOT/'generated/all.metal').read_bytes()).hexdigest(),
                'words_compared':checks,'official_known_answer_vectors':6,
                'guards':'all preserved','native_Mac_status':'not run on this VPS'}
        (ROOT/'generated/linux-verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

if __name__=='__main__': main()
