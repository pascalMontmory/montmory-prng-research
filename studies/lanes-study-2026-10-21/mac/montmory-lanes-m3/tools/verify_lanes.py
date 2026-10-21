#!/usr/bin/env python3
"""Execute unchanged Metal integer bodies as C++ with UBSan for new geometries.

This cannot validate Objective-C/Metal compilation or Apple GPU timing.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
import verify_emulation as v

ROOT=Path(__file__).resolve().parents[1]
v.ROOT=ROOT
reference='''#include "lane_reference.hpp"
#include <algorithm>
extern "C" void params(uint64_t seed,uint64_t base,uint64_t n,uint32_t lanes,uint32_t workers,uint32_t shift,BenchParams* p) {
 *p=parameters(seed,base,n,lanes,workers,shift);
}
extern "C" int expected(unsigned engine,const BenchParams* p,uint64_t* out) {
 auto values=laneReference(*p,engine);
 // Check the new efficient oracle against the old scalar index oracle too.
 for(uint64_t i=0;i<p->words;i+=(p->words<100000?1:257))
  if(values[i]!=referenceWord(engine,*p,i)) return 1;
 if(values.back()!=referenceWord(engine,*p,p->words-1))return 2;
 std::copy(values.begin(),values.end(),out);return 0;
}
'''
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    gpu=v.compile_library(v.emulated_source(),tmp/'gpu.so')
    cpu=v.compile_library(reference,tmp/'cpu.so')
    gpu.fill.argtypes=[C.c_uint,C.POINTER(v.Params),C.POINTER(C.c_uint64),C.c_bool]
    gpu.fill.restype=None
    cpu.params.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(v.Params)]
    cpu.params.restype=None
    cpu.expected.argtypes=[C.c_uint,C.POINTER(v.Params),C.POINTER(C.c_uint64)]
    cpu.expected.restype=C.c_int
    cases=0;words=0
    def check(n,lanes,workers,seed,base,shift,engine):
        global cases,words
        p=v.Params();cpu.params(seed,base,n,lanes,workers,shift,C.byref(p))
        actual=(C.c_uint64*(n+16))();C.memset(actual,255,C.sizeof(actual))
        expected=(C.c_uint64*n)()
        assert cpu.expected(engine,C.byref(p),expected)==0,'efficient oracle mismatch'
        gpu.fill(engine,C.byref(p),actual,False)
        assert C.string_at(actual,n*8)==bytes(expected),(n,lanes,workers,seed,base,shift,engine)
        assert all(x==(1<<64)-1 for x in actual[n:]),'guard overwritten'
        cases+=1;words+=n
    for lanes in [256,512,1024,2048,4096,8192,16384]:
        for workers in [1024,2048,8192,32768,65536,131072,262144]:
            for seed in [0,42,(1<<64)-1]:
                for shift in [0,11]:
                    for engine in [10,1,2]:
                        check(65536,lanes,workers,seed,0,shift,engine)
        print('Reduced-size geometry passed:',lanes,flush=True)
    # Actual benchmark volume, every logical geometry and physical thread count.
    for lanes in [256,512,1024,2048,4096,8192,16384]:
        check(8388608,lanes,65536,42,0,0,10)
        print('Full-volume geometry passed:',lanes,flush=True)
    for workers in [1024,2048,8192,32768,131072,262144]:
        check(8388608,2048,workers,42,0,11,10)
    # Subsequent batches follow the same contract when base is a whole block.
    for seed in [0,42,(1<<64)-1]:
        for engine in [10,1,2]:
            check(65536,2048,8192,seed,65536,0,engine)
    report={'status':'pass','cases':cases,'words_compared':words,
            'generated_source_sha256':hashlib.sha256((ROOT/'generated/all.metal').read_bytes()).hexdigest(),
            'sanitizer':'undefined, abort on first error','guards':'preserved',
            'scope':'C++ execution of translated Metal integer bodies, all proposed geometries; efficient CPU oracle checked against original scalar oracle',
            'native_Mac_status':'not run on VPS; native compilation and all-output checks occur in run.sh'}
    (ROOT/'generated/linux-lanes-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
