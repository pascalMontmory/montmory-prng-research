#!/usr/bin/env python3
"""Verify candidate integer kernels as C++ against the original scalar C oracle.

This does not compile Metal or simulate GPU scheduling/performance.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
import verify_emulation as v

ROOT=Path(__file__).resolve().parents[1]
v.ROOT=ROOT
MAX=(1<<64)-1
reference='''#include "lane_reference.hpp"
#include <algorithm>
extern "C" void params(uint64_t seed,uint64_t base,uint64_t n,uint32_t lanes,uint32_t workers,uint32_t shift,BenchParams* p) {
 *p=parameters(seed,base,n,lanes,workers,shift);
}
extern "C" void expected(const BenchParams* p,uint64_t* out) {
 auto values=laneReference(*p,10);std::copy(values.begin(),values.end(),out);
}
extern "C" void poison(const uint64_t* expected,uint64_t* actual,uint64_t n) {
 for(uint64_t i=0;i<n;++i)actual[i]=~expected[i];
}
'''
components='''
extern "C" unsigned opt_checks(uint64_t* count) {
 uint xs[]={0,1,2,3,0x7fffffffu,0x80000000u,0x80000001u,0xfffffffeu,0xffffffffu};
 ulong keys[]={0,1,0xffffffffUL,0x100000000UL,0x8000000000000000UL,0xffffffffffffffffUL};
 *count=0;
 for(uint x:xs)for(ulong key:keys)for(uint a=1;a<64;a+=2)for(uint b=1;b<64;b+=2) {
  if(opt_f<true>(x,key,a,b)!=mmc_f(ulong(x),key,a,b))return 1;
  if(opt_f<false>(x,key,a,b)!=mmc_f(ulong(x),key,a,b))return 2;
  ++*count;
 }
 ulong r=1234;
 for(unsigned i=0;i<100000;++i) {
  r=splitmix64(r);ulong a=r;r=splitmix64(r);ulong b=r;
  MmcPair32 c=opt_add(opt_pair(a),opt_pair(b));
  if((ulong(c.lo)|(ulong(c.hi)<<32))!=a+b)return 3;
  ++*count;
 }
 return 0;
}
'''
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    gpu=v.compile_library(v.emulated_source()+components,tmp/'gpu.so')
    cpu=v.compile_library(reference,tmp/'cpu.so')
    gpu.fill.argtypes=[C.c_uint,C.POINTER(v.Params),C.POINTER(C.c_uint64),C.c_bool]
    gpu.fill.restype=None
    gpu.opt_checks.argtypes=[C.POINTER(C.c_uint64)];gpu.opt_checks.restype=C.c_uint
    cpu.params.argtypes=[C.c_uint64,C.c_uint64,C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(v.Params)]
    cpu.params.restype=None
    cpu.expected.argtypes=[C.POINTER(v.Params),C.POINTER(C.c_uint64)];cpu.expected.restype=None
    cpu.poison.argtypes=[C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.c_uint64];cpu.poison.restype=None
    component_count=C.c_uint64()
    assert gpu.opt_checks(C.byref(component_count))==0,'component mismatch'
    cases=0;words=0
    def check(n,lanes,workers,seed,base,shift):
        global cases,words
        p=v.Params();cpu.params(seed,base,n,lanes,workers,shift,C.byref(p))
        expected=(C.c_uint64*n)();cpu.expected(C.byref(p),expected)
        expected_bytes=bytes(expected)
        for engine in range(13,20):
            actual=(C.c_uint64*(n+16))();C.memset(actual,255,C.sizeof(actual))
            # Complement the oracle, so no unfilled word can accidentally pass.
            cpu.poison(expected,actual,n)
            gpu.fill(engine,C.byref(p),actual,False)
            assert C.string_at(actual,n*8)==expected_bytes,(n,lanes,workers,seed,base,shift,engine)
            assert all(x==MAX for x in actual[n:]),'guard overwritten'
            cases+=1;words+=n
    for n,lanes in [(32,32),(64,32),(2048,32),(65536,2048),(65536,8192)]:
        for workers in [7,lanes,lanes*8,lanes*32,257]:
            for seed in [0,1,42,MAX]:
                for base in [0,n,MAX-3]:
                    for shift in [0,11]:check(n,lanes,workers,seed,base,shift)
        print('Geometry PASS:',n,lanes,flush=True)
    # Actual benchmark volume for all physical-thread counts, both formats.
    for workers in [8192,32768,65536,131072]:
        for shift in [0,11]:check(8388608,2048,workers,42,0,shift)
        print('Full volume PASS, workers:',workers,flush=True)
    report={'status':'pass','cases':cases,'words_compared':words,
            'component_checks':component_count.value,'guards':'preserved',
            'source_sha256':hashlib.sha256((ROOT/'generated/all.metal').read_bytes()).hexdigest(),
            'sanitizer':'UBSan, abort on first error',
            'scope':'C++ transcription of all seven new kernels versus original scalar C; fallback/tiled paths, short lanes, idle threads, carry/wrap boundaries',
            'native_Mac_status':'not run on this VPS; native full-output checks required by run.sh'}
    (ROOT/'generated/optimization-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
