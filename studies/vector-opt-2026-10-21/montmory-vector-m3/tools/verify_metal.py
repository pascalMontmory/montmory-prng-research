#!/usr/bin/env python3
"""Execute actual new shader arithmetic/ownership in C++ SIMD under UBSan.
Metal address-space attributes and vector constructors are translated.
This does not run Apple's compiler, memory model, or GPU.
"""
from pathlib import Path
import hashlib,json,re,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    parts=['generated/previous.metal','src/wide_metal_prefix.inc','src/wide_math.inc','src/wide_metal_suffix.inc']
    generated='\n'.join((ROOT/p).read_text() for p in parts)
    assert generated==(ROOT/'generated/all.metal').read_text()
    source=generated
    prefix=(ROOT/'src/wide_metal_prefix.inc').read_text()
    source=source.replace(prefix,'''namespace montmory_wide_metal {
using V=montmory_wide::V;
using montmory_wide::splat;
using montmory_wide::mask_lt;
using montmory_wide::mask_ne;
using montmory_wide::high_product;
''')
    source=source.replace('V(uint(c),uint(c+step),uint(c+2*step),uint(c+3*step))','V{uint(c),uint(c+step),uint(c+2*step),uint(c+3*step)}')
    source=source.replace('V(uint(c>>32),uint((c+step)>>32),uint((c+2*step)>>32),uint((c+3*step)>>32))','V{uint(c>>32),uint((c+step)>>32),uint((c+2*step)>>32),uint((c+3*step)>>32)}')
    source=re.sub(r'#include <metal_\w+>','',source)
    source=re.sub(r'\[\[.*?\]\]','',source)
    source=re.sub(r'constant uint mmc_rot([012])\s*;',r'uint mmc_rot\1=1;',source)
    source=re.sub(r'^\s*#pragma unroll\s*$','',source,flags=re.M)
    for a,b in [('kernel',''),('device',''),('thread',''),('constant','const')]:
        source=re.sub(r'\b'+a+r'\b',b,source)
    source=re.sub(r'\bconst\s+const\b','const',source)
    source='''#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include "wide_cpu.hpp"
using uint=uint32_t;using ulong=uint64_t;using uchar=uint8_t;
namespace metal {inline uint mulhi(uint a,uint b){return uint(uint64_t(a)*b>>32);}}
namespace shader {
'''+source+'\n}\n'+r'''
#include "reference.hpp"
int main() {
    uint64_t checked=0,cases=0;
    for(auto geometry: {std::pair<uint64_t,uint32_t>{32,32},{64,32},{2048,32},{65536,2048},{65536,32},{8388608,2048}}) {
        auto n=geometry.first;auto lanes=geometry.second;
        for(uint64_t seed:{UINT64_C(0),UINT64_C(1),UINT64_C(42),UINT64_MAX})
        for(uint64_t base:{UINT64_C(0),n,UINT64_MAX-3})
        for(unsigned workers:{7u,lanes,lanes*8,lanes*32,257u})
        for(unsigned shift:{0u,11u}) {
            // Large blocks exercise native-like scheduling; small blocks exhaust edge patterns.
            if(n>65536 && (seed!=42 || base!=n || workers!=lanes*32))continue;
            BenchParams p=parameters(seed,base,n,lanes,workers,shift);
            shader::BenchParams q;static_assert(sizeof(q)==sizeof(p),"ABI");std::memcpy(&q,&p,sizeof(p));
            std::vector<uint64_t> expected(n),out(n+2);
            for(uint64_t i=0;i<n;++i)expected[i]=referenceWord(10,p,i);
            for(unsigned engine:{20u,21u}) {
                for(uint64_t i=0;i<n;++i)out[i+1]=~expected[i];out[0]=out[n+1]=0x12345678;
                for(unsigned tid=0;tid<workers;++tid) {
                    if(engine==20)shader::montmory_vector4(out.data()+1,q,tid);
                    else shader::montmory_vector4_striped(out.data()+1,q,tid);
                }
                if(!std::equal(expected.begin(),expected.end(),out.begin()+1)||out[0]!=0x12345678||out[n+1]!=0x12345678)
                    throw std::runtime_error("Shader transcription mismatch");
                checked+=n;++cases;
            }
        }
    }
    std::cout<<"{\"status\":\"pass\",\"words\":"<<checked<<",\"cases\":"<<cases<<"}\n";
}
'''
    # Keep the strict warning policy; use separate lines for test buffer setup.
    source=source.replace('out[i+1]=~expected[i];out[0]', 'out[i+1]=~expected[i];\n                out[0]')
    with tempfile.TemporaryDirectory() as d:
        cpp=Path(d)/'verify.cpp';binary=Path(d)/'verify';cpp.write_text(source)
        subprocess.run(['g++','-std=c++17','-O2','-march=native','-Wall','-Wextra','-Werror',
            '-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(ROOT/'src'),
            '-I'+str(ROOT/'vendor/include'),str(cpp),'-o',str(binary)],check=True)
        report=json.loads(subprocess.check_output([str(binary)],text=True))
    report.update(source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        scope='Actual generated shader bodies transcribed to x86 C++ SIMD with UBSan; not native Metal',
        native_metal_status='pending')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
