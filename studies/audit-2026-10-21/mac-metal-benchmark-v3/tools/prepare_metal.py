#!/usr/bin/env python3
"""Preprocess official headers for runtime Metal compilation without include paths.

Only preprocessing is performed here. Apple SDK headers remain runtime includes.
No algorithm is rewritten; the generated functions are checked separately on CPU.
"""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for name in ('metal_stdlib','metal_integer'):
            (tmp/name).write_text('MONTMORY_INCLUDE_'+name.upper()+'\n')
        cmd=[os.environ.get('CXX','c++'),'-E','-P','-x','c++','-std=c++17',
             '-D__METAL_MACOS__=1','-D__METAL_VERSION__=240','-DR123_USE_SSE=0',
             '-DR123_USE_AES_NI=0','-I',str(tmp),'-I',str(ROOT/'vendor/include'),
             str(ROOT/'src/bench.metal')]
        source=subprocess.check_output(cmd,text=True)
    source=source.replace('MONTMORY_INCLUDE_METAL_STDLIB','#include <metal_stdlib>')
    source=source.replace('MONTMORY_INCLUDE_METAL_INTEGER','#include <metal_integer>')
    source='// Generated from the shipped sources; Random123 license: vendor/LICENSE.Random123.\n'+source
    (ROOT/'generated/all.metal').write_text(source)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for prefix in ('src','vendor') for p in sorted((ROOT/prefix).rglob('*')) if p.is_file()}
    metadata={'random123_commit':'9545ff6413f258be2f04c1d319d99aaef7521150',
              'method':'C++ preprocessing with __METAL_MACOS__, Apple includes retained; no Metal compilation on Linux',
              'generated_sha256':hashlib.sha256(source.encode()).hexdigest(),'inputs':sources}
    (ROOT/'generated/provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Generated',len(source),'bytes of runtime Metal source.')

if __name__=='__main__': main()
