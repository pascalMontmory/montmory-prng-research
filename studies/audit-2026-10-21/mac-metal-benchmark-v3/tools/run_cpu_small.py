#!/usr/bin/env python3
"""SmallCrush of the generated Metal arithmetic, executed on CPU, not on GPU."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,tempfile
from verify_emulation import ROOT,emulated_source

def main():
    if len(sys.argv)!=2: raise SystemExit('usage: run_cpu_small.py /path/to/testu01-install')
    install=Path(sys.argv[1]).resolve()
    driver=r'''
#include "reference.hpp"
#include <iostream>
extern "C" {
#include "unif01.h"
#include "bbattery.h"
void fill(unsigned,const BenchParams*,uint64_t*,bool);
}
static BenchParams p;
static std::vector<uint64_t> output;
static uint64_t position;
static double next01() {
 if(position==p.words) {
  fill(12,&p,output.data(),false);
  p.base+=p.words;position=0;
 }
 return double(output[position++]>>11)*0x1p-53;
}
int main() {
 p=parameters(42,0,UINT64_C(2048)*4096,2048,65536,0);
 output.resize(p.words);position=p.words;
 char name[]="Montmory v3 spec_dual CPU emulation";
 auto gen=unif01_CreateExternGen01(name,next01);
 std::cout<<"CPU emulation only; seed=42 logical_lanes=2048 draws=4096 physical_threads=65536\n";
 bbattery_SmallCrush(gen);unif01_DeleteExternGen01(gen);
}
'''
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp);(temp/'metal.cpp').write_text(emulated_source());(temp/'driver.cpp').write_text(driver)
        cmd=['c++','-std=c++17','-O3','-Wall','-Wextra','-Werror','-I',str(ROOT/'src'),'-I',str(ROOT/'vendor/include'),
             '-I',str(install/'include'),str(temp/'metal.cpp'),str(temp/'driver.cpp'),'-L',str(install/'lib'),
             '-ltestu01','-lprobdist','-lmylib','-lm','-o',str(temp/'small')]
        subprocess.run(cmd,check=True)
        result=subprocess.run([str(temp/'small')],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    log=ROOT/'generated/cpu-smallcrush.log';log.write_text(result.stdout)
    passed='All tests were passed' in result.stdout
    historical=json.loads((ROOT/'reference/historical-smallcrush-pvalues.json').read_text())['printed_pvalues']
    pvalues=re.findall(r'^.*p-value.*:\s*(\S+)\s*$',result.stdout,re.M)
    assert pvalues==historical,('Historical printed p-values differ',pvalues,historical)
    report={'status':'pass' if passed else 'inspect TestU01 summary','scope':'SmallCrush of v3 spec_dual generated Metal adapted to C++ on x86; not native GPU',
            'seed':42,'logical_lanes':2048,'draws_per_lane':4096,'physical_threads':65536,
            'generated_source_sha256':hashlib.sha256((ROOT/'generated/all.metal').read_bytes()).hexdigest(),
            'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'historical_printed_pvalues_identical':len(pvalues),'Crush':'not run','BigCrush':'not run'}
    (ROOT/'generated/cpu-smallcrush.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print(result.stdout[-1800:])
    if not passed: raise SystemExit(3)
if __name__=='__main__': main()
