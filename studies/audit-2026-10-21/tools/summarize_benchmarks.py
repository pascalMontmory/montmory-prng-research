#!/usr/bin/env python3
"""Check recorded CPU benchmarks and summarize all engines, including losses."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'


def main():
    summaries=[]
    checksums={}
    for workload in ('bulk','sparse'):
        for compiler in ('native','portable'):
            name='benchmark-'+('sparse-' if workload=='sparse' else '')+compiler+'.csv'
            rows=list(csv.DictReader((E/name).open()))
            assert len(rows)==440,(name,len(rows))
            groups={}
            for r in rows:
                assert r['workers']==r['observed_workers']
                assert int(r['words'])==8388608
                key=(workload,r['engine'],r['seed'])
                value=(r['words'],r['checksum'])
                assert checksums.setdefault(key,value)==value,(name,key)
                g=(r['engine'],r['mode'],int(r['workers']))
                group=groups.setdefault(g,{})
                trial=int(r['trial'])
                assert trial not in group
                group[trial]=float(r['ns_per_word'])
            result=[]
            for (engine,mode,workers),values in sorted(groups.items()):
                assert sorted(values)==list(range(11))
                median=stats.median(values.values())
                mont=groups[('montmory_original',mode,workers)]
                ratios=[values[t]/mont[t] for t in values]
                result.append({'engine':engine,'mode':mode,'workers':workers,
                  'median_ns_per_u64':median,'million_u64_per_second':1000/median,
                  'minimum_ns_per_u64':min(values.values()),'maximum_ns_per_u64':max(values.values()),
                  'paired_montmory_speedup_median':stats.median(ratios),
                  'paired_montmory_speedup_minimum':min(ratios),
                  'paired_montmory_speedup_maximum':max(ratios),
                  'montmory_faster_trials':sum(v>1 for v in ratios),'trials':11})
            summaries.append({'workload':workload,'compiler':compiler,'file':name,
                              'sha256':hashlib.sha256((E/name).read_bytes()).hexdigest(),
                              'measurements':len(rows),'results':result})
    (E/'benchmark-summary.json').write_text(json.dumps({'status':'pass: complete trials, actual worker counts and cross-mode/cross-compiler checksums verified',
      'scope':'CPU x86 only; bulk and isolated-index workloads are separate', 'runs':summaries},indent=2)+'\n')
    for s in summaries:
        print(s['workload'],s['compiler'])
        for r in sorted(s['results'],key=lambda r:r['median_ns_per_u64']):
            if r['mode']=='store64' and r['workers']==1:
                print('  %-22s %.3f ns/u64  paired Montmory speedup %.3fx (%d/11)'%
                  (r['engine'],r['median_ns_per_u64'],r['paired_montmory_speedup_median'],r['montmory_faster_trials']))


if __name__=='__main__': main()
