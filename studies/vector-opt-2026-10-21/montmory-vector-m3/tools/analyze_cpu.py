#!/usr/bin/env python3
"""Recompute paired CPU comparisons, preserving every observation."""
import csv,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path
MASK=(1<<64)-1
def seed(n):
    n=(n+0x9E3779B97F4A7C15)&MASK
    n=((n^(n>>30))*0xBF58476D1CE4E5B9)&MASK
    n=((n^(n>>27))*0x94D049BB133111EB)&MASK
    return n^(n>>31)
def main():
    directory=Path(sys.argv[1]);groups=defaultdict(dict);timed_words=0
    for row in csv.DictReader((directory/'cpu-measurements.csv').open()):
        trial=int(row['trial']);n=int(row['words']);t=float(row['seconds'])
        assert 0<=trial<9 and int(row['seed'])==seed(3000+trial)
        assert int(row['base'])==(trial%2)*8388608 and n in (65536,8388608)
        assert row['full_comparison_pass']=='true' and math.isfinite(t) and t>0
        assert math.isclose(float(row['ns_per_word']),t*1e9/n,rel_tol=1e-12)
        key=(n,row['engine'],int(row['workers']))
        assert trial not in groups[key];groups[key][trial]=t;timed_words+=n
    summaries=[]
    for (n,e,w),values in sorted(groups.items()):
        assert set(values)==set(range(9));ref=groups[(n,'scalar_pair',w)]
        ratios=[ref[i]/values[i] for i in range(9)]
        summaries.append(dict(words=n,engine=e,workers=w,GB_per_second=n*8/statistics.median(values.values())/1e9,
            median_paired_ratio_vs_pair=statistics.median(ratios),wins=sum(values[i]<ref[i] for i in range(9))))
    reported=list(csv.DictReader((directory/'cpu-summary.csv').open()))
    assert len(reported)==len(summaries)
    for r in reported:
        a=next(a for a in summaries if (a['words'],a['engine'],a['workers'])==(int(r['words']),r['engine'],int(r['workers'])))
        for field in ('GB_per_second','median_paired_ratio_vs_pair'):
            assert math.isclose(float(r[field]),a[field],rel_tol=1e-12)
        assert int(r['seeds_faster'])==a['wins']
    proof=json.loads((directory/'cpu-conformance.json').read_text())
    assert proof['status']=='pass' and proof['checked_words']==timed_words*2
    pools=defaultdict(list);seen=set()
    for r in csv.DictReader((directory/'pool-confirmation.csv').open()):
        trial=int(r['trial']);rnd=int(r['round']);repeat=int(r['repeat']);e=r['engine'];t=float(r['seconds'])
        assert 0<=trial<7 and 0<=rnd<3 and 0<=repeat<5 and int(r['seed'])==seed(6000+trial)
        assert int(r['words'])==65536 and r['full_comparison_pass']=='true' and math.isfinite(t) and t>0
        k=(trial,rnd,repeat,e);assert k not in seen;seen.add(k);pools[e,trial].append(t)
    assert len(seen)==315
    pool_summary=[]
    for ref in ('sequential_pair','spawn4_pair'):
        ratios=[];per_seed=[]
        for trial in range(7):
            assert len(pools[ref,trial])==len(pools['pool4_pair',trial])==15
            a=statistics.median(pools[ref,trial]);b=statistics.median(pools['pool4_pair',trial])
            ratios.append(a/b);per_seed.append(dict(seed=seed(6000+trial),reference_seconds=a,pool_seconds=b,ratio=a/b))
        pool_summary.append(dict(reference=ref,median_paired_speedup=statistics.median(ratios),wins=sum(x>1 for x in ratios),
            seeds=7,per_seed=per_seed))
    result=dict(status='pass',cpu_checked_words=proof['checked_words'],cpu_results=summaries,pool_confirmation=pool_summary,
        scope='Observed CPU timings, no exclusions; no GPU or statistical certification')
    (directory/'cpu-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for a in pool_summary:
        print(f"Pool 4 / {a['reference']}: ratio apparie median {a['median_paired_speedup']:.3f}, {a['wins']}/7 seeds plus rapides (65536 mots).")
if __name__=='__main__':main()
