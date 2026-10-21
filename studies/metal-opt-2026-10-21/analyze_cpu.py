#!/usr/bin/env python3
"""Recompute the held-out CPU comparison from archived raw observations."""
import csv
import json
import statistics
from pathlib import Path
root=Path(__file__).resolve().parent/'results'
rows=list(csv.DictReader((root/'cpu-confirmation.csv').open()))
assert len(rows)==20 and all(r['full_comparison_pass']=='true' for r in rows)
values={(int(r['trial']),r['engine'],int(r['workers'])):float(r['ns_per_word']) for r in rows}
assert len(values)==20
report={}
comparisons=[('one_worker',('indexed',1),('incremental_pair',1)),
             ('selected_workers',('indexed',16),('incremental_pair',8))]
for name,baseline,candidate in comparisons:
    a=[values[(i,*baseline)] for i in range(5)]
    b=[values[(i,*candidate)] for i in range(5)]
    ratios=[x/y for x,y in zip(a,b)]
    report[name]=dict(baseline=baseline,candidate=candidate,
        baseline_median_ns=statistics.median(a),candidate_median_ns=statistics.median(b),
        baseline_GBps=8/statistics.median(a),candidate_GBps=8/statistics.median(b),
        median_paired_speedup=statistics.median(ratios),faster_seeds=sum(x>1 for x in ratios),
        paired_speedups=ratios)
training=list(csv.DictReader((root/'cpu-measurements.csv').open()))
assert len(training)==180 and all(r['full_comparison_pass']=='true' for r in training)
assert len({(r['trial'],r['engine'],r['workers']) for r in training})==180
assert json.loads((root/'cpu-conformance.json').read_text())['fully_compared_words']==3019898880
report['verification']=dict(search_timed_runs=180,search_compared_words_including_warmup=3019898880,
    confirmation_runs=20,confirmation_compared_words=167772160,native_Metal='pending')
(root/'cpu-confirmation-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
