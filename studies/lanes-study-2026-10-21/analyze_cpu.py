#!/usr/bin/env python3
"""Validate the completed CPU campaign and regenerate descriptive summaries."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
P = json.loads((ROOT / 'protocol.json').read_text())
rows = list(csv.DictReader((RESULTS / 'cpu_measurements.csv').open()))
proof = json.loads((RESULTS / 'cpu_conformance.json').read_text())
fixed_phase = 'physical_workers_same_stream'
geometry_phase = 'logical_geometry_changes_stream'
expected = {(fixed_phase, 2048, w) for w in P['cpu_workers']}
expected |= {(geometry_phase, l, 4) for l in P['logical_geometries_lanes'] if l != 2048}
groups = defaultdict(list)
per_trial = defaultdict(dict)
for r in rows:
    phase, lanes, workers = r['phase'], int(r['logical_lanes']), int(r['requested_workers'])
    key = phase, lanes, workers
    trial = int(r['trial'])
    assert key in expected and 0 <= trial < P['repetitions']
    assert int(r['actual_workers']) == workers
    assert int(r['words']) == P['words_per_run']
    assert int(r['draws_per_lane']) == P['words_per_run'] // lanes
    assert key not in per_trial[trial], 'duplicate measurement'
    per_trial[trial][key] = r
    ns = float(r['ns_per_word'])
    assert math.isfinite(ns) and ns > 0
    assert math.isclose(ns, float(r['seconds']) * 1e9 / P['words_per_run'], rel_tol=1e-10)
    groups[key].append(ns)
assert len(per_trial) == P['repetitions']
for trial, trial_rows in per_trial.items():
    assert set(trial_rows) == expected
    assert len({r['seed'] for r in trial_rows.values()}) == 1
    if trial == 0:
        assert {r['seed'] for r in trial_rows.values()} == {'42'}
    # Checksums are an extra consistency check. The C++ program compared all words.
    assert len({r['checksum'] for key, r in trial_rows.items() if key[0] == fixed_phase}) == 1
assert proof['status'] == 'pass'
assert proof['fully_compared_words'] == len(rows) * P['words_per_run']
changes = list(csv.DictReader((RESULTS / 'cpu_geometry_changes.csv').open()))
assert len(changes) == P['repetitions'] * len(P['logical_geometries_lanes'])
change_keys = set()
for r in changes:
    trial, lanes = int(r['trial']), int(r['logical_lanes'])
    assert (trial, lanes) not in change_keys
    change_keys.add((trial, lanes))
    assert trial in per_trial and lanes in P['logical_geometries_lanes']
    assert r['seed'] == next(iter(per_trial[trial].values()))['seed']
    assert int(r['draws_per_lane']) == P['words_per_run'] // lanes
    count = int(r['different_words_from_T2048_D4096'])
    assert count == 0 if lanes == 2048 else 0 < count <= P['words_per_run']

def quantile(values, q):
    values = sorted(values)
    index = (len(values)-1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (index-lo) * (values[hi]-values[lo])

baseline = quantile(groups[(fixed_phase, 2048, 1)], .5)
summary = []
for (phase, lanes, workers), values in sorted(groups.items()):
    median = quantile(values, .5)
    summary.append(dict(phase=phase, logical_lanes=lanes,
        draws_per_lane=P['words_per_run']//lanes, workers=workers,
        repetitions=len(values), median_ns_per_word=median,
        q10_ns_per_word=quantile(values,.1), q90_ns_per_word=quantile(values,.9),
        stored_GB_per_second=8/median,
        speedup_vs_one_worker_same_stream=baseline/median if phase==fixed_phase else ''))
with (RESULTS/'cpu_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]))
    writer.writeheader();writer.writerows(summary)
report = dict(status='pass', measurements=len(rows), configurations=len(groups),
    fully_compared_words=proof['fully_compared_words'],
    fixed_stream_best_measured_workers=min((r for r in summary if r['phase']==fixed_phase),
        key=lambda r:r['median_ns_per_word'])['workers'],
    scope='Descriptive results on this shared x86 VPS; no M3 measurement or statistical quality claim.',
    quantiles='Linear empirical interpolation, not confidence intervals.',
    results=summary)
(RESULTS/'cpu_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
