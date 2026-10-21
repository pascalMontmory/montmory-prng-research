#!/usr/bin/env python3
"""Recompute archived Mac summaries and inspect first-use effects; no row removal."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
RUN = '20261021T162407Z-22542'
DATA = ROOT / 'received' / ('mac-' + RUN) / 'results' / RUN
KIT = ROOT / 'montmory-metal-opt-m3'
OUT = ROOT / 'results' / ('mac-analysis-' + RUN)
SPECIAL = {'montmory_const_pair', 'montmory_const_stripe_pair', 'montmory_const_stripe_quad'}


def read_json(name):
    return json.loads((DATA / name).read_text())


def read_csv(name):
    with (DATA / name).open() as source:
        return list(csv.DictReader(source))


def quantile(values, fraction):
    values = sorted(values)
    i = fraction * (len(values) - 1)
    a = int(i)
    b = min(a + 1, len(values) - 1)
    return values[a] + (i - a) * (values[b] - values[a])


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-12), (a, b)


def splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def group_key(row):
    return row['engine'], row['format'], int(row['physical_threads']), int(row['threadgroup'])


def family(row):
    return row['engine'], row['format']


def paired(reference, candidate):
    assert [r['seed'] for r in reference] == [r['seed'] for r in candidate]
    ratios = [float(a['ns_per_word']) / float(b['ns_per_word']) for a, b in zip(reference, candidate)]
    return {'median_paired_speedup': statistics.median(ratios),
            'min_paired_speedup': min(ratios), 'max_paired_speedup': max(ratios),
            'wins': sum(x > 1 for x in ratios), 'trials': len(ratios), 'paired_ratios': ratios}


def main():
    OUT.mkdir(exist_ok=True)
    archived_manifest = (DATA / 'kit-SHA256SUMS.txt').read_bytes()
    assert archived_manifest == (KIT / 'SHA256SUMS').read_bytes()
    manifest_count = 0
    for line in archived_manifest.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        assert hashlib.sha256((KIT / name).read_bytes()).hexdigest() == digest, name
        manifest_count += 1
    assert (DATA / 'golden-vectors.txt').read_bytes() == (KIT / 'reference/golden-vectors.txt').read_bytes()
    assert (DATA / 'protocol.json').read_bytes() == (KIT / 'protocol.json').read_bytes()
    assert (DATA / 'linux-verification.json').read_bytes() == (KIT / 'generated/optimization-verification.json').read_bytes()
    assert (DATA / 'exit-code.txt').read_text().strip() == '0'
    assert all(line.endswith(': OK') for line in (DATA / 'integrity.txt').read_text().splitlines())
    assert len((DATA / 'integrity.txt').read_text().splitlines()) == manifest_count
    protocol = read_json('protocol.json')
    rows = read_csv('measurements.csv')
    search = collections.defaultdict(list)
    confirmation = collections.defaultdict(list)
    seen = set()
    for ordinal, r in enumerate(rows):
        key = group_key(r)
        phase, trial = r['phase'], int(r['trial'])
        identity = phase, trial, key
        assert identity not in seen
        seen.add(identity)
        expected_seed = splitmix64(1000 + trial) if phase == 'confirmation' else (42 if trial == 0 else splitmix64(trial))
        assert int(r['seed']) == expected_seed
        assert int(r['words']) == 8388608
        assert int(r['logical_lanes']) == 2048 and int(r['draws_per_lane']) == 4096
        assert r['full_comparison_pass'] == 'true'
        assert float(r['wall_seconds']) >= float(r['gpu_seconds']) > 0
        close(r['ns_per_word'], float(r['gpu_seconds']) * 1e9 / int(r['words']))
        close(r['stored_GB_per_second'], 8 / float(r['ns_per_word']))
        r['_ordinal'] = ordinal
        (search if phase == 'search' else confirmation)[key].append(r)
    expected_keys = {(e, f, w, g) for e in protocol['engine_names'].values()
                     for f in protocol['formats'] for w in protocol['workers'] for g in protocol['groups']}
    assert set(search) == expected_keys and len(rows) == 2574
    for group in search.values():
        assert [int(r['trial']) for r in group] == list(range(7))
    selected = {family(r): group_key(r) for r in read_json('selection-before-confirmation.json')['selected']}
    assert set(confirmation) == set(selected.values()) and len(selected) == 22
    for f, key in selected.items():
        minimum = min(statistics.median(float(r['ns_per_word']) for r in v)
                      for k, v in search.items() if k[:2] == f)
        close(statistics.median(float(r['ns_per_word']) for r in search[key]), minimum)
    reported = {family(r): r for r in read_json('summary.json')['results']}
    recomputed = []
    for f, key in sorted(selected.items()):
        group = confirmation[key]
        assert [int(r['trial']) for r in group] == list(range(5))
        ns = [float(r['ns_per_word']) for r in group]
        item = {'engine': f[0], 'format': f[1], 'threads': key[2], 'group': key[3],
                'median_ns_per_word': statistics.median(ns),
                'stored_GB_per_second': 8 / statistics.median(ns),
                'q10_ns_per_word': quantile(ns, .1), 'q90_ns_per_word': quantile(ns, .9),
                'gpu_GB_per_second_per_trial': [float(r['stored_GB_per_second']) for r in group],
                'wall_ns_per_word_per_trial': [float(r['wall_seconds']) * 1e9 / 8388608 for r in group]}
        item.update(paired(confirmation[selected['montmory_mulhi_pair', f[1]]], group))
        for field in ('median_ns_per_word', 'stored_GB_per_second', 'q10_ns_per_word', 'q90_ns_per_word'):
            close(item[field], reported[f][field])
        close(item['median_paired_speedup'], reported[f]['median_paired_speedup_vs_v3'])
        assert item['wins'] == reported[f]['seeds_faster_than_v3']
        recomputed.append(item)
    assert read_json('metadata.json')['fully_compared_words'] == len(rows) * 8388608 * 2

    cpu = read_csv('cpu-measurements.csv')
    cpu_groups = collections.defaultdict(list)
    for r in cpu:
        assert r['full_comparison_pass'] == 'true'
        assert int(r['seed']) == (42 if int(r['trial']) == 0 else splitmix64(int(r['trial'])))
        close(r['ns_per_word'], float(r['seconds']) * 1e9 / int(r['words']))
        cpu_groups[r['engine'], int(r['workers'])].append(r)
    assert len(cpu) == 144 and len(cpu_groups) == 16
    cpu_reported = {(r['engine'], int(r['workers'])): r for r in read_csv('cpu-summary.csv')}
    cpu_recomputed = []
    for key, group in sorted(cpu_groups.items()):
        assert [int(r['trial']) for r in group] == list(range(9))
        ns = [float(r['ns_per_word']) for r in group]
        baseline = cpu_groups['indexed', key[1]]
        med = statistics.median(ns)
        item = {'engine': key[0], 'workers': key[1], 'median_ns_per_word': med,
                'stored_GB_per_second': 8 / med,
                'ratio_of_medians_vs_indexed': statistics.median(float(r['ns_per_word']) for r in baseline) / med}
        item.update(paired(baseline, group))
        close(item['stored_GB_per_second'], cpu_reported[key]['stored_GB_per_second'])
        close(item['ratio_of_medians_vs_indexed'], cpu_reported[key]['speedup_vs_indexed_same_workers'])
        close(quantile(ns, .1), cpu_reported[key]['q10'])
        close(quantile(ns, .9), cpu_reported[key]['q90'])
        cpu_recomputed.append(item)
    assert read_json('cpu-conformance.json')['fully_compared_words'] == len(cpu) * 8388608 * 2

    # Exploratory diagnosis only: these rows are not removed from any summary.
    preparation = read_json('pipeline-preparation.json')
    first_uses = []
    for trial in range(5):
        for engine in sorted(SPECIAL):
            group = [r for r in rows if r['phase'] == 'confirmation' and int(r['trial']) == trial and r['engine'] == engine]
            assert len(group) == 2
            first, second = group
            entries = [p for p in preparation if p['engine'] == engine and p['seed_first_use'] == int(first['seed'])]
            assert len(entries) == 1 and entries[0]['specialized_rotations']
            first_uses.append({'trial': trial, 'seed': first['seed'], 'engine': engine,
                'first_format': first['format'], 'first_group': int(first['threadgroup']),
                'second_format': second['format'], 'second_group': int(second['threadgroup']),
                'first_GB_per_second': float(first['stored_GB_per_second']),
                'second_GB_per_second': float(second['stored_GB_per_second']),
                'second_over_first_throughput': float(first['ns_per_word']) / float(second['ns_per_word']),
                'pipeline_preparation_seconds': entries[0]['preparation_seconds'],
                'rotations': [entries[0]['r0'], entries[0]['r1'], entries[0]['r2']]})
    with (OUT / 'specialized-first-use.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(first_uses[0]))
        writer.writeheader()
        writer.writerows(first_uses)

    comparisons = {}
    for fmt in ('raw64', 'mantissa53'):
        reference = confirmation[selected['montmory_split_stripe_pair', fmt]]
        for engine in ('philox4x32_10', 'threefry2x32_20'):
            comparisons[engine + '/' + fmt] = paired(reference, confirmation[selected[engine, fmt]])
    archive = ROOT.parent / ('montmory-metal-opt-results-' + RUN + '.tar.gz')
    report = {'status': 'archive_consistent_summaries_recomputed',
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'manifest_files_matched_to_delivered_kit': manifest_count,
        'mac_reported_conformance': read_json('conformance.json'),
        'mac_cpu_fully_compared_words': read_json('cpu-conformance.json')['fully_compared_words'],
        'mac_gpu_campaign_fully_compared_words': read_json('metadata.json')['fully_compared_words'],
        'measurement_rows': {'cpu': len(cpu), 'gpu_search': sum(map(len, search.values())), 'gpu_confirmation': sum(map(len, confirmation.values()))},
        'thermal_states_reported': dict(collections.Counter(r['thermal_state'] for r in rows)),
        'cpu': cpu_recomputed, 'gpu_confirmation': recomputed,
        'competitors_vs_split_counter_descriptive': comparisons,
        'specialized_first_use': first_uses,
        'first_use_summary': {'pipelines': 15,
            'first_below_40_GB_per_second': sum(r['first_GB_per_second'] < 40 for r in first_uses),
            'second_below_40_GB_per_second': sum(r['second_GB_per_second'] < 40 for r in first_uses),
            'median_second_over_first_throughput': statistics.median(r['second_over_first_throughput'] for r in first_uses)},
        'limitations': [
            'Archive records native execution supplied by the user; analysis does not independently re-execute Metal.',
            'One timed dispatch per job and seed, after two warm-up dispatches. Search/confirmation ordering is recorded in CSV.',
            'Specialized pipelines are compiled lazily inside measured campaign; compilation wall time is excluded, but post-compilation transients were not controlled.',
            'First/second use comparison also changes format and sometimes threadgroup; association is not a causal isolation.',
            'Thermal state 0 throughout does not measure frequency, scheduler contention or GPU utilization.',
            'All rows retained. No statistical superiority or steady-state hardware optimum claimed from these five seeds.',
            'Full BigCrush archive has not been received in this directory; its reported PASS is separate evidence.'
        ]}
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    for engine, label in [('indexed', 'Indexé'), ('incremental_pair', 'Par paires')]:
        rr = sorted((r for r in cpu_recomputed if r['engine'] == engine), key=lambda r: r['workers'])
        axes[0].plot([r['workers'] for r in rr], [r['stored_GB_per_second'] for r in rr], 'o-', label=label)
    axes[0].set(title='CPU M3 — médianes, 9 seeds', xlabel='Workers', ylabel='Go/s stockés (raw64)', xticks=[1, 2, 4, 8])
    axes[0].legend()
    engines = ['montmory_mulhi_pair', 'montmory_split_stripe_pair', 'philox4x32_10', 'threefry2x32_20']
    for i, engine in enumerate(engines):
        r = next(r for r in recomputed if r['engine'] == engine and r['format'] == 'raw64')
        axes[1].scatter([i + (j - 2) * .045 for j in range(5)], r['gpu_GB_per_second_per_trial'], s=28)
        axes[1].plot([i - .2, i + .2], [r['stored_GB_per_second']] * 2, color='black', linewidth=2)
    axes[1].set(title='Metal raw64 — 5 seeds de confirmation', ylabel='Go/s stockés', xticks=range(4), xticklabels=['V3', 'Compteur\nscindé', 'Philox', 'Threefry'], ylim=(0, 110))
    for r in first_uses:
        axes[2].plot([1, 2], [r['first_GB_per_second'], r['second_GB_per_second']], 'o-', alpha=.55)
    axes[2].set(title='15 pipelines spécialisés — confirmation', ylabel='Go/s stockés', xticks=[1, 2], xticklabels=['Première\nmesure', 'Seconde\nmesure'], ylim=(0, 110))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Montmory — données natives M3 reçues ; toutes les mesures conservées', fontsize=14)
    fig.savefig(OUT / 'mac-analysis.png', dpi=180)
    fig.savefig(OUT / 'mac-analysis.svg')
    plt.close(fig)
    print(json.dumps({'status': report['status'], 'counts': report['measurement_rows'], 'first_use': report['first_use_summary'], 'output': str(OUT)}))


if __name__ == '__main__':
    main()
