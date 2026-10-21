#!/usr/bin/env python3
"""Verify imported bytes, curated links, evidence levels and recorded data counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'build/integrity-check.json')
    args = p.parse_args()
    manifest = read(ROOT / 'evidence/import-manifest.json')
    paths = set()
    for entry in manifest['files']:
        assert entry['path'] not in paths, entry['path']
        paths.add(entry['path'])
        path = ROOT / entry['path']
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'], path
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'], path
    levels = {'raw-audited', 'historical-log', 'user-terminal', 'cpu-execution', 'exact-certificate'}
    index = read(ROOT / 'evidence/index.json')
    for entry in index['records']:
        assert entry['level'] in levels, entry
        for rel in entry['files']:
            assert (ROOT / rel).is_file(), rel
    link_count = 0
    curated = [*ROOT.glob('*.md'), *sorted((ROOT / 'docs').rglob('*.md'))]
    for doc in curated:
        # Curated documents use ordinary inline Markdown links with path targets.
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', doc.read_text()):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            local = unquote(target.split('#', 1)[0])
            assert (doc.parent / local).exists(), f'{doc.relative_to(ROOT)}: {target}'
            link_count += 1
    kit_manifests = 0
    for file in (ROOT / 'studies').rglob('manifest.json'):
        data = read(file)
        if not isinstance(data, dict) or not isinstance(data.get('sha256'), dict):
            continue
        for rel, digest in data['sha256'].items():
            assert hashlib.sha256((file.parent / rel).read_bytes()).hexdigest() == digest, (file, rel)
        kit_manifests += 1
    complete_text_manifests = 0
    # The root audit and Certified historical manifests refer to excluded
    # binaries or paths outside this assembled record; check runnable kits here.
    for file in (ROOT / 'studies').rglob('SHA256SUMS'):
        if file.parent.name == 'evidence':
            continue
        for line in file.read_text().splitlines():
            match = re.match(r'^([0-9a-f]{64})\s+[* ]?(.+)$', line)
            if match:
                digest, rel = match.groups()
                assert hashlib.sha256((file.parent / rel).read_bytes()).hexdigest() == digest, (file, rel)
        complete_text_manifests += 1
    tpu = ROOT / 'studies/tpu-opt-2026-10-21'
    csv_path = tpu / '20261021T185102Z-tpu-opt/campaign/measurements.csv'
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3708
    assert sum(r['phase'] == 'search' for r in rows) == 936
    assert sum(r['phase'] == 'confirmation' for r in rows) == 2772
    audit = read(tpu / 'results/audit-20261021T185102Z-tpu-opt/audit.json')
    assert audit['samples_checked'] == 3708 and audit['summaries_recomputed'] == 44
    assert audit['decisions']['8388608']['retained'] == 'pallas_carry_128'
    assert audit['decisions']['16777216']['retained'] == 'pallas_carry_64'
    tacm = read(ROOT / 'studies/tacm-comparison-2026-10-21/results/summary.json')
    assert tacm['primary_comparisons'] == 54 and tacm['advantages'] == 0
    assert tacm['calibration_pilot_gate_pass'] is False
    assert tacm['calibration_extension_gate_pass'] is True
    for file in (ROOT / 'notebooks').glob('*.ipynb'):
        data = read(file)
        assert data['nbformat'] == 4
        for i, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'{file.name}:cell{i}', 'exec')
    result = {'status': 'pass', 'imported_files': len(paths), 'curated_local_links': link_count,
              'evidence_records': len(index['records']), 'intact_json_kit_manifests': kit_manifests,
              'intact_text_kit_manifests': complete_text_manifests,
              'tpu_raw_rows': len(rows), 'tpu_summary_rows': 44,
              'scope': 'integrity and documented data consistency; no new native execution'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
