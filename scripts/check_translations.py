#!/usr/bin/env python3
"""Check the FR/EN reading editions, local links and unchanged notebook code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)\)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prose(text):
    """Remove fenced code before checking rendered Markdown links/headings."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            token = match[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return '\n'.join(lines)


def markdown(path):
    if path.suffix == '.ipynb':
        notebook = json.loads(path.read_text())
        return '\n\n'.join(''.join(c['source']) for c in notebook['cells']
                           if c['cell_type'] == 'markdown')
    return path.read_text()


def anchors(text):
    """GitHub-style anchors for headings used in the curated documentation."""
    seen = {}
    result = set()
    for line in prose(text).splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        title = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', match[1])
        slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        count = seen.get(slug, 0)
        result.add(f'{slug}-{count}' if count else slug)
        seen[slug] = count + 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/translation-check.json')
    args = parser.parse_args()
    registry = json.loads((ROOT / 'docs/translations.json').read_text())
    require(registry['schema_version'] == 1, 'Unsupported translation registry')
    require(not (ROOT / 'legal').exists(), 'Private legal directory is forbidden in the public edition')
    paths = set()
    documents = []
    notebook_pairs = code_cells = 0
    private_pairs = frozen_references = 0
    for pair in registry['pairs']:
        for lang in ('fr', 'en'):
            rel = pair[lang]
            require(rel not in paths, f'Duplicate translation path: {rel}')
            paths.add(rel)
            path = (ROOT / rel).resolve()
            require(path.is_relative_to(ROOT) and path.is_file(), f'Missing file: {rel}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == pair[f'{lang}_sha256'],
                    f'{rel}: review both language editions and update their registry hashes')
            documents.append(path)
            frozen = pair['reference_frozen'] and lang == pair['reference_language']
            if frozen:
                frozen_references += 1
            if pair['kind'] != 'notebook' and not frozen:
                other = pair['en' if lang == 'fr' else 'fr']
                target = os.path.relpath(ROOT / other, path.parent)
                require(target in LINK.findall(prose(path.read_text())),
                        f'{rel}: missing language-switch link to {other}')
        if pair['fr'].startswith('legal/'):
            private_pairs += 1
            english = (ROOT / pair['en']).read_text()
            require('PRIVATE — reading translation only' in english,
                    f"{pair['en']}: missing private translation notice")
        if pair['kind'] == 'notebook':
            notebook_pairs += 1
            original = json.loads((ROOT / pair[pair['reference_language']]).read_text())
            translated = json.loads((ROOT / pair['en']).read_text())
            require(len(original['cells']) == len(translated['cells']),
                    f"{pair['en']}: cell count changed")
            require({k: v for k, v in original.items() if k != 'cells'} ==
                    {k: v for k, v in translated.items() if k != 'cells'},
                    f"{pair['en']}: notebook metadata changed")
            for i, (a, b) in enumerate(zip(original['cells'], translated['cells'])):
                require(a['cell_type'] == b['cell_type'], f"{pair['en']}: cell {i} changed type")
                if a['cell_type'] == 'markdown':
                    require({k: v for k, v in a.items() if k != 'source'} ==
                            {k: v for k, v in b.items() if k != 'source'},
                            f"{pair['en']}: cell {i} metadata changed")
                    require(a['source'] != b['source'],
                            f"{pair['en']}: explanatory cell {i} not translated")
                else:
                    require(a == b, f"{pair['en']}: executable cell {i} changed")
                    code_cells += a['cell_type'] == 'code'
    # Every current reading page needs an entry; archived studies and sealed
    # historical v1 documents intentionally remain outside the translation scope.
    expected = [*ROOT.glob('*.md'), *ROOT.joinpath('docs').rglob('*.md'),
                *ROOT.joinpath('notebooks').glob('*.md'),
                *ROOT.joinpath('notebooks').glob('*.ipynb'),
                ROOT / 'LICENSE',
                *ROOT.joinpath('applications/jax_rng/validation').glob('*.md')]
    for path in expected:
        require(path.relative_to(ROOT).as_posix() in paths,
                f'{path.relative_to(ROOT)}: current page missing from translation registry')
    local_links = anchor_links = 0
    for document in documents:
        for target in LINK.findall(prose(markdown(document))):
            url = urlsplit(target.strip('<>'))
            if url.scheme or url.netloc:
                continue
            destination = (document.parent / unquote(url.path)).resolve() if url.path else document
            require(destination.is_relative_to(ROOT) and destination.exists(),
                    f'{document.relative_to(ROOT)}: broken local link {target}')
            local_links += 1
            if url.fragment and destination.suffix in {'.md', '.ipynb'}:
                require(unquote(url.fragment) in anchors(markdown(destination)),
                        f'{document.relative_to(ROOT)}: missing heading for {target}')
                anchor_links += 1
    # This existing report identifies the French application protocol by hash.
    # A reading translation must not rewrite that original evidence.
    report = json.loads((ROOT / 'applications/jax_rng/validation/cpu-smoke/report.json').read_text())
    for rel, digest in report['source_sha256'].items():
        require(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest,
                f'{rel}: original application-validation source changed')
    result = {'status': 'pass', 'language_pairs': len(registry['pairs']),
              'files_checked': len(paths), 'local_links': local_links,
              'heading_links': anchor_links, 'notebook_pairs': notebook_pairs,
              'identical_executable_cells': code_cells,
              'private_legal_pairs': private_pairs, 'frozen_references': frozen_references,
              'scope': 'reading editions, hashes, links and notebook equivalence; '
                       'no linguistic certification, native run or statistical test'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
