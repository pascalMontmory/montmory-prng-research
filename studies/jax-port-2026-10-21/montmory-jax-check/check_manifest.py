"""Validate delivered sources and reference files using Python's standard library."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for name, expected in manifest["sha256"].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise SystemExit(f"Invalid manifest path: {name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Source fingerprint mismatch: {name}")
print(f"Empreintes des {len(manifest['sha256'])} fichiers du kit : PASS.")
