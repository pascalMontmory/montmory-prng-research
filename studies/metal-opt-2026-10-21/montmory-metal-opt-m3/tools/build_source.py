#!/usr/bin/env python3
"""Reconstruct the candidate Metal library without changing the v3 base."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
(root/'generated/all.metal').write_bytes((root/'generated/v3.metal').read_bytes()+b'\n'+(root/'src/optimizations.metal').read_bytes())
