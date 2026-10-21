# Validation of the new benchmark

[Français](README.md) · [English](README.en.md) · [Documentation FR/EN](../../../docs/en/languages.md)

The [CPU report](cpu-smoke/report.json) and [44 measurements](cpu-smoke/measurements.csv) come from a successful local VPS run: one volume, two seeds and two repetitions. The program compared 393,216 integer words against their family references and checked all 44 timed outputs. Montmory's integer reference is the C code; third-party APIs serve as their family references.

This short check validates execution, integer conformance and application outputs. It does not measure Pallas kernels on TPU, rerun a statistical battery or establish a commercial gain. Reference files remain in unchanged archives.

The [additional check](check-report.json) verified uniform endpoints, rejection of three deliberately corrupted outputs, 18 application cases with Pallas interpreted on CPU and paired-ratio calculation. Interpretation is not native TPU execution. Rerun it from the repository root using the pinned JAX interpreter:

```bash
.venv-jax/bin/python applications/jax_rng/check.py
```

Hashes in the report identify the sources actually executed. A base commit with `git_dirty=true` means the new benchmark was still being added; hashes then identify its exact state.
