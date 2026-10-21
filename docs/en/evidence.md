# Evidence register and traceability

[Français](../evidence.md) · [English](evidence.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Machine-readable index](../../evidence/index.json) · [Imported hashes](../../evidence/import-manifest.json)

## Evidence levels

| Level | Definition |
|---|---|
| **Audited raw data** | Received CSV/JSON or logs, structural checks and recomputed conclusions using frozen sources |
| **Historical log** | A complete available log whose historical execution was not repeated in the current session |
| **Reported terminal summary** | Values copied from the user's terminal; individual data unavailable for independent recalculation |
| **CPU execution / emulation** | Local checking of values, a protocol or a transcription; not native GPU/TPU execution |
| **Exact identity / certificate** | Integer calculation establishing a specific property or counterexample; scope limited to that property |

Here, “audited” means reviewed and recalculated by a separate script within the same project. It does not mean independent certification by an external organization.

## Where to find each result

| Topic | Main location | What is actually present |
|---|---|---|
| Engine and exact certificates | [Audit](../../studies/audit-2026-10-21/MATHEMATIQUES.md) | Sources, certificates, vectors, identities and counterexamples |
| TestU01 | [Dedicated register](testu01.md) | Complete historical logs, CPU replays and separately identified recent excerpts |
| CPU Random123 comparisons | [Data](../../studies/audit-2026-10-21/evidence/benchmark-native.csv) | Measurements, compilation, known vectors and frozen implementations |
| Corrected TACM | [Study](../../studies/tacm-comparison-2026-10-21/README.md) | Protocol, calibration pilot, extension, raw estimates and analysis |
| CPU workers and lanes | [Study](../../studies/lanes-study-2026-10-21/README.md) | Local x86 measurements and stream-preservation checks |
| Initial M3 optimization | [Extracted archive](../../studies/metal-opt-2026-10-21/received) | Native measurements and audit; first-use effects identified |
| Separate M3 confirmation | [Summary](../../studies/metal-opt-2026-10-21/results/mac-confirm-reported-20261021T163954Z-22644.json) | Reported terminal output; raw archive missing |
| SIMD and worker pool | [Study](../../studies/vector-opt-2026-10-21/README.md) | Local x86 data; ARM64/Metal results reported through the terminal |
| JAX CPU port | [Study](../../studies/jax-port-2026-10-21/README.md) | Portable integer port and CPU checks |
| First TPU campaign | [Data](../../studies/jax-tpu-2026-10-21/20261021T180635Z-colab-tpu) | 770 observations, native reports, CPU audit |
| Final TPU optimization | [Data](../../studies/tpu-opt-2026-10-21/20261021T185102Z-tpu-opt) | 3,708 observations, 104 compilations, frozen selection, 44 recomputed summaries |

## Native conformance matrix

| Platform | Documented checks | Limitation |
|---|---|---|
| x86 CPU | C reference, 1,600 vectors, certificates, boundaries, partitions, comparisons and reconstruction of TPU hashes | Results apply to the recorded machines and compilers |
| ARM64 Apple M3 | 1,600 ARM64/x86 vectors and equivalent variants; first optimization archive received | Some later confirmations are available only through the terminal |
| M3 Metal GPU | Millions of native comparisons against CPU reference; checked timed outputs | M3 tested; no automatic generalization to all Apple Silicon |
| Historical CUDA GPU | Recovered sources and logs, transcriptions checked on CPU | Corrected version not run on an NVIDIA GPU in the recent audit |
| TPU v5 lite | 1,972,720 historical comparisons, 1,684,047 candidate comparisons including Threefry, timed-output hashes | One native device; no native multi-TPU evidence |

This matrix does not claim that every newly written variant ran natively on every architecture. The common reference links the campaigns; the actual ports and platforms tested remain identified.

## Preservation

The **884 imported files** are copied unchanged and recorded by size, relative origin and SHA256 in `import-manifest.json`. This repository's documentation and scripts are separate from those imports. Old archives containing additional files are not artificially reconstructed.

Caches, virtual environments, received compiled libraries, executables and redundant archives are excluded. Sources needed by current audits are present. The [exclusion register](../../evidence/excluded-files.json) retains the rule and affected paths; it does not claim to provide a complete copy of every old bundle.

Some historical bundle manifests therefore mention absent binaries. The entry points in `scripts/` are the checks supported for this repository. The frozen TPU kits, however, retain all files listed in their manifests.

The [initial received M3 archive](../../evidence/archives/montmory-metal-opt-results-20261021T162407Z-22542.tar.gz) is also retained: the historical audit requires its hash. Audits load no received binary; required oracles are rebuilt from source.

## Evidence still missing

The complete archive of the latest native M3 v3 BigCrush, native archives for the separate Metal confirmation and vector repair, and an independent new TPU session are absent. Their corresponding summaries remain explicitly labeled. These gaps do not prevent replaying audits for campaigns whose raw measurements are available.
