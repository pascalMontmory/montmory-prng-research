# Start here: evaluate the CPU and TPU implementations

[Français](start-here.fr.md) · [English](start-here.md) · [Documentation FR/EN](en/languages.md)

[Research overview](../README.md) · [Reproduction commands](en/reproduce.md) · [Measured results](en/results.md)

This page is for an engineering team reviewing the code. The repository contains two TPU implementations: **Montmory**, preserving the recorded Montmory stream, and **Threefry2x32 at 20 rounds**, implemented in Pallas as a strong comparator against the JAX API. They are different generators. This is a research repository with frozen executable kits, not an installed SDK or a completed JAX PRNG plugin.

This research record is publicly readable. Publication does not activate a general license for original contributions; obtain an appropriate authorization for uses requiring one. Third-party permissions retain their own scope. See [rights status](../LICENSE).

## First check: no cloud account or accelerator required

From a machine with Python 3.11+ and a C compiler supporting UBSan (`cc`, GCC or Clang):

```bash
git clone https://github.com/pascalMontmory/montmory-prng-research.git
cd montmory-prng-research
python3 scripts/check_repository.py
python3 scripts/check_reference.py
```

Both commands should finish successfully and print `"status": "pass"`. The first checks the frozen imported files and evidence structure. The second compiles the C oracle, checks **1,600 vectors** against C and Python, and reproduces the exact mathematical certificates, including the documented limitations. Fresh reports go to `build/`. They are local CPU checks, not a new TestU01 or TPU run.

On macOS, install the Xcode command-line tools if no compiler is available. If a dependency installation fails, use the separate environments documented in [reproduce.md](en/reproduce.md); the old TACM analysis environment and the JAX environment have different pins.

## Find the implementation you need

| Objective | File to read | What to look for |
|---|---|---|
| Understand the scalar Montmory algorithm | [mmc_core.c](../src/reference/mmc_core.c), [header](../src/reference/mmc_core.h), [integer Python model](../src/reference/reference.py) | State increment, three Feistel rounds, final mixer; the reference fixes semantics. |
| Understand the global output sequence | [Stream contract](en/contract.md) | `(engine version, seed, global index, T, D, format)`; logical lanes affect the stream. |
| Understand the portable JAX arithmetic | [montmory_jax.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/montmory_jax.py) | Exact 64-bit arithmetic represented by pairs of uint32 values. |
| Inspect optimized Montmory TPU kernels | [kernels.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py) | `small_product`, `round_carry`, `stream`, and `pallas_range`; tiling changes execution, not the intended stream. |
| Inspect the Threefry TPU comparator | [same kernel source](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py) | `threefry_bits_at` and the `threefry=True` path; 20 rounds, JAX partitionable uint32 output convention. |
| Call Montmory and apply a measured TPU selection | [tuned.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/tuned.py) | `generate` and `load_tpu_selection`; explicit matching of JAX/jaxlib and TPU kind. |
| Review selection and measurement | [campaign.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/campaign.py) | Separate search/confirmation seeds, precompilation, completion waits, checks outside timing. |

The C API is stateful **per logical lane**. The JAX interface is indexed over the documented global stream. Do not interpret the count of physical workers as the lane parameter or assume these APIs are interchangeable without the mapping in the stream contract.

## Minimal Montmory call on JAX CPU

Create the pinned JAX environment using [reproduce.md, section 4](en/reproduce.md#4-check-the-jax-port-locally). From the repository root, use that environment's interpreter:

```bash
JAX_PLATFORMS=cpu .venv-jax/bin/python - <<'PY'
from pathlib import Path
import runpy
import sys

kit = Path('studies/tpu-opt-2026-10-21/montmory-tpu-opt').resolve()
sys.path.insert(0, str(kit))
import tuned

low, high = tuned.generate(seed=42, base=0, count=4)
words = [int(low[i]) | (int(high[i]) << 32) for i in range(4)]
reference = runpy.run_path('src/reference/reference.py')
assert words == [reference['at'](42, 0, i + 1) for i in range(4)]
print([f'{word:016x}' for word in words])
PY
```

The returned arrays are low/high **uint32 halves**, not floating-point uniforms. The Python reconstruction above uses arbitrary-precision integers and does not require enabling JAX uint64. The per-lane reference starts its draw count at 1; `base` starts at 0. These four positions lie in logical lane 0 for the fixed `T=2048,D=4096` geometry. See the full contract before extending this mapping to other positions.

The example was executed on the local JAX 0.10.2 CPU environment and printed:

```text
['df2a584382db0894', 'c5d3386642e0256b', 'ec5078b8efb699cc', 'eaa6f5b8e64a247f']
```

Without an explicitly loaded native selection, `generate` uses the portable path. `load_tpu_selection` accepts a passing native campaign for the matching device kind and software versions. A recorded selection applies only to the measured output sizes; it is not universal hardware autotuning.

## Run the TPU study, or audit it without a TPU

For a new **native TPU** campaign, use the [optimization notebook](../notebooks/Montmory_TPU_optimisation.en.ipynb), select an actual TPU runtime, and follow [reproduce.md, section 5](en/reproduce.md#5-rerun-the-native-tpu-campaign). It contains both Montmory and Threefry/Pallas candidates. Keep the new result archive and its environment details. The previous recorded hardware was a TPU v5 lite; compilation or speeds on another generation are not guaranteed by those results.

For a **CPU reanalysis of the existing TPU evidence**, install NumPy in the dedicated audit environment and run `python3 scripts/replay_audit.py tpu`. This rechecks 3,708 archived measurements; it does not execute a TPU. The [CSV guide](en/csv-guide.md) explains the distinction between displayed GB/s and paired performance ratios.

The Threefry Pallas path in this campaign implements the specified JAX output convention for the recorded shapes and counters below `2^32`. It is not claimed to be a complete general-purpose replacement for all JAX `random` APIs, counter ranges, `split` or `fold_in` behavior. The Montmory prototype likewise does not register new validated JAX key-splitting semantics.

## Interpret a successful evaluation

A successful local check establishes agreement for its tested inputs. A successful native campaign adds measured performance for that hardware, software, shape and timing boundary. Neither implies cryptographic security, exact uniformity, independence of all logical lanes, or an application-level speedup. The [limitations](en/limitations.md) and [evidence register](en/evidence.md) explain which claims are finite tests, exact certificates, raw-data audits or terminal summaries.

For an evaluation report, retain the commit, source hashes, device, compiler/JAX versions, seed/index/lane geometry, output format, individual measurements and output-check results. Report both optimized competitors on the same workload; do not compare only against an untuned baseline. The current large-volume TPU results are approximately **89.49 GB/s Montmory** and **93.01 GB/s Threefry Pallas**, on one recorded session.

The files under `studies/` are historical snapshots. Read the curated pages first; use a working copy under `build/` for new campaigns. Do not edit a frozen kernel to make an old manifest pass.
