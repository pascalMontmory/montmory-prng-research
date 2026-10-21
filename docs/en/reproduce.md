# Reproduce the checks

[Français](../reproduce.md) · [English](reproduce.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Methodology](methodology.md)

All commands start at the repository root unless stated otherwise. The local checks below need neither cloud credentials nor TPU hardware.

## 1. Integrity and exact reference

Requirements: Python 3.11 or newer and a C compiler with UBSan (`cc`/Clang/GCC).

```bash
python3 scripts/check_repository.py
python3 scripts/check_reference.py
```

The first command checks hashes of the 884 imported files, local links in the main documentation, kit JSON manifests and result counts. The second compiles the unchanged C source with UBSan, checks it against 1,600 vectors, compares the Python model and recomputes the exact mixer and lane certificates.

New outputs go into `build/`, which Git ignores. No statistical battery is rerun.

## 2. Recompute the TPU audit on CPU

Additional requirement: NumPy. The audit needs neither JAX nor `libtpu`.

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install numpy==2.4.6
.venv-audit/bin/python scripts/replay_audit.py tpu
```

The script works in a temporary copy, rebuilds the C oracle from source, regenerates Montmory references and checks the 3,708 rows, measurement order, selection and 44 summaries. It does not execute any compiled library received from the TPU. A dated directory under `build/` retains the log and recomputed analysis.

A successful data audit is not a native benchmark replication. Temporary paths and some report metadata may differ; scientific values and checks should agree.

## 3. Recompute Metal and TACM analyses

The Metal analysis needs matplotlib in addition to NumPy. It covers **the first received M3 archive**, not the confirmations available only as terminal summaries:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install numpy==1.24.2 scipy==1.10.1 pandas==1.5.3 matplotlib==3.6.3
.venv-analysis/bin/python scripts/replay_audit.py metal
.venv-analysis/bin/python scripts/replay_audit.py tacm
```

TACM also requires a C++17 compiler (`g++`). The script rebuilds the comparators and their checks, recomputes conclusions from archived estimates and independently replays 180 estimates. It does not rerun the entire billion-output campaign or adjust thresholds.

TACM and JAX versions use separate environments: do not merge their historical dependency files into one environment.

## 4. Check the JAX port locally

This optional step actually runs the port on the **available CPU**, without measuring a TPU:

```bash
python3 -m venv .venv-jax
.venv-jax/bin/python -m pip install -r studies/tpu-opt-2026-10-21/montmory-tpu-opt/requirements.txt
JAX_PLATFORMS=cpu .venv-jax/bin/python studies/tpu-opt-2026-10-21/montmory-tpu-opt/verify_cpu.py --out build/jax-cpu-new
JAX_PLATFORMS=cpu .venv-jax/bin/python studies/tpu-opt-2026-10-21/montmory-tpu-opt/verify_candidates.py --backend cpu --out build/jax-candidates-new
```

On macOS, install the Xcode command-line tools. Interpreted Pallas execution on CPU checks values, not native TPU compilation or performance. Use a new output path for each retained campaign.

This environment can also replay the first TPU campaign's audit:

```bash
.venv-jax/bin/python scripts/replay_audit.py tpu-initial
```

## 5. Rerun the native TPU campaign

Download the [standalone optimization notebook in English](../../notebooks/Montmory_TPU_optimisation.en.ipynb), import it into Colab, select a **TPU** runtime and run the cells in order. The notebook embeds the frozen kit; no GitHub authentication is needed after downloading the file. Its executable cells and original diagnostic messages are unchanged.

Before timing, the log must report backend `tpu` and an actual device. `No jellyfish device found` means TPU initialization failed; it is not a performance result. Colab allocation time and compilation are excluded from benchmark timings.

The notebook uses pinned versions, verifies sources, runs conformance/search/confirmation and provides a result archive. Retain the new archive with its identifier; do not replace the 21 October archive. TPU runtime availability depends on the service and is not guaranteed by this repository.

On an already configured TPU VM, a copy of the kit can also be used:

```bash
mkdir -p build
cp -R studies/tpu-opt-2026-10-21/montmory-tpu-opt build/tpu-new
.venv-jax/bin/python -m pip install -r build/tpu-new/requirements-tpu.txt
.venv-jax/bin/python build/tpu-new/run.py
```

`run.py` requires a TPU by default. `--cpu-smoke` only checks orchestration on CPU and must never be labeled a TPU result.

## 6. Rerun Metal and TestU01 on Apple Silicon

For Metal confirmation, copy the kit outside the frozen archives:

```bash
mkdir -p build
cp -R studies/metal-opt-2026-10-21/montmory-metal-confirm-m3 build/metal-confirm-new
bash build/metal-confirm-new/run.sh
```

To rerun SmallCrush using the repaired TestU01 launcher:

```bash
mkdir -p build
cp -R studies/audit-2026-10-21/mac-metal-benchmark-v3 build/metal-testu01-new
cp -R studies/audit-2026-10-21/mac-metal-testu01-repair-v1 build/metal-testu01-new/testu01-repair-v1
bash build/metal-testu01-new/testu01-repair-v1/run.sh montmory_mulhi_pair small
```

The `crush`, `big` or `all` arguments run the other batteries and may occupy the machine for a long time. Each battery restarts with seed 42 and `T=2048,D=4096`. The kit contains TestU01 1.2.3 source and its notice; installation is local to the working directory.

These commands are documented from the retained kits. Assembling the repository did not rerun a native Metal or TPU campaign. Other historical scripts may contain old VPS paths or depend on intentionally excluded binaries; they are not all portable entry points for this repository.

## 7. Preserve new evidence

Keep together: commit, source manifest, dependency/compiler versions, hardware, logical geometry, seeds, indices, format, protocol, individual measurements, output hashes, exit code and analysis. Identify a new campaign, a reanalysis and a terminal transcription separately. The [CI workflow](../../.github/workflows/validate.yml) runs only CPU checks of the research record.

The [historical local report](../../results/repository-validation.json) records checks completed when the research record was assembled. It is not evidence of a hosted CI pass for this public edition.

## 8. Public GitHub Actions

The [Actions page](https://github.com/pascalMontmory/montmory-prng-research/actions) records checks for the public repository. A run that does not start is not a test result. Local validation, archived native runs and hosted CI are separate evidence categories.

For the bilingual editions, also run `python3 scripts/check_translations.py`. The [language guide](languages.md) describes the translated scope and the materials preserved unchanged.
