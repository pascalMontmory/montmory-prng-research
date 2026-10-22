# Exact Sobol32 prototype for JAX and TPU

[Français](README.fr.md) · [English](README.md)

Developed by Pascal Montmory for ENINCA CONSULTING. This is a separate experiment;
the Montmory/Threefry engines and historical measurements are unchanged.
Original contributions remain subject to the repository [LICENSE](../../LICENSE).

**Status: CPU validation and Pallas CPU interpretation only. Native TPU compilation,
throughput, GPU execution and application benefits are not established.**
The CPU pilot is a functional test, not evidence of TPU performance. No improvement
over every existing Sobol implementation is claimed.

**Try it:** [English Colab notebook](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/sobol/notebooks/Eninca_Sobol_TPU.en.ipynb)
or [French Colab notebook](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/sobol/notebooks/Eninca_Sobol_TPU.ipynb).
The notebook embeds its checked source kit. Select a TPU runtime; access depends
on your own Colab allocation and quota. The default run is a small pilot.

## Exact contract

- 128 dimensions, 32 direction bits, output shape `(points, 128)`, dtype `uint32`.
- Logical indices satisfy `0 <= start < start+points <= 2**32`; no stream wrap.
- Dimension order and direction-bit order are fixed. The first point is retained.
- Direction numbers and optional LMS+digital-shift scrambling come from **SciPy
  1.17.1**. NumPy **2.4.6**, JAX/jaxlib **0.10.2** are pinned.
- `seed=None` is unscrambled; integer seeds use SciPy's LMS+shift, not a newly
  invented scrambling algorithm. Scrambling is prepared on the host.
- The durable stream identity is the **exported direction matrix and shift**, not
  just a seed. Each benchmark saves them in NPZ and records a canonical SHA-256
  of little-endian direction bytes followed by shift bytes. A future SciPy/NumPy
  version must not silently redefine the input parameters.
- Each dimension's word is `shift XOR directions selected by Gray(index)`.
  CPU, GPU and TPU implementations must return the same integers for identical
  parameters and indices. Only CPU/interpreter execution has been performed here.
- Optional `uniform24` maps the upper 24 bits to exact float32 dyadic values in
  `[0,1)`. It deliberately discards eight bits and is **not** SciPy's full 32-bit
  float64 coordinate mapping. Raw32 is the benchmark's only output format.

The current public API is a research interface: host `reference.check_range`
must precede calls to `kernels.make_kernel`. Its `aligned=True` fast path also
requires a start divisible by the variant's block size. The campaign enforces
that requirement. The general path handles arbitrary starts and tails through
aligned padding and slicing. Discarded padding may wrap; returned logical indices
never wrap. Allocations of `2**31` or more output elements are outside this prototype.
Multi-device sharding and dimensions other than 128 are not implemented.

## What is optimized

For a power-of-two block size B, an aligned base b, and `0 <= r < B`:

```text
Gray(b+r) = Gray(b) XOR Gray(r)
Sobol(b+r) = common(b) XOR local_table[r]
```

The shift occurs once in `common`. The local table is unshifted. This standard
linearity identity is an implementation technique, not a new Sobol construction.
The highest low Gray bit of an aligned base is at `log2(B)-1`, which is retained.
The kernel shares that base contribution across the block instead of evaluating
32 direction terms independently for every point.

Eight candidates: a direct JAX Gray implementation; JAX block implementations
at 128/256/512 points; a direct Pallas kernel at 32 points; and Pallas block
kernels at 128/256/512. All inputs remain runtime parameters. There is no
seed-specific constant folding. The block tables consume 64/128/256 KiB per
parameter set. Their creation and device upload are recorded separately.

## Verification and measurement

`verify.py` checks public SciPy outputs, an explicit eight-point fixture, an
independent sequential recurrence, high indices, tails, partitions, the final
legal point, invalid ranges, and float32 mapping boundaries. The SciPy and host
routes share SciPy's direction data; they do not constitute an independent
derivation of Joe–Kuo's tables. High-index tests use the host recurrence instead
of requesting billions of SciPy fast-forward steps.

`run.py` refuses backend fallback and saves diagnostics on failure. CPU mode
interprets Pallas for correctness but **never times that interpreter**. Its
performance pilot times only native JAX CPU functions. Native TPU mode checks
and times all eight candidates, on the device actually reported by JAX.

The full campaign uses 512, 8,192 and 131,072 points (256 KiB, 4 MiB, 64 MiB).
Three search scrambling seeds select the best JAX path, including the direct
Gray path, and the best Pallas path independently. Five new confirmation seeds
then compare those fixed choices, with seven randomized repetitions each.
Quick mode uses two volumes, one search seed and two confirmation seeds with
three repetitions; it is only a pilot.

Compilation and warm-up precede timing. Each timed output is compared in full
against the host oracle outside the timer. Timings include dispatch, allocation,
output writes and `block_until_ready`; they exclude setup, upload, compilation,
warm-up, host transfer and verification. The result is a **resident-input,
reused-parameter** benchmark, not cold one-shot performance. Setup durations are
recorded in `preparation.json`. The input stream range is aligned and varies by
case. The same range is used for every candidate in a case.

Reports retain raw CSV times, selection, per-seed paired ratios, source hashes,
parameter hashes, environment versions, and checked-word counts. A ratio above
one means faster than the selected JAX baseline. Median paired ratios need not
equal quotients of summary GB/s. No universal or statistically significant
speedup is inferred from a pilot.

The saved [CPU report](validation/cpu/report.json) records 2,097,152 words compared
with public SciPy output and 14,314,496 JAX/Pallas-interpreted words checked in
conformance tests. Its [raw pilot timings](validation/cpu/measurements.csv) are
CPU-only; the block path did not win consistently across the tested sizes/cases.
Parameters and environment details are retained alongside the report.

## Run

Use a separate virtual environment with Python supported by the pinned packages:

```bash
python -m pip install 'jax==0.10.2' 'jaxlib==0.10.2' 'numpy==2.4.6' 'scipy==1.17.1'
python applications/sobol/run.py --backend cpu --quick --out build/sobol-cpu
```

On an allocated TPU, install the same NumPy/SciPy versions and `jax[tpu]==0.10.2`,
then use `--backend tpu`. Add `--verify-only` for conformance only; remove `--quick`
for the full campaign. Output directories must be new. Failures also produce a
`report.json`, `error.log`, and adjacent `.tar.gz` archive.

Build self-contained French/English Colab notebooks and a checked source kit:

```bash
python applications/sobol/build_notebook.py --out build/sobol-delivery
```

The notebook first checks TPU allocation with full subprocess logs. It defaults
to quick mode. It never silently switches to CPU, requests cloud resources, or
transmits results. The final cell downloads results or startup diagnostics.

## Scope and sources

Sobol is a low-discrepancy sequence, **not an IID PRNG**. Arbitrary subsets remain
exact stream slices but need not retain the balance properties of a full `2**m`
sample. No TestU01 campaign or application error/convergence comparison is claimed.

- [SciPy Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html): direction numbers, LMS+shift and balance requirements.
- [Joe–Kuo direction-number data](https://web.maths.unsw.edu.au/~fkuo/sobol/): source credited by SciPy.
- [JAX Pallas TPU](https://docs.jax.dev/en/latest/pallas/tpu/details.html): backend-specific memory and tiling constraints.
- [Third-party notices](../../THIRD_PARTY_NOTICES.md): SciPy/NumPy/JAX are external dependencies. No ownership of Sobol, its direction-number tables, or upstream implementations is claimed.
