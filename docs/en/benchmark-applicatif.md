# JAX application benchmark — protocol v1

[Français](../benchmark-applicatif.md) · [English](benchmark-applicatif.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Program](../../applications/jax_rng/benchmark.py) · [Raw kernel results](results.md)

**Objective: measure an operation that uses the RNG, with the consumer inside the same `jit`.** This protocol is separate from the raw-throughput campaign and changes none of its kernels. It provides a first reproducible application case; it is neither complete AI training nor a VaR engine.

## Three operations

| Operation | Timed work | Retained output |
|---|---|---|
| `uniform` | Generate words, extract 23 bits per uint32, convert to float32 uniforms strictly inside `(0,1)`. | A `(2, count64)` float32 array. |
| `normal` | Same generation and conversion, followed by Box–Muller with both sine/cosine outputs. | A `(2, count64)` float32 array. |
| `gaussian_expectation` | Same generation and normals, followed by a mean estimating `E[exp(-Z²/2)]`. | One float32 scalar. |

The ideal expectation for a continuous standard normal is `1/sqrt(2)`. Estimates are retained as diagnostics; this value is neither a bitwise reference nor a statistical pass threshold.

The uniform grid is `((word >> 9) + 0.5) * 2^-23`, excluding zero and one. It uses 23 bits per half-word, independently of the historical `mantissa53` format. It produces a discrete approximation to normals whose tails are limited by that grid. This explicit choice does not validate extreme financial quantiles.

## Comparisons specified before measurement

All three operations use exactly the same float32 transformations for Montmory `jax_original`, twenty-round Threefry2x32 JAX in partitionable mode, Philox4x32 JAX and, on TPU, the two Pallas variants below. Montmory geometry remains `T=2048,D=4096`. Configurations come from the **earlier raw-throughput campaign**; they are not selected using the new application results.

| `count64` | Float32 numbers produced | Montmory Pallas | Threefry Pallas |
|---|---:|---|---|
| 65,536 | 131,072 | `pallas_original_32` | `threefry_pallas_16` |
| 1,048,576 | 2,097,152 | `pallas_original_32` | `threefry_pallas_32` |
| 8,388,608 | 16,777,216 | `pallas_carry_128` | `threefry_pallas_128` |
| 16,777,216 | 33,554,432 | `pallas_carry_64` | `threefry_pallas_128` |

The program also includes **`threefry_native_normal`**: a direct call to `jax.random.normal`, then the same consumer for the expectation. This comparison represents common application usage. It uses a different transformation and must not be described as an identical-output comparison with Box–Muller. Differences in discretization, tails and dependencies must be examined before a production substitution.

The Pallas path retains its previous partitioning and may materialize intermediate arrays. An enclosing `jit` does not guarantee fusion with the Pallas body. This integration effect is part of the application result to measure. No specialized Philox Pallas implementation is included.

## Measurement and checks

- **JAX/jaxlib 0.10.2 required**, one device; no timed Pallas emulation on CPU and no silent TPU→CPU fallback.
- Montmory base `2^32−4097`. This base is not applied to Threefry counters; their outputs follow JAX's `(2,count64)` convention. Different engines are not expected to produce the same numbers.
- Seven new seeds defined in the program: `101`, `1009`, `65537`, `0xFEDCBA9876543210`, `0x800000000000002D`, `0x0123456789ABCDEF`, `0xFFFFFFFFFFFFFFF1`.
- Explicit compilation of every operation for a volume before timing, two warm-up calls per seed/operation, three blocks of three repetitions per seed.
- Deterministically shuffled order within each repetition. Matching seeds and blocks pair the comparisons; they are not seven independent hardware sessions.
- Wall time from the call through `block_until_ready`, including output allocation. Compilation, key preparation, host transfer and checks are excluded. Arguments remain dynamic to prevent precomputing a constant result.
- Complete integer arrays checked before timing: Montmory against the C reference; Threefry Pallas against the JAX API. Third-party APIs serve as their family references, without an additional independent oracle.
- Every timed output checked, including shape and type. Uniforms: exact equality with NumPy computation. Box–Muller and expectation: NumPy float64 reference, prespecified tolerances `atol=rtol=3e-5`. Native JAX normals: untimed JAX reference; their reduction is recomputed in float64.
- JSON report with environment, commit and worktree state, source hashes, verification counts and estimates; CSV containing every measurement, including order, seed, block and maximum error.

The project's cross-architecture reproducibility contract covers the **integer stream**. This program's floating-point tolerances do not promise bitwise-identical normals or reductions across CPU/GPU/TPU. Benchmark conformance replaces neither TestU01 nor TACM.

## Reading the results

The primary time is the median of per-seed medians. Application throughput is reported as **millions of variates processed per second**; stored-output GB/s is calculated only for the two operations materializing arrays. No fictitious memory throughput is assigned to a scalar output.

A ratio compares **per-seed paired median times**, then takes their median: `reference time / candidate time`. Above 1 means a faster candidate. The report also gives the number of favorable seeds. Comparisons are descriptive, with no automatic superiority test or selection after observation.

The primary operation for a future application claim is **`gaussian_expectation` at 16,777,216 words**, comparing Montmory Pallas against Threefry Pallas and native `jax.random.normal`. Other sizes and operations remain secondary results to report, including regressions. Confirmation in a new hardware session must precede a commercial performance promise. Gains in this case do not automatically transfer to a customer's model.

## Execution

From the repository root, in the pinned JAX environment described in [reproduce.md](reproduce.md), without changing the archives:

```bash
# Short VPS or Mac CPU check: 2 seeds, 2 repetitions, 1 volume.
.venv-jax/bin/python applications/jax_rng/benchmark.py \
  --backend cpu --smoke --out build/application-cpu-smoke

# Campaign on a VM with an actual TPU.
.venv-jax/bin/python applications/jax_rng/benchmark.py \
  --backend tpu --out build/application-tpu-v1
```

The output directory must be new. In Colab, use the interpreter where JAX is installed, such as `python`, and an authorized copy of the repository in the VM. A C compiler is required for the reference. The program installs no dependencies, creates no cloud resources and transmits nothing. Full campaigns contain 4,284 TPU measurements or 2,772 CPU measurements; short mode qualifies no TPU performance.

The [retained local CPU validation](../../applications/jax_rng/validation/README.en.md) documents the check and its limits. **No native TPU application result is provided yet.** The CPU benchmark confirms that the experiment runs, not that Pallas accelerates the application on TPU.

Method reference: [benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html), particularly compilation and synchronization of asynchronous computations.
