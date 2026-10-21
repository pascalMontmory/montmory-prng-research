# Validation methodology

[Français](../methodology.md) · [English](methodology.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Provenance](evidence.md) · [Reproduction](reproduce.md)

## Separate questions

| Question | Check | What passing does not establish |
|---|---|---|
| Does the program implement the defined stream? | C reference, integer model, vectors, boundaries, CPU/GPU/TPU comparisons | Statistical quality or speed |
| Does the stream exhibit the targeted defects? | TestU01 and corrected TACM diagnostics | Absence of every defect or cryptographic security |
| Is an implementation faster? | Repeated measurements with checked outputs and an explicit timing boundary | Superiority on other hardware or in a complete application |

The studied optimizations preserve the engine. No Markov chain was added to the confirmed implementations.

## Conformance before and during timing

Reference sources and kits are frozen by SHA256. Checks cover ordinary and extreme seeds, carry boundaries, work partitioning and historical vectors. Outputs from each timed call are checked **outside the timing window**.

In the latest TPU campaign, the 3,420 calls from engines with a fixed CPU reference use 240 pre-established hashes. The 40 Montmory hashes were then recomputed from a fresh compilation of the frozen C source. The 288 RBG calls check repeatability only on the same backend. These categories are not combined into a cross-architecture guarantee.

Full output arrays are not retained; hashes and reports are. The word count including repetitions is not the number of distinct inputs. The TPU campaign checked 10,406,854,656 Montmory words including repetitions, but only 167,772,160 distinct seed/index pairs.

## TPU search and confirmation

The [frozen campaign program](../../studies/tpu-opt-2026-10-21/montmory-tpu-opt/campaign.py) defines four volumes, 26 configurations and the pseudorandom measurement schedule.

1. **Search:** three seeds, three repetitions, four volumes, 26 configurations; 936 observations.
2. **Frozen selection:** two Montmory candidates and one optimized Threefry comparator are selected using search results. The selection JSON is hashed before confirmation.
3. **Confirmation:** seven new seeds, three blocks of three repetitions, eleven finalists per volume; 2,772 observations.
4. **Adoption:** the primary candidate replaces the previous `32×128` Pallas implementation only if its paired gain is at least 3% and it is faster for all seven seeds. Otherwise the reference is retained.

A secondary candidate that is faster during confirmation is not promoted after the fact. At 64 MiB, this specifically prevents opportunistic selection of `pallas_carry_64` after observing its results.

The 104 compilations precede their corresponding measurements. Calls wait for result completion. The audit verifies counts, absence of duplicates, planned order, seed separation, selection, decisions and all 44 summaries. Official JAX documentation explains why compilation and asynchronous execution must be distinguished in a [benchmark](https://docs.jax.dev/en/latest/benchmarking.html).

## Timing boundaries

| Campaign | Measured time | Exclusions or specific conditions |
|---|---|---|
| ARM64 CPU / workers | Caller elapsed time | `pair_reused_workers` excludes initial pool construction; `spawn` variants include worker creation/join |
| Metal | GPU timestamps for submitted work | Does not measure a complete CPU→GPU→CPU application |
| TPU JAX/Pallas | Host call, allocation and completion wait | Compilation, key preparation, result transfer and verification excluded |
| TestU01 | CPU time reported by the battery | Not a benchmark of the GPU kernel alone |

CPU, Metal and TPU GB/s must not be compared as though they had identical timing boundaries. The write baseline is neither a PRNG nor a universal hardware ceiling.

## Statistics, calibration and multiple comparisons

Complementary diagnostics evaluate explicit functions with known reference moments. They use disjoint pairs where required by the assumed distribution, a shared uniform conversion and prespecified comparison families. [TACM](tacm.md) documents assumptions, approximations, multiplicity corrections and the failed calibration pilot.

An isolated p-value near 0.01 or 0.99 is not a quality score. Trying to move every p-value toward 0.5 would be a poor target. Conclusions depend on each protocol's rules and scope.

## Uncertainty and reproducibility

Per-seed medians, dispersion and block comparisons are retained. Seven seeds from one session do not replace seven independent hardware sessions. Current TPU results still require replication in a new session and on other models before extrapolation.

The repository CI checks integrity, exact certificates and data consistency on CPU. It does not rerun SmallCrush/Crush/BigCrush, has no TPU and does not establish new native performance results.
