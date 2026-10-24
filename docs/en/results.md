# Results as of 24 October 2026

[Français](../results.md) · [English](results.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Methodology](methodology.md) · [Reading the figures](csv-guide.md)

## TPU: confirmed optimization with a strong comparator

Native campaign `20261021T185102Z-tpu-opt`, **one TPU v5 lite**, JAX/jaxlib 0.10.2, geometry `T=2048,D=4096`, starting index `2^32−4097`. Throughputs below use the median of seven per-seed median times; each seed has nine confirmation measurements.

| Implementation | 512 KiB | 8 MiB | 64 MiB | 128 MiB |
|---|---:|---:|---:|---:|
| **Retained Montmory** | **2.120** | **15.859** | **62.458** | **89.488** |
| Previous Montmory Pallas | 2.120 | 15.859 | 54.200 | 73.498 |
| Portable Montmory JAX | 2.026 | 15.138 | 51.075 | 66.769 |
| Threefry2x32 JAX | 1.979 | 12.982 | 31.954 | 33.364 |
| Selected Threefry2x32 Pallas | 2.013 | 15.917 | 65.249 | 93.006 |
| Threefry4x32 JAX | 1.921 | 10.691 | 20.396 | 21.331 |
| Philox2x32 JAX | 1.885 | 10.650 | 20.302 | 20.925 |
| Philox4x32 JAX | 1.664 | 7.867 | 12.193 | 12.463 |
| RBG, same backend only | 2.096 | 18.465 | 86.811 | 90.791 |
| Write baseline, not a PRNG | 2.131 | 17.790 | 91.322 | 151.754 |

Unit: **decimal GB/s**; volumes: binary KiB/MiB. [Complete data and audit](../../studies/tpu-opt-2026-10-21/results/audit-20261021T185102Z-tpu-opt/README.md).

| Volume | Retained configuration | Paired gain over previous Pallas | Reason |
|---|---|---:|---|
| 512 KiB | `pallas_original_32` | 0% | Primary candidate: −0.63%, faster for only 2/7 seeds |
| 8 MiB | `pallas_original_32` | 0% | Primary: +1.64%, 6/7; threshold not met |
| 64 MiB | `pallas_carry_128` | +15.42% | Threshold met; 7/7 seeds and 21/21 seed/block pairs faster |
| 128 MiB | `pallas_carry_64` | +23.25% | Threshold met; 7/7 seeds and 21/21 seed/block pairs faster |

Paired gains are not quotients of the rounded throughputs in the first table. At 128 MiB, +23.25% throughput corresponds to approximately **18.86% less time** under that paired ratio.

Against Threefry2x32 JAX, Montmory's paired ratios are ×1.926 at 64 MiB and ×2.692 at 128 MiB. Against **Threefry Pallas**, they are 0.960 and 0.961 respectively: Montmory remains about 4% slower. This Pallas comparator retains twenty rounds and the outputs of the Threefry2x32 JAX API used here. No specialized Philox Pallas implementation was measured; ratios against JAX Philox APIs do not rank every possible implementation.

The first TPU measurements, in the [18:06 campaign](../../studies/jax-tpu-2026-10-21/README.md), did not include the Threefry Pallas comparator. They were therefore insufficient to claim general superiority.

## Hardware PRNG and the measured RBG comparator

The software kernels aim to approach useful generation throughput while retaining a defined, portable integer stream. The [README](../../README.md#why-optimize-a-software-prng-on-tpu) explains the tradeoff with the hardware PRNG.

In this campaign, `rbg_same_backend` specifically means `jax.random.key(..., impl="rbg")` followed by `jax.random.bits(..., dtype=uint32)`; see `arguments` and `function` in [kernels.py](../../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py). This path uses XLA RBG. JAX does not give it the same identical CPU/GPU/TPU output guarantee as Threefry. [JAX PRNG configuration](https://docs.jax.dev/en/latest/jax.random.html#advanced-rng-configuration).

The table compares call-to-ready throughput, including allocation and writes. At 128 MiB, Threefry Pallas reaches 93.006 GB/s, Montmory Pallas 89.488 and RBG 90.791. At 64 MiB, the respective values are 65.249, 62.458 and 86.811. Their proximity at 128 MiB therefore does not generalize to every size.

**No dedicated kernel using the Pallas hardware calls `prng_seed`/`prng_random_bits` was timed in this campaign.** The RBG figure is neither the chip's maximum rate nor a direct comparison with every hardware path. Software throughput exceeding this RBG path does not prove that its elementary operations are faster than the silicon PRNG's operations.

Montmory and Threefry outputs are checked against their defined references; RBG is checked only for repeatability on the same backend. This makes the intended tradeoff explicit: performance with stream stability. Stability rests on a software contract; these raw-generation campaigns do not demonstrate its preservation across multiple TPU generations or, by themselves, an application speedup. Dropout training is measured separately below.

## Apple M3: retain the best measured path

The first received Metal optimization archive is [audited here](../../studies/metal-opt-2026-10-21/results/mac-analysis-20261021T162407Z-22542/analysis.json). It contains 144 CPU and 2,574 GPU observations. The analysis detects first-use effects in some specialized variants. A separate campaign subsequently precompiled and warmed every pipeline.

The **separate confirmation below is a reported terminal summary**, with seven new seeds and fifteen measurements per seed. Its complete raw archive was not received in this record.

| Metal raw64, 64 MiB produced | Displayed GB/s | Paired ratio to Montmory v3 |
|---|---:|---:|
| Montmory `mulhi_pair` | 84.787 | 1.000 |
| Montmory `split_stripe_pair` | 85.218 | 1.005 |
| Philox4x32-10 | 88.979 | 1.049 |
| Threefry2x32-20 | 92.082 | 1.086 |
| Write baseline, not a PRNG | 96.018 | 1.133 |

[Summary and provenance](../../studies/metal-opt-2026-10-21/results/mac-confirm-reported-20261021T163954Z-22644.json). Here Threefry is **slower than the write baseline**, and faster than Montmory.

Later Metal vector experiments do not improve this picture: `vector4` is around 76.6 GB/s against 84.8 for `mulhi_pair`. `split_stripe_pair` gains about 0.4%. These later results were also [reported through the terminal](../../studies/vector-opt-2026-10-21/README.md), without a complete received native archive. Explicit vectorization is therefore not adopted as an automatic improvement.

## CPU: launch overhead matters

On M3, the initial archive confirms the benefit of scalar pair computation: 12.487 GB/s with eight workers against 9.746 GB/s for the indexed variant, with identical outputs. The paired gain is about 28%.

For small blocks of 65,536 words, terminal confirmation of the four-worker pool reports a paired gain of **30.7%** over creating four workers on each call, across seven seeds. Pool gains are much smaller on large blocks. [Sources, protocol and vector results](../../studies/vector-opt-2026-10-21/README.md).

On the x86 VPS, the [physical partitioning test](../../studies/lanes-study-2026-10-21/README.md) gives 2.02 / 3.87 / 7.38 / 3.35 / 3.09 GB/s with 1 / 2 / 4 / 8 / 16 workers. Four workers are best in that session. This is not a rule for every processor and does not change logical lanes.

The [Random123 CPU comparison](../../studies/audit-2026-10-21/evidence/benchmark-summary.json) changes ranking with compilation, contiguous or scattered access and the engine compared. It does not establish universal Montmory dominance.

## Statistical quality and applications

The [TestU01 register](testu01.md) identifies available passes and their provenance limits. [TACM comparisons](tacm.md) show no Montmory statistical advantage under the prespecified criterion. TPU optimizations changed neither engine constants nor rounds, but did not rerun TestU01 or TACM.

Small-model dropout training now has native TPU measurements, presented below. No native gain for complete VaR, pricing applications or energy consumption is established. This evidence does not quantify a commercial benefit.

A [first JAX application benchmark](benchmark-applicatif.md) is now available: uniforms, normals and a Monte Carlo estimator with the consumer inside the same `jit`. Its short CPU check is retained separately. It does not yet provide native TPU application measurements or confirm gains in a customer's application.

## Dropout: native TPU training on 24 October 2026

The [detailed report](../../applications/dropout/validation/tpu-20261024/README.md)
retains 1,800 measurements, exact sources, checks and learning curves for a
21,252-parameter MLP on TPU v5 lite. Pallas reduces Montmory step time against
Montmory JAX by **5.76%, 5.10% and 6.27%** across three seeds.
All 900 JAX/Pallas loss pairs are numerically identical; the runner also reports
zero final-weight differences, although weights were not exported in the archive.

Montmory Pallas remains close to Threefry JAX: +0.75%, −0.89% and −0.98% step time.
Threefry Pallas takes about 0.3–0.5% more time than Threefry JAX. Moving to Pallas
changes the implementation, not the Threefry algorithm; the TPU hardware PRNG
is not used here.

Mean final accuracy is 82.55% for Montmory and 83.01% for Threefry. One session,
three seeds and a synthetic task establish neither superior quality nor
statistical equivalence. Comparing with the earlier CPU functional check gives
about 1.48× for Montmory Pallas, but remains indicative and does not isolate a
PRNG-specific gain. No comparable GPU training run is available.

The [Sobol32 prototype](../../applications/sobol/README.md) and its TPU notebooks
are separate. Native TPU compilation and performance remain unvalidated.
