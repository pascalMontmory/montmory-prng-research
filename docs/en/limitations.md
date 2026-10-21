# Limitations and claims that must not be made

[Français](../limitations.md) · [English](limitations.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md)

1. **No universally superior PRNG is established.** Ranking depends on compiler, hardware, size and execution path. Threefry Pallas remains ahead of Montmory at confirmed large TPU volumes.
2. **No cryptographic security is claimed.** A Feistel structure, long state period and TestU01 pass do not constitute a security analysis.
3. **Exact uniformity is disproved for the documented mixer.** The non-injectivity certificate and first-mantissa-bin count are retained. This certificate alone does not measure application impact.
4. **Lane independence is not proved.** Lanes sharing a seed reindex the same state space. Reproducibility must not be confused with statistical independence.
5. **TestU01 passes have a specific scope.** Seeds, geometry and adapter matter. Code conformance checks do not add statistical battery passes.
6. **TACM establishes no statistical advantage.** The corrected campaign covers six functions and explicit tests; it does not validate every historical diagnostic or every Monte Carlo use.
7. **TPU performance comes from one session on one device.** Seven seeds and 21 seed/block pairs are not that many independent sessions. No energy comparison or native multi-TPU evaluation is available.
8. **Bitwise reproducibility concerns the integer contract.** The same seeds with a different geometry may change the stream. Application floating-point transformations and reductions need their own checks.
9. **Output checking partly relies on hashes.** Complete buffers are not archived; retained data allow reference reconstruction and hash auditing, not reinspection of every original native buffer.
10. **Evidence levels differ.** A value reported through the terminal remains distinct from a received, audited CSV. Missing items are listed in the [register](evidence.md).
11. **The other PRNGs are serious comparators.** Philox and Threefry predate this work; counter-based PRNGs, splittable keys and cross-architecture ports are not presented as Montmory inventions.
12. **No complete banking or AI application gain is demonstrated.** Generation performance alone does not establish reduced VaR error, faster pricing, regulatory compliance or accelerated training.

These limitations are part of the research result. Successful optimizations do not hide them.
