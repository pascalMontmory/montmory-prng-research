# References and prior work

[Français](../references.md) · [English](references.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md)

Montmory is evaluated in the context of existing PRNGs. Deterministic counter-based access and portable outputs are not claimed as general novelties.

| Primary source | Use in this record |
|---|---|
| [Salmon et al., Parallel Random Numbers: As Easy as 1, 2, 3 (2011)](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf) | Philox, Threefry and parallel counter-based generation |
| [Random123, commit 9545ff6](https://github.com/DEShawResearch/random123/tree/9545ff6413f258be2f04c1d319d99aaef7521150) | C/C++ comparator sources; known vectors and license retained |
| [TestU01, authors' documentation](https://www-labs.iro.umontreal.ca/~simul/testu01/guideshorttestu01.pdf) | Statistical batteries and parameters; archived version 1.2.3 |
| [JAX: benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) | Separate compilation/execution and synchronized measurements |
| [JAX: Pallas TPU PRNG](https://docs.jax.dev/en/latest/pallas/tpu/prng.html) | Pseudorandom generation in Pallas and scope of TPU options |
| [JAX: PRNG configuration](https://docs.jax.dev/en/latest/jax.random.html#advanced-rng-configuration) | Implementation names, API choices and reproducibility limits |
| [JAX: TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html) | Tile organization and memory; distinction from PRNG logical lanes |
| [NumPy: PCG64DXSM](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html) | Distinguish the standard engine from Montmory's project-specific historical name |
| [Blackman and Vigna: xoshiro256**](https://prng.di.unimi.it/xoshiro256starstar.c) | TACM comparator; attribution and notice present in source |

JAX documentation links may evolve. Repository results remain tied to **JAX/jaxlib 0.10.2** and frozen sources, not to a promise about every future version.

See also the [archived prior-work analysis](../../studies/audit-2026-10-21/ANTERIORITE.md), [commit-identified references](../../studies/audit-2026-10-21/evidence/references.json) and [TACM sources](../../studies/tacm-comparison-2026-10-21/SOURCES.md). No endorsement by Google, Apple, NVIDIA, D. E. Shaw Research or the TestU01 authors is implied.
