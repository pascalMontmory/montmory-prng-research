# Stream contract and exact properties

[Français](../contract.md) · [English](contract.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [C reference](../../src/reference/mmc_core.c) · [Integer Python model](../../src/reference/reference.py)

## Definition

The documented engine is the **recovered Montmory 3R engine**, used by the Metal v3 variants and TPU ports. “v3” may identify a Metal kit; it does not mean that each optimization introduced a new algorithm.

For global output index `k ≥ 0`, with `T > 0` logical lanes and `D > 0` consecutive outputs per lane:

```text
lane(k)  = floor(k / D) mod T
gamma(l) = (0x9E3779B97F4A7C15 + 2*l) mod 2^64
state(k) = (seed + (k+1)*gamma(lane(k))) mod 2^64
word(k)  = mixer(feistel_3R(state(k), seed))
mant53   = word(k) >> 11
u01      = mant53 * 2^-53
```

The state advances **before** producing the output. The per-lane model therefore starts `n` at 1; the global interface starts `k` at 0. The exact parameters and operations are defined in the two sources above and in the [detailed historical mathematical audit](../../studies/audit-2026-10-21/MATHEMATIQUES.md), preserved in its original language.

The main geometry in the recent work is `T=2048, D=4096`. The final TPU benchmark starts at `k=2^32−4097` to cross a counter boundary. Statistical campaigns must not be confused with this measurement window.

An output is identified by `(version, seed, k, T, D, format)`. Changing `T` or `D` may change the stream. Changing the physical work distribution can preserve it if each output retains the same global index and formula.

## Three different meanings of parallelism

| Term | Meaning in this work | Does it change the contract? |
|---|---|---|
| Logical lane | Identifier used to compute `gamma` | Yes, if its mapping from `k` changes |
| CPU worker / Metal thread / group | Organization of computation and writes | No, if indices remain identical |
| Pallas tile | A block of outputs processed together | No, if indices remain identical |

`pallas_carry_64` is **a Montmory implementation for TPU**. `64` means 64 rows of 128 outputs, or 8,192 words per tile; it does not mean 64 cores. `carry` uses a carry identity to calculate exactly the same integer result. The engine's three rounds and constants are unchanged. The [kernel sources](../../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py) remain those used in the campaign.

## Exact facts and their limits

For a fixed lane, the increment is odd: the **state counter** has an exact period of `2^64`. The three-round Feistel is a permutation of 64-bit words. Neither fact establishes cryptographic security or the minimum output period after the final mixer.

The final mixer applies a rotation whose selector depends on its input. It **is not injective**. These four distinct inputs all produce the integer output `7`:

```text
189da44d5bb5734c  63408dd79b3fd590
8abd642d2d15ccec  f3f0c11c862f9315
```

The certificate exhaustively computes candidates for the 32 possible rotations and checks each one. The first mantissa bin, `mant53=0`, has **2,066 preimages**, compared with 2,048 for an exactly uniform distribution over a `2^64` state cycle. The excess probability mass for this bin alone is `18/2^64`. This exact observation does not, by itself, measure practical Monte Carlo error.

Lanes sharing a seed traverse the same state space in different orders. Independence and absence of overlap are not guaranteed by construction. In particular, lanes 0 and 1 reach the same state at step `2^63`. The [certificates](../../studies/audit-2026-10-21/evidence/exact.json) and `scripts/check_reference.py` make these limitations verifiable.

The historical name `pcg_dxsm` refers to a project-specific mixer; **it must not be equated with standard PCG64DXSM**.

## Scope of bitwise reproducibility

The contract covers integer `raw64` words and integer `mantissa53` values. The prescribed conversion to binary64 is exact. Subsequent application transformations (`log`, `sin`, normal variates, parallel sums, valuations) do not automatically become identical across architectures.

Historical vectors, boundary checks and timed-output hashes provide substantial experimental conformance evidence. They do not cover every seed and index. Passing conformance checks does not imply a new statistical battery pass.
