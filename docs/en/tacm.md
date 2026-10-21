# Complementary TACM diagnostics: corrected status

[Français](../tacm.md) · [English](tacm.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Frozen protocol](../../studies/tacm-comparison-2026-10-21/protocol.json) · [Detailed historical methods](../../studies/tacm-comparison-2026-10-21/METHODES.md)

TACM is the name used here for the project's complementary diagnostic program. The archived campaign is **a corrected, explicitly defined subset**, not a general validation of every historical idea bearing that name.

Its objective is to examine Monte Carlo estimators and certain relationships between outputs, complementing TestU01 batteries. Passing a general battery does not, by itself, establish accuracy for a particular application function.

## Experimental design

| Item | Value |
|---|---|
| Engines | Montmory, Philox4x32-10, Threefry2x32-20, Threefry4x64-20, xoshiro256**, NumPy PCG64, PCG64DXSM, MT19937, SFC64 and Philox4x64 |
| Repetitions | 2,048 seeds per engine, organized into 128 groups of 16 |
| Outputs | 65,536 words per seed; common conversion `(raw64 >> 11) × 2^-53` |
| Functions | `U`, `U²`, sine, event `U>0.99`, product of two uniforms, joint rare event |
| Pairs | Disjoint pairs for two-coordinate functions |
| Primary comparisons | 9 competitors × 6 functions = 54 |
| Advantage rule | MSE reduction greater than 10%, Bonferroni-adjusted p-value < 0.05, and the same effect direction in both predefined halves of the groups |

The campaign produces **1,342,177,280 functional words** and **122,880 estimates**. Seeds are deterministic and distinct under the protocol's namespaces; this does not prove their mathematical independence.

Starting at index zero, 65,536 outputs cover only 16 of Montmory's 2,048 logical lanes. A separate check covers 64 seeds, 2,048 lanes and 16 positions per lane for three engines. The first experiment must not be presented as an exhaustive exploration of every lane.

## Calibration: retain the failed pilot too

Initial calibration contains 512 trials per reference source. It does not meet its global criterion. This failure is retained in the data.

An [addendum fixed before the functional comparison](../../studies/tacm-comparison-2026-10-21/calibration_addendum.json) specifies one extension to 4,096 trials per source, without changing thresholds. The extension meets the criterion: the eight observed alert rates range approximately from 0.781% to 1.196% for a nominal 1% threshold. The three strongly injected defects are detected in all 4,096 trials of each of the two sources.

These checks use two reference PRNGs. They are neither a perfect physical IID source nor a guarantee of power against every weak defect. Several distributions and second-level tests remain approximate; the methods identify them.

## Result

**None of the 54 comparisons establishes the prespecified Montmory advantage.** None of the 93 adjusted diagnostic comparisons raises an alert. This establishes neither statistical equivalence of all engines nor absence of differences for other functions or sizes.

Montmory MSEs relative to nominal IID variance are approximately 0.978 / 0.987 / 0.995 / 1.016 / 0.966 / 1.003 across the six functions. Values slightly below 1 are not enough to claim a better PRNG: they are fluctuating estimates.

An independent check of 180 estimates using `math.fsum` and exact polynomial moments is [archived](../../studies/tacm-comparison-2026-10-21/results/independent_replay.json). The [machine-readable summary](../../studies/tacm-comparison-2026-10-21/results/summary.json), CSVs and recalculation scripts are included.

## Corrections to historical diagnostics

Earlier interpretations relying on a universal variance bound, a `bridge_gap` that is zero by construction, or overlapping pairs treated as independent are not reintroduced into this comparison. The [audit work](../../studies/audit-2026-10-21/README.md) documents that context.

“Improving calibration” means improving the reliability of diagnostic thresholds. It does not prove that Montmory has a defect or mean its p-values should be moved toward 0.5. That wording alone justifies no engine modification.
