# Decisions and corrections

[Français](../history.md) · [English](history.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md)

This table describes the lineage of work available on 21 October 2026. It does not assign new execution dates to recovered historical logs.

| Stage | Work / observation | Retained consequence |
|---|---|---|
| Source recovery | C, Metal and CUDA references; historical TestU01 logs | Sources and hashes retained; log lineage examined |
| Mathematical audit | Odd counter increment, bijective Feistel; non-injective final mixer | State period distinguished from output period; no exact-uniformity claim |
| Historical CUDA audit | Missing bounds guard for irregular geometry; incorrect old Philox comparator and throughput conversion | Old figures excluded from meaningful comparisons; correction not presented as natively validated |
| CPU comparison | Random123 and known vectors; separate compilation and access modes | Ranking depends on context |
| Corrected TACM | Failed calibration pilot, one documented successful extension; no MSE advantage established | Pilot and negative results retained |
| Metal v3 | Repaired TestU01 integration, native conformance; three batteries reported passed | Latest BigCrush classified as a terminal summary; full archive still absent |
| CPU/Metal optimization | Pair computation, workers, distribution variants | CPU gains, small GPU gain; first-use effects contaminated some initial measurements |
| Separate Metal confirmation | Precompilation and warm-up; new seeds | Threefry and Philox remain ahead; raw confirmation archive missing |
| Vector experiments | Initial Apple Clang failure on `auto` deduction of `id`, then repair; CPU checks with explicit evaluation order | Traces retained; GPU `vector4` slower, CPU pool useful on small blocks |
| JAX port and first Colab attempts | Local checks followed by environment errors without an allocated TPU | Initialization failures are neither engine failures nor TPU measurements |
| First TPU campaign, 18:06 | Native conformance and 770 observations | Initial comparison lacked a specialized Threefry Pallas implementation |
| TPU optimization, 18:51 | Exact carries, tile sizes, added Threefry Pallas; 3,708 observations | Montmory gains at 64/128 MiB; Threefry Pallas remains faster |
| Repository assembly | Documentation, provenance and reproducible audits | No engine change; algorithm development paused |

Environment errors, rejected trials and approaches not adopted remain accessible in the studies. Their presence in an archive does not make them part of the protocol's recommended configuration.
