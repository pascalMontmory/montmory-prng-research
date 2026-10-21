# TestU01 register

[Français](../testu01.md) · [English](testu01.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Limitations](limitations.md)

A TestU01 pass means the battery reported no failure under its rules for **the stream and adapter actually executed**. It is not a universal proof of randomness, cryptographic security, lane independence or superiority over another engine.

## Available evidence

| Execution | Results | Available evidence level |
|---|---|---|
| Historical Metal, `Montmory-GPU-3R+DXSM` | SmallCrush 15, Crush 144, BigCrush 160 statistics: passes | Three complete logs imported with historical sources |
| Historical CUDA | Three batteries reported passed in logs | Complete imported logs; provenance and defects in old benchmarks examined separately |
| CPU audit replays | Seven SmallCrush passes covering multiple seeds/adapters and an irregular geometry | Sources, metadata and local logs retained |
| Native M3 Metal v3, `montmory_mulhi_pair` | SmallCrush 15: 4.74 s; Crush 144: 15 min 35.72 s; BigCrush 160: 1 h 44 min 36.42 s | Reported terminal excerpts/summaries; complete archive of latest BigCrush not received |
| Final TPU optimization | No new statistical battery | Conformance and performance checks only |

The counts 15, 144 and 160 are **statistics reported in the summary**, not necessarily the number of lines containing “p-value” throughout the log. Times are CPU times reported by TestU01, not GPU kernel times.

## Complete historical logs

- [Metal SmallCrush](../../studies/audit-2026-10-21/snapshot/montmory_gpu_test/montmory_gpu_test/logs/smallcrush_gpu.log)
- [Metal Crush](../../studies/audit-2026-10-21/snapshot/montmory_gpu_test/montmory_gpu_test/logs/crush_gpu.log)
- [Metal BigCrush](../../studies/audit-2026-10-21/snapshot/montmory_gpu_test/montmory_gpu_test/logs/bigcrush_gpu.log)
- [CUDA logs](../../studies/audit-2026-10-21/snapshot/cuda-ovh/logs)
- [CPU SmallCrush replay metadata](../../studies/audit-2026-10-21/evidence/smallcrush-runs.json)

Historical CUDA and Metal logs have identical printed p-values for the compared batteries. The [lineage analyses](../../studies/audit-2026-10-21/evidence/cuda-bigcrush-lineage.json) explain the scope of this observation. Two files naming different hosts are therefore not automatically two independent repetitions.

Historical CUDA geometry varies with the battery: `2048×4096`, `4096×8192`, `8192×16384`. Recent v3 geometry remains `2048×4096`. These campaigns must not be conflated.

## Latest M3 v3 BigCrush result

The [retained terminal summary](../../evidence/user-reported/m3-v3-bigcrush-summary.txt) explicitly reports 160 statistics and “All tests were passed”. Its [provenance record](../../evidence/user-reported/m3-v3-bigcrush.json) states that the full native archive remains absent from the repository. This text is a transcription of the supplied message, not a reconstructed log.

The tested adapter supplies uniforms derived from 53-bit mantissas. Passing through this adapter does not replace examining every projection of `raw64`. The audit also contains separate SmallCrush runs on 32-bit halves, but no exhaustive BigCrush coverage of all such projections is claimed.

Philox and Threefry also have published TestU01 results, depending on variant and round count. Passing is not an exclusive Montmory advantage; see [Random123 and its results](https://github.com/DEShawResearch/random123).
