# Understanding the campaign CSV

[Français](../csv-guide.md) · [English](csv-guide.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [Reading workbook](../../results/Montmory_TPU_resultats_lisibles.xlsx) · [Raw CSV](../../studies/tpu-opt-2026-10-21/20261021T185102Z-tpu-opt/campaign/measurements.csv)

**One CSV row represents one timed call, not an engine's final performance.** The file contains 936 search rows and 2,772 confirmation rows. Comparing two isolated rows does not replace summarizing repetitions.

## Columns

| Column | Meaning |
|---|---|
| `phase` | `search` selects configurations; `confirmation` evaluates them on new seeds |
| `engine` | Exact implementation; for example, `pallas_carry_64` is Montmory |
| `words` | Number of 64-bit outputs, or 8 bytes per word |
| `seed` | Exact seed; keep this column as **text** in Excel |
| `block` | Measurement block, numbered from zero |
| `repetition` | Repetition within the block, numbered from zero |
| `nanoseconds` | Call duration until the result is ready; lower is faster |
| `output_sha256` | Output hash checked outside timing |
| `check` | Fixed CPU reference, or same-backend repeatability only for RBG |

## Units

```text
bytes = words × 8
milliseconds = nanoseconds / 1 000 000
microseconds = nanoseconds / 1 000
GB/s = bytes / nanoseconds
```

The final formula works because a decimal GB contains `10^9` bytes and a second contains `10^9` nanoseconds. A MiB contains `2^20` bytes.

| `words` | Bytes | Readable size |
|---:|---:|---|
| 65,536 | 524,288 | 512 KiB |
| 1,048,576 | 8,388,608 | 8 MiB |
| 8,388,608 | 67,108,864 | 64 MiB |
| 16,777,216 | 134,217,728 | 128 MiB |

The result measures neither HBM traffic directly nor complete application throughput.

## Example at 128 MiB

For `pallas_carry_64`, one confirmation observation with seed 7 takes **1,364,480 ns**:

```text
1 364 480 ns = 1.36448 ms
134 217 728 / 1 364 480 = approximately 98.37 GB/s
```

That 98.37 GB/s value describes this observation. The median of the nine times for this seed is **1,462,309 ns**. The median of the seven per-seed medians is **1,499,840 ns**, or **89.488 GB/s**. The summary table uses this last throughput.

The difference between 98.37 and 89.49 therefore does not mean the table is wrong: the numbers summarize different sets of measurements.

## Why is the ratio not exactly the quotient of the displayed GB/s values?

Two different calculations are deliberately retained:

```text
displayed engine throughput
  = bytes / median_over_seeds(median_of_9_times)

paired candidate/reference ratio
  = median_over_seeds(reference_median_time / candidate_median_time)
```

The median of ratios is generally not the ratio of medians. Pairing compares both engines on the same seed.

A ratio of **1.2325** means approximately **23.25% more throughput** than the reference. The corresponding time reduction is `1−1/1.2325`, or about **18.86%**. A ratio below 1 means the candidate is slower.

`7/7` means a faster median for each of seven seeds. It does not mean seven independent hardware sessions and is not a p-value.

## Choose the right table

Use the [44 confirmation summaries](../../studies/tpu-opt-2026-10-21/results/audit-20261021T185102Z-tpu-opt/confirmation-summary.csv) for conclusions, followed by the selection rule in [results](results.md). Search selects; confirmation decides. The best individual observation across all trials is not automatically the configuration to adopt.
