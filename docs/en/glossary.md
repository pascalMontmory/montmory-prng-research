# Reading glossary

[Français](../glossary.md) · [English](glossary.md) · [Documentation FR/EN](languages.md)

[Home](../../README.md) · [CSV guide](csv-guide.md)

| Term | Meaning in this repository |
|---|---|
| PRNG | Deterministic algorithm producing a sequence intended for pseudorandom uses |
| Seed | Initial parameter; it does not describe the entire stream contract by itself |
| `raw64` | Unsigned 64-bit integer word |
| `mantissa53` | The upper 53 bits of a word retained as an integer; multiply by `2^-53` for the prescribed binary64 uniform |
| Conformance | Agreement between an implementation and the reference for checked inputs |
| Statistical quality | Behavior of the stream against defects targeted by specified tests |
| Calibration | Checking a diagnostic's thresholds on controls; not tuning a PRNG to obtain a favorable p-value |
| MSE | Mean squared error of an estimator relative to its reference value |
| Bias | Difference between an estimator's expectation and its target; finite estimates fluctuate |
| Logical lane | Identifier used in the engine formula; not a physical core |
| Worker | CPU execution task; worker count can change without changing the stream if indices are preserved |
| Pallas | JAX API for specialized kernels; here, a TPU implementation of the same integer computation |
| Tile | Block of outputs processed together by a Pallas kernel |
| Search / confirmation | Selection on an initial seed set followed by evaluation on different seeds |
| Paired ratio | Ratio of two engines' times for one seed, then summarized over seeds |
| SHA256 hash | Deterministic digest used to check preservation or agreement of data |
| RBG | JAX comparator handled separately; repeatability checked on the same backend only |
| Write baseline | Program issuing memory writes for comparison; it does not generate a PRNG stream |

Passing conformance checks, passing a statistical battery and improving throughput answer three different questions. None replaces the other two.
