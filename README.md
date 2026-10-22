# Montmory PRNG — reproducible research record

[Français](README.fr.md) · [English](README.md) · [Documentation FR/EN](docs/en/languages.md)

[Start here: code and first checks](docs/start-here.md) · [Results](docs/en/results.md) · [Methodology](docs/en/methodology.md) · [Reproduce](docs/en/reproduce.md) · [Evidence register](docs/en/evidence.md)

**A documented investigation of one deterministic integer generator across CPU, Apple Metal GPU and Google TPU.** This repository preserves the implementation, protocols, raw measurements, independent reanalysis, corrections and known limitations of work completed on **21 October 2026**. Algorithm development is paused.

## Why optimize a software PRNG on TPU?

TPUs include a fast sequential hardware PRNG whose implementation varies across chip generations. JAX advises against relying on its exact behavior and recommends software Threefry when stream stability matters. [JAX hardware PRNG documentation](https://docs.jax.dev/en/latest/pallas/tpu/prng.html#using-the-hardware-prng).

**Our objective is to approach the accelerator's useful generation throughput while retaining an exactly specified, portable integer stream.** We optimize software implementations of Threefry and Montmory so that execution strategy can change while the defined outputs remain fixed. The stream contract is specified in software rather than by the TPU generation. Native TPU evidence currently covers **one TPU v5 lite**; validation on other generations remains future work.

At 128 MiB, this campaign reports **93.006 GB/s for Threefry Pallas, 89.488 GB/s for Montmory Pallas and 90.791 GB/s for JAX/XLA RBG**. The recorded RBG path is an API comparator, not a direct measurement of the hardware PRNG's maximum rate. This comparison motivates the work without claiming that software matches the silicon generator in every workload. [Hardware PRNG, measured RBG and comparison scope](docs/en/results.md#hardware-prng-and-the-measured-rbg-comparator).

This work is conducted for **ENINCA CONSULTING by its founder and president, Pascal Montmory**. It covers the Montmory generator and optimized implementations of the pre-existing Threefry algorithm, with provenance and reproducible evaluation procedures.

**Languages:** the current documentation and notebook instructions are available in English and French; use the language links above. Historical source files, logs and data retain their original bytes. Private legal documents and their Git history are excluded from this public edition. See the [language guide](docs/en/languages.md).

The central engineering result is preservation of the reference output while changing execution strategy. The stream contract includes the **engine version, seed, absolute index and logical lane geometry**. It is more specific than “the same seed gives the same numbers”. See the [exact contract](docs/en/contract.md).

## What the evidence establishes

| Question | Result | Evidence and scope |
|---|---|---|
| Can the reference stream be reproduced? | C reference vectors agree with tested ARM64, x86, Metal and TPU paths. | [Conformance matrix](docs/en/evidence.md); finite comparisons, not a proof for all inputs or future hardware. |
| Does it pass TestU01? | Historical SmallCrush, Crush and BigCrush logs report passes. The native M3 v3 run also reports all three passes. | [TestU01 record](docs/en/testu01.md). Historical complete logs are included; the latest M3 v3 BigCrush result is a **terminal summary**, not a received complete archive. |
| Did TPU optimization help? | Confirmed paired throughput gains of **15.42% at 64 MiB** and **23.25% at 128 MiB** against the previous Montmory Pallas kernel. | One TPU v5 lite, JAX/jaxlib 0.10.2, seven held-out seeds; [3,708 measurements audited](studies/tpu-opt-2026-10-21/results/audit-20261021T185102Z-tpu-opt/README.md). |
| Is Montmory faster than every comparator? | **No.** Optimized Threefry Pallas is about 4% faster at those large TPU sizes. Threefry and Philox also lead the reported M3 comparison. | [Comparable configurations and timing boundaries](docs/en/results.md). |
| Do complementary TACM diagnostics show a statistical advantage? | **No advantage established in 54 prespecified MSE comparisons.** | Ten engines, 2,048 seeds per engine, six workloads; [corrected TACM study](docs/en/tacm.md). |
| Is exact uniformity or cryptographic security established? | **No.** The final mixer is non-injective; exact counterexamples are included. | [Mathematical scope](docs/en/contract.md) and [limitations](docs/en/limitations.md). |

![Native TPU confirmation, four output sizes and matched comparators](studies/tpu-opt-2026-10-21/results/audit-20261021T185102Z-tpu-opt/confirmation-tpu.png)

The figure measures output bytes divided by caller elapsed time. It is not a hardware memory-bandwidth specification or an end-to-end Monte Carlo speedup. [Full table and interpretation](docs/en/results.md).

## Read the work in order

1. [Contract and exact mathematics](docs/en/contract.md): what must stay unchanged.
2. [Methodology](docs/en/methodology.md): correctness, calibration, selection and confirmation.
3. [Results](docs/en/results.md): positive, negative and inconclusive findings.
4. [Understanding the CSV](docs/en/csv-guide.md): units, medians, paired ratios and examples.
5. [Evidence register](docs/en/evidence.md): each claim linked to its actual supporting files.
6. [Reproduction](docs/en/reproduce.md): local audits without a GPU/TPU, then native campaigns.
7. [Limitations](docs/en/limitations.md), [correction history](docs/en/history.md) and [prior work](docs/en/references.md).

The [readable spreadsheet](results/Montmory_TPU_resultats_lisibles.xlsx) contains all 3,708 TPU observations and their summaries. It supplements the CSV; it does not replace it.

A [glossary](docs/en/glossary.md) explains the terminology used in the protocols and tables.

The new [JAX application benchmark](docs/en/benchmark-applicatif.md) covers uniforms, normals and a complete Monte Carlo estimator. Local CPU checks are recorded separately; native TPU application performance is not yet measured.

The separate [Sobol32 prototype](applications/sobol/README.md) studies exact low-discrepancy point generation with both JAX and Pallas block implementations. CPU correctness is checked; native TPU compilation and performance remain unverified.

The [dropout training experiment](applications/dropout/README.md) compares Montmory and Threefry in a small neural network, with mask generation and application fused in Pallas. CPU checks, learning curves and English/French Colab notebooks are available. **Native TPU compilation and any training speedup remain to be validated.**

## Verify this record locally

Python 3.11+ and a C compiler are sufficient for the first two checks:

```bash
git clone https://github.com/pascalMontmory/montmory-prng-research.git
cd montmory-prng-research
python3 scripts/check_repository.py
python3 scripts/check_reference.py
```

Replay the TPU data audit on a CPU, with NumPy installed:

```bash
python3 scripts/replay_audit.py tpu
```

This rechecks recorded data and rebuilds the C oracle. It **does not execute or benchmark a TPU**. [Dependencies and other audits](docs/en/reproduce.md).

The [repository verification report](results/repository-validation.json) records the local checks and four successful CPU reanalyses performed when assembling this repository.

**Public edition:** this repository has a new history containing the scientific record only. [Publication scope](docs/en/publication.md) · [GitHub Actions](https://github.com/pascalMontmory/montmory-prng-research/actions). Historical local reports remain separate from any new hosted CI result.

## Repository layout

```text
src/reference/   unchanged C/Python reference and 1,600 golden vectors
applications/    new application benchmark, CPU checks and separate evidence
docs/            curated explanation, methodology, results and limitations
evidence/        claim index, imported-file hashes, provenance and exclusions
studies/         frozen sources, protocols, raw data and historical analyses
notebooks/       standalone native TPU validation and optimization notebooks
results/         readable workbook and repository verification report
scripts/         integrity, exact-reference and CPU reanalysis entry points
third_party/     dependency notices; original notices also remain in each kit
```

Start with `docs/`, not with a historical study README. Files under `studies/` preserve what was known when each study was written; some contain earlier “pending” conclusions or paths from the original machine. The [current evidence register](docs/en/evidence.md) resolves those differences.

This is a research record, not a production SDK, a cryptographic generator, a JAX upstream contribution, or evidence of banking/model approval. No new license grant is made for original Montmory work. Dependency terms are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite the repository **and the exact commit**; see [CITATION.cff](CITATION.cff).
