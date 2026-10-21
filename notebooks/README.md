# Native TPU notebooks

[Français](README.fr.md) · [English](README.md) · [Documentation FR/EN](../docs/en/languages.md)

| Campaign | English | Français |
|---|---|---|
| Optimization, search and held-out confirmation | [Runbook](Montmory_TPU_optimisation.en.ipynb) | [Original](Montmory_TPU_optimisation.ipynb) |
| Initial port validation and measurements | [Runbook](Montmory_TPU_validation.en.ipynb) | [Original](Montmory_TPU_validation.ipynb) |

Start with the optimization notebook for the most recent protocol. Download it, open it in Colab, select an actual **TPU** runtime, and run its code cells in order. Access depends on your account and quotas. The notebook embeds the executable kit; no GitHub credential is needed after download. Keep each new result archive separately.

The English editions translate explanatory Markdown cells only. Every executable cell, embedded kit, recorded output and notebook metadata value is identical to its French original. Diagnostic messages therefore remain in their original language. `python3 scripts/check_translations.py`, run from the repository root, checks this equivalence without executing the campaign.

The initial-validation notebook describes what was known when it was prepared. Its statement that native TPU results were still pending is historical; the [current evidence register](../docs/en/evidence.md) documents the subsequent native campaigns. Original notebooks and frozen kits are retained byte for byte.

Read the [reproduction guide](../docs/en/reproduce.md), [results](../docs/en/results.md) and [CSV guide](../docs/en/csv-guide.md) before interpreting a run. Translating these notebooks does not rerun benchmarks or statistical batteries and does not grant new usage rights.
