# Public research edition

[Français](../publication.md) · [English](publication.md)

This repository publishes the scientific record of work by Pascal Montmory for ENINCA CONSULTING. It was prepared from research revision `4d8a1a9316f6672d32949e8a4d7022006433fdcf` on 21 October 2026, with a new Git history. The private source repository and its history remain private; no private legal documents are included.

All **884 imported research files** retain their recorded bytes and SHA-256 hashes. Sources, benchmark data, historical logs, third-party notices and embedded notebook kits are preserved. Documentation, clone URLs and the translation checker were adapted to this public edition. The public [translation registry](../translations.json) lists only the included reading editions.

Public visibility does not activate a new general license for original contributions. The [rights notice](../../LICENSE) and [third-party notices](../../THIRD_PARTY_NOTICES.md) describe their respective scope.

Start with [results](results.md), [methodology](methodology.md), [reproduction](reproduce.md) and the [TPU notebooks](../../notebooks/README.md). The initial 21 October publication ran no new native hardware benchmark. The 24 October addition retains a [new TPU dropout training archive](../../applications/dropout/validation/tpu-20261024/README.md), received from Colab and audited on CPU. Cite this public repository and the exact public commit used.

```bash
python3 scripts/check_repository.py
python3 scripts/check_translations.py
python3 scripts/check_reference.py
```
