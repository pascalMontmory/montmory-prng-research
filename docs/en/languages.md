# Documentation in English and French

[Français](../languages.md) · [English](languages.md) · [Documentation FR/EN](languages.md)

The current reader documentation is available in both languages. Both reading paths describe the same engine, results, limitations and reproduction conditions.

| Topic | English | Français |
|---|---|---|
| Overview | [Home](../../README.md) | [Présentation](../../README.fr.md) |
| Team onboarding | [Guide](../start-here.md) | [Guide](../start-here.fr.md) |
| Stream contract and properties | [Contract](contract.md) | [Contrat](../contract.md) |
| Methodology | [Methodology](methodology.md) | [Méthode](../methodology.md) |
| Results | [Results](results.md) | [Résultats](../results.md) |
| Reading CSV files | [Guide](csv-guide.md) | [Guide](../csv-guide.md) |
| Evidence register | [Evidence](evidence.md) | [Preuves](../evidence.md) |
| Reproduction | [Commands](reproduce.md) | [Commandes](../reproduce.md) |
| TestU01 | [Register](testu01.md) | [Registre](../testu01.md) |
| TACM diagnostics | [Analysis](tacm.md) | [Analyse](../tacm.md) |
| Limitations | [Limitations](limitations.md) | [Limites](../limitations.md) |
| Correction history | [History](history.md) | [Historique](../history.md) |
| References | [Sources](references.md) | [Sources](../references.md) |
| Glossary | [Terms](glossary.md) | [Termes](../glossary.md) |
| JAX application benchmark | [Protocol](benchmark-applicatif.md) | [Protocole](../benchmark-applicatif.md) |
| Local application validation | [Reading guide](../../applications/jax_rng/validation/README.en.md) | [Lecture](../../applications/jax_rng/validation/README.md) |
| TPU notebooks | [Guide and English editions](../../notebooks/README.md) | [Guide et originaux](../../notebooks/README.fr.md) |
| Rights status | [Reference text](../../LICENSE) | [Reading translation](../../LICENSE.fr.md) |
| Third-party notices | [Notices](../../THIRD_PARTY_NOTICES.md) | [Lecture](../../THIRD_PARTY_NOTICES.fr.md) |

## Historical materials' language

Sources, code identifiers, logs, CSV files, figures, the workbook and historical studies remain in their original language. Their bytes and hashes are preserved. Translated pages explain these materials without rewriting them or turning an earlier limitation into a new result.

Notebook explanatory cells have English editions. Code, embedded kits and diagnostics remain identical. The French application protocol already identified by hash in the CPU report is preserved without adding navigation; its translation is accessible above.

## Public edition

This scientific edition has a new Git history. Private legal documents and their translations are excluded; they remain in a separate private record. See [publication scope](publication.md).

## Maintaining both languages

[translations.json](../translations.json) lists each pair and the hashes of this revision. After an edit, review both versions, checking figures, limitations and commands, then update only the hashes of reviewed pairs. A hash does not certify linguistic quality.

```bash
python3 scripts/check_translations.py
python3 scripts/check_repository.py
```

These checks detect missing files, changes not reflected in the registry, broken local links and notebook code differences. They do not replace human review or run statistical tests or native benchmarks.
