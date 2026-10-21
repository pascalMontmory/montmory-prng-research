# Documentation en français et en anglais

[Français](languages.md) · [English](en/languages.md) · [Documentation FR/EN](languages.md)

La documentation de lecture actuelle est disponible dans les deux langues. Les deux parcours décrivent le même moteur, les mêmes résultats, les mêmes limites et les mêmes conditions de reproduction.

| Sujet | Français | English |
|---|---|---|
| Présentation | [Accueil](../README.fr.md) | [Overview](../README.md) |
| Prise en main d'une équipe | [Guide](start-here.fr.md) | [Guide](start-here.md) |
| Contrat du flux et propriétés | [Contrat](contract.md) | [Contract](en/contract.md) |
| Méthodologie | [Méthode](methodology.md) | [Methodology](en/methodology.md) |
| Résultats | [Résultats](results.md) | [Results](en/results.md) |
| Lecture des CSV | [Guide](csv-guide.md) | [Guide](en/csv-guide.md) |
| Registre des preuves | [Preuves](evidence.md) | [Evidence](en/evidence.md) |
| Reproduction | [Commandes](reproduce.md) | [Commands](en/reproduce.md) |
| TestU01 | [Registre](testu01.md) | [Register](en/testu01.md) |
| Diagnostics TACM | [Analyse](tacm.md) | [Analysis](en/tacm.md) |
| Limites | [Limites](limitations.md) | [Limitations](en/limitations.md) |
| Historique des corrections | [Historique](history.md) | [History](en/history.md) |
| Références | [Sources](references.md) | [Sources](en/references.md) |
| Glossaire | [Termes](glossary.md) | [Terms](en/glossary.md) |
| Benchmark applicatif JAX | [Protocole](benchmark-applicatif.md) | [Protocol](en/benchmark-applicatif.md) |
| Validation applicative locale | [Lecture](../applications/jax_rng/validation/README.md) | [Reading guide](../applications/jax_rng/validation/README.en.md) |
| Notebooks TPU | [Guide et originaux](../notebooks/README.fr.md) | [Guide and English editions](../notebooks/README.md) |
| Statut des droits | [Traduction de lecture](../LICENSE.fr.md) | [Texte de référence](../LICENSE) |
| Notices des tiers | [Lecture](../THIRD_PARTY_NOTICES.fr.md) | [Notices](../THIRD_PARTY_NOTICES.md) |

## Langue des pièces historiques

Les sources, identifiants de code, journaux, CSV, graphiques, classeur et études historiques restent dans leur langue d'origine. Leurs octets et leurs empreintes sont conservés. Les pages traduites expliquent ces pièces sans les réécrire ni transformer une ancienne réserve en résultat nouveau.

Les cellules explicatives des notebooks disposent d'éditions anglaises. Le code, les kits embarqués et les diagnostics restent identiques. Le protocole applicatif français déjà identifié par son empreinte dans le rapport CPU est conservé sans ajout de navigation ; sa traduction est accessible ci-dessus.

## Édition publique

Cette édition scientifique possède un historique Git neuf. Les documents juridiques privés et leurs traductions sont exclus ; ils restent dans un dossier privé distinct. Voir le [périmètre de publication](publication.md).

## Maintenir les deux langues

[translations.json](translations.json) recense chaque paire et les empreintes de cette révision. Après une modification, relire les deux versions, vérifier chiffres, réserves et commandes, puis mettre à jour uniquement les empreintes des paires revues. Une empreinte ne certifie pas la qualité linguistique.

```bash
python3 scripts/check_translations.py
python3 scripts/check_repository.py
```

Ces contrôles détectent fichiers manquants, changements non reportés dans le registre, liens locaux cassés et différences de code entre notebooks. Ils ne remplacent pas une relecture humaine et ne lancent aucun test statistique ou benchmark natif.
