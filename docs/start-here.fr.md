# Prise en main : évaluer les implémentations CPU et TPU

[Français](start-here.fr.md) · [English](start-here.md) · [Documentation FR/EN](languages.md)

[Présentation des travaux](../README.fr.md) · [Commandes de reproduction](reproduce.md) · [Résultats mesurés](results.md)

Cette page s'adresse à une équipe technique qui examine le code. Le dépôt contient deux implémentations TPU : **Montmory**, conservant le flux Montmory documenté, et **Threefry2x32 à vingt tours**, implémenté en Pallas comme comparateur exigeant face à l'API JAX. Ce sont des générateurs différents. Ce dépôt rassemble des kits de recherche figés et exécutables ; ce n'est pas un SDK installé ni un plugin PRNG JAX achevé.

Ce dossier de recherche est accessible publiquement. La publication n'active aucune licence générale sur les contributions originales ; obtenir une autorisation adaptée pour les usages qui en nécessitent une. Les permissions des tiers conservent leur périmètre. Voir le [statut des droits](../LICENSE.fr.md).

## Premier contrôle : aucun compte cloud ni accélérateur requis

Depuis une machine disposant de Python 3.11+ et d'un compilateur C prenant en charge UBSan (`cc`, GCC ou Clang) :

```bash
git clone https://github.com/pascalMontmory/montmory-prng-research.git
cd montmory-prng-research
python3 scripts/check_repository.py
python3 scripts/check_reference.py
```

Les deux commandes doivent se terminer avec succès et afficher `"status": "pass"`. La première vérifie les fichiers importés figés et la structure des preuves. La seconde compile l'oracle C, compare **1 600 vecteurs** au C et au Python et reproduit les certificats mathématiques exacts, y compris les limites documentées. Les nouveaux rapports vont dans `build/`. Ce sont des contrôles CPU locaux, pas une nouvelle exécution TestU01 ou TPU.

Sur macOS, installer les outils de ligne de commande Xcode si aucun compilateur n'est disponible. Si une dépendance ne s'installe pas, utiliser les environnements séparés de [reproduce.md](reproduce.md) : les anciens environnements TACM et JAX ont des versions épinglées différentes.

## Trouver l'implémentation utile

| Objectif | Fichier à lire | Points à examiner |
|---|---|---|
| Comprendre l'algorithme Montmory scalaire | [mmc_core.c](../src/reference/mmc_core.c), [en-tête](../src/reference/mmc_core.h), [modèle Python entier](../src/reference/reference.py) | Incrément d'état, trois tours Feistel, mélangeur final ; la référence fixe la sémantique. |
| Comprendre la séquence globale | [Contrat du flux](contract.md) | `(version du moteur, seed, indice global, T, D, format)` ; les lanes logiques influencent le flux. |
| Comprendre l'arithmétique JAX portable | [montmory_jax.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/montmory_jax.py) | Arithmétique exacte sur 64 bits représentée par des paires de uint32. |
| Examiner les noyaux Montmory TPU optimisés | [kernels.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py) | `small_product`, `round_carry`, `stream` et `pallas_range` ; les tuiles modifient l'exécution, pas le flux attendu. |
| Examiner le comparateur Threefry TPU | [Même source des noyaux](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py) | `threefry_bits_at` et la voie `threefry=True` ; vingt tours, convention de sortie uint32 partitionable de JAX. |
| Appeler Montmory et appliquer une sélection TPU mesurée | [tuned.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/tuned.py) | `generate` et `load_tpu_selection` ; correspondance explicite de JAX/jaxlib et du type de TPU. |
| Examiner sélection et mesures | [campaign.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/campaign.py) | Seeds distinctes de recherche/confirmation, précompilation, attentes de complétion, contrôles hors chronométrage. |

L'API C maintient un état **par lane logique**. L'interface JAX est indexée sur le flux global documenté. Le nombre de workers physiques ne remplace pas le paramètre de lane ; ces API ne sont pas interchangeables sans la correspondance définie dans le contrat du flux.

## Appel Montmory minimal sur JAX CPU

Créer l'environnement JAX épinglé selon [reproduce.md, section 4](reproduce.md#4-vérifier-le-port-jax-localement). Depuis la racine du dépôt, utiliser son interpréteur :

```bash
JAX_PLATFORMS=cpu .venv-jax/bin/python - <<'PY'
from pathlib import Path
import runpy
import sys

kit = Path('studies/tpu-opt-2026-10-21/montmory-tpu-opt').resolve()
sys.path.insert(0, str(kit))
import tuned

low, high = tuned.generate(seed=42, base=0, count=4)
words = [int(low[i]) | (int(high[i]) << 32) for i in range(4)]
reference = runpy.run_path('src/reference/reference.py')
assert words == [reference['at'](42, 0, i + 1) for i in range(4)]
print([f'{word:016x}' for word in words])
PY
```

Les tableaux renvoyés sont les **moitiés uint32 basse/haute**, pas des uniformes flottants. La reconstruction Python emploie des entiers de précision arbitraire et ne nécessite pas d'activer uint64 dans JAX. Le compteur de tirages de la référence par lane commence à 1 ; `base` commence à 0. Ces quatre positions appartiennent à la lane logique 0 avec `T=2048,D=4096`. Lire le contrat complet avant d'étendre cette correspondance à d'autres positions.

L'exemple a été exécuté dans l'environnement local JAX 0.10.2 CPU et a affiché :

```text
['df2a584382db0894', 'c5d3386642e0256b', 'ec5078b8efb699cc', 'eaa6f5b8e64a247f']
```

Sans sélection native explicitement chargée, `generate` utilise la voie portable. `load_tpu_selection` accepte une campagne native réussie correspondant au type de périphérique et aux versions logicielles. Une sélection enregistrée ne s'applique qu'aux volumes mesurés ; ce n'est pas un réglage matériel universel automatique.

## Exécuter l'étude TPU ou l'auditer sans TPU

Pour une nouvelle campagne **TPU native**, utiliser le [notebook d'optimisation](../notebooks/Montmory_TPU_optimisation.ipynb), choisir un véritable environnement TPU et suivre [reproduce.md, section 5](reproduce.md#5-relancer-la-campagne-native-tpu). Il contient les candidats Montmory et Threefry/Pallas. Conserver la nouvelle archive de résultats et son environnement. Le matériel précédemment enregistré était un TPU v5 lite ; ces résultats ne garantissent ni compilation ni vitesse sur une autre génération.

Pour une **réanalyse CPU des preuves TPU existantes**, installer NumPy dans l'environnement d'audit et lancer `python3 scripts/replay_audit.py tpu`. Cela revérifie 3 708 mesures archivées sans exécuter de TPU. Le [guide CSV](csv-guide.md) explique la différence entre Go/s affichés et rapports de performance appariés.

La voie Threefry Pallas implémente la convention de sortie JAX spécifiée pour les formes enregistrées et les compteurs inférieurs à `2^32`. Elle n'est pas présentée comme un remplacement général de toutes les API JAX `random`, plages de compteurs, ou sémantiques `split`/`fold_in`. Le prototype Montmory n'enregistre pas non plus de nouvelles sémantiques validées de séparation de clés JAX.

## Interpréter une évaluation réussie

Un contrôle local réussi établit l'accord pour les entrées testées. Une campagne native réussie ajoute une performance mesurée pour ce matériel, ce logiciel, cette forme et ce périmètre chronométré. Cela n'implique ni sécurité cryptographique, ni uniformité exacte, ni indépendance de toutes les lanes, ni gain applicatif. Les [limites](limitations.md) et le [registre des preuves](evidence.md) distinguent tests finis, certificats exacts, audits de données brutes et résumés terminal.

Pour un rapport d'évaluation, conserver commit, empreintes, appareil, versions compilateur/JAX, seed/indice/géométrie, format, mesures individuelles et résultats des contrôles. Comparer les deux concurrents optimisés sur la même charge, pas seulement une référence non optimisée. Les résultats actuels aux grands volumes TPU sont d'environ **89,49 Go/s pour Montmory** et **93,01 Go/s pour Threefry Pallas**, dans une session enregistrée.

Les fichiers de `studies/` sont des instantanés historiques. Lire d'abord les pages de synthèse et employer une copie sous `build/` pour une nouvelle campagne. Ne pas modifier un noyau figé pour faire passer un ancien manifeste.
