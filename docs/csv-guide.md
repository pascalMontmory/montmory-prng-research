# Comprendre la campagne CSV

[Français](csv-guide.md) · [English](en/csv-guide.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Classeur de lecture](../results/Montmory_TPU_resultats_lisibles.xlsx) · [CSV brut](../studies/tpu-opt-2026-10-21/20261021T185102Z-tpu-opt/campaign/measurements.csv)

**Une ligne du CSV représente un appel chronométré, pas la performance finale d'un moteur.** Il contient 936 lignes de recherche et 2 772 lignes de confirmation. Comparer deux lignes isolées ne remplace pas la synthèse des répétitions.

## Colonnes

| Colonne | Signification |
|---|---|
| `phase` | `search` sélectionne les configurations ; `confirmation` les évalue sur de nouvelles seeds |
| `engine` | Implémentation précise ; par exemple `pallas_carry_64` est Montmory |
| `words` | Nombre de sorties de 64 bits, donc 8 octets par mot |
| `seed` | Graine exacte ; conserver cette colonne comme **texte** dans Excel |
| `block` | Bloc de mesures, numéroté à partir de zéro |
| `repetition` | Répétition dans le bloc, numérotée à partir de zéro |
| `nanoseconds` | Durée de l'appel jusqu'à la disponibilité du résultat ; plus petit = plus rapide |
| `output_sha256` | Empreinte de la sortie contrôlée hors chronométrage |
| `check` | Référence CPU fixe, ou seulement répétabilité sur le même backend pour RBG |

## Unités

```text
octets = words × 8
millisecondes = nanoseconds / 1 000 000
microsecondes = nanoseconds / 1 000
Go/s = octets / nanoseconds
```

La dernière formule fonctionne car un Go décimal vaut `10^9` octets et une seconde `10^9` nanosecondes. Un Mio vaut `2^20` octets.

| `words` | Octets | Volume lisible |
|---:|---:|---|
| 65 536 | 524 288 | 512 Kio |
| 1 048 576 | 8 388 608 | 8 Mio |
| 8 388 608 | 67 108 864 | 64 Mio |
| 16 777 216 | 134 217 728 | 128 Mio |

Le résultat n'est ni une mesure directe de trafic HBM ni un débit de calcul applicatif complet.

## Exemple à 128 Mio

Pour `pallas_carry_64`, une observation de confirmation avec la seed 7 vaut **1 364 480 ns** :

```text
1 364 480 ns = 1,36448 ms
134 217 728 / 1 364 480 = environ 98,37 Go/s
```

Ce nombre de 98,37 Go/s correspond à cette observation. La médiane des neuf temps de cette seed vaut **1 462 309 ns**. La médiane des sept médianes par seed vaut **1 499 840 ns**, soit **89,488 Go/s**. C'est ce dernier débit qui apparaît dans le tableau de synthèse.

L'écart entre 98,37 et 89,49 ne signifie donc pas que le tableau est faux : les nombres ne résument pas le même ensemble de mesures.

## Pourquoi le ratio ne correspond-il pas exactement au quotient des Go/s ?

Deux calculs différents sont volontairement conservés :

```text
débit affiché d'un moteur
  = octets / médiane_des_seeds(médiane_des_9_temps)

ratio apparié candidat/référence
  = médiane_des_seeds(temps_médian_référence / temps_médian_candidat)
```

La médiane des quotients n'est pas en général le quotient des médianes. L'appariement compare les deux moteurs sur la même seed.

Un ratio de **1,2325** signifie environ **23,25 % de débit supplémentaire** par rapport à la référence. La réduction de temps associée est `1−1/1,2325`, soit environ **18,86 %**. Un ratio inférieur à 1 signifie que le candidat est plus lent.

`7/7` signifie « médiane plus rapide pour chacune des sept seeds ». Ce n'est pas sept sessions matérielles indépendantes et ce n'est pas une p-value.

## Choisir le bon tableau

Pour la conclusion, utiliser les [44 synthèses de confirmation](../studies/tpu-opt-2026-10-21/results/audit-20261021T185102Z-tpu-opt/confirmation-summary.csv), puis la règle de sélection dans [les résultats](results.md). La recherche sert à choisir ; la confirmation sert à décider. Le meilleur chiffre observé parmi tous les essais n'est pas automatiquement la configuration à adopter.
