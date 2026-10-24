# Résultats au 24 octobre 2026

[Français](results.md) · [English](en/results.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Méthodologie](methodology.md) · [Lire les chiffres](csv-guide.md)

## TPU : une optimisation confirmée, avec un comparateur exigeant

Campagne native `20261021T185102Z-tpu-opt`, **un TPU v5 lite**, JAX/jaxlib 0.10.2, géométrie `T=2048,D=4096`, départ `2^32−4097`. Les débits ci-dessous sont calculés à partir de la médiane des sept temps médians par seed ; chaque seed possède neuf mesures de confirmation.

| Implémentation | 512 Kio | 8 Mio | 64 Mio | 128 Mio |
|---|---:|---:|---:|---:|
| **Montmory retenu** | **2,120** | **15,859** | **62,458** | **89,488** |
| Montmory Pallas précédent | 2,120 | 15,859 | 54,200 | 73,498 |
| Montmory JAX portable | 2,026 | 15,138 | 51,075 | 66,769 |
| Threefry2x32 JAX | 1,979 | 12,982 | 31,954 | 33,364 |
| Threefry2x32 Pallas sélectionné | 2,013 | 15,917 | 65,249 | 93,006 |
| Threefry4x32 JAX | 1,921 | 10,691 | 20,396 | 21,331 |
| Philox2x32 JAX | 1,885 | 10,650 | 20,302 | 20,925 |
| Philox4x32 JAX | 1,664 | 7,867 | 12,193 | 12,463 |
| RBG, même backend seulement | 2,096 | 18,465 | 86,811 | 90,791 |
| Écriture, non PRNG | 2,131 | 17,790 | 91,322 | 151,754 |

Unité : **Go/s décimaux** ; volumes : Kio/Mio binaires. [Données et audit complets](../studies/tpu-opt-2026-10-21/results/audit-20261021T185102Z-tpu-opt/README.md).

| Volume | Configuration conservée | Gain apparié sur l'ancien Pallas | Motif |
|---|---|---:|---|
| 512 Kio | `pallas_original_32` | 0 % | Candidat principal : −0,63 %, seulement 2/7 seeds gagnantes |
| 8 Mio | `pallas_original_32` | 0 % | Candidat principal : +1,64 %, 6/7 ; seuil non atteint |
| 64 Mio | `pallas_carry_128` | +15,42 % | Seuil atteint ; 7/7 seeds et 21/21 couples seed/bloc gagnants |
| 128 Mio | `pallas_carry_64` | +23,25 % | Seuil atteint ; 7/7 seeds et 21/21 couples seed/bloc gagnants |

Les gains appariés ne sont pas les quotients des débits arrondis du premier tableau. À 128 Mio, +23,25 % de débit représente environ **18,86 % de temps économisé** selon ce rapport apparié.

Face à Threefry2x32 JAX, les rapports appariés Montmory sont ×1,926 à 64 Mio et ×2,692 à 128 Mio. Face à **Threefry Pallas**, ils sont respectivement 0,960 et 0,961 : Montmory reste environ 4 % moins rapide. Ce comparateur Pallas conserve les vingt tours et les sorties de l'API Threefry2x32 JAX utilisée. Aucun Philox Pallas spécialisé n'a été mesuré ; les rapports aux API JAX Philox ne permettent pas de classer toutes les implémentations possibles.

Les premières mesures TPU, dans la [campagne 18:06](../studies/jax-tpu-2026-10-21/README.md), ne comprenaient pas ce comparateur Threefry Pallas. Elles ne suffisaient donc pas à annoncer une supériorité générale.

## PRNG matériel et comparateur RBG mesuré

L'objectif des noyaux logiciels est de nous approcher du débit utile de génération tout en conservant un flux entier défini et portable. Le [README](../README.fr.md#pourquoi-optimiser-un-prng-logiciel-sur-tpu-) explique le compromis avec le PRNG matériel.

Dans cette campagne, `rbg_same_backend` désigne précisément `jax.random.key(..., impl="rbg")`, puis `jax.random.bits(..., dtype=uint32)` ; voir les fonctions `arguments` et `function` dans [kernels.py](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py). Ce chemin utilise le générateur RBG de XLA. JAX ne lui attribue pas la même garantie de sorties identiques CPU/GPU/TPU qu'à Threefry. [Configuration des PRNG JAX](https://docs.jax.dev/en/latest/jax.random.html#advanced-rng-configuration).

Les débits du tableau comparent le temps de l'appel jusqu'au résultat prêt, allocation et écritures comprises. À 128 Mio, Threefry Pallas atteint 93,006 Go/s, Montmory Pallas 89,488 et RBG 90,791. À 64 Mio, les valeurs sont respectivement 65,249, 62,458 et 86,811. La proximité observée à 128 Mio ne se généralise donc pas à tous les volumes.

**Aucun noyau dédié aux appels matériels Pallas `prng_seed`/`prng_random_bits` n'a été chronométré dans cette campagne.** Le chiffre RBG ne représente ni le débit maximal du circuit, ni une comparaison directe avec toutes les voies matérielles possibles. Un débit logiciel supérieur à ce chemin RBG ne prouve pas que ses opérations élémentaires sont plus rapides que celles du PRNG silicium.

Les sorties Montmory et Threefry sont contrôlées contre leurs références définies ; RBG est contrôlé seulement en répétabilité sur le même backend. Ce choix explicite le compromis recherché : performance et stabilité du flux. La stabilité repose sur un contrat logiciel ; ces campagnes de génération brute ne démontrent pas son respect sur plusieurs générations de TPU, ni à elles seules un gain applicatif. L’entraînement dropout est mesuré séparément ci-dessous.

## Apple M3 : conserver le meilleur chemin mesuré

La première archive Metal d'optimisation reçue est [auditée ici](../studies/metal-opt-2026-10-21/results/mac-analysis-20261021T162407Z-22542/analysis.json). Elle contient 144 observations CPU et 2 574 GPU. L'analyse détecte un effet de première utilisation sur certaines variantes spécialisées. Une campagne séparée a ensuite précompilé et chauffé tous les pipelines.

La **confirmation séparée ci-dessous est un résumé terminal transmis**, avec sept nouvelles seeds et quinze mesures par seed. Son archive brute complète n'a pas été reçue dans ce dossier.

| Metal raw64, 64 Mio produits | Go/s affichés | Rapport apparié à Montmory v3 |
|---|---:|---:|
| Montmory `mulhi_pair` | 84,787 | 1,000 |
| Montmory `split_stripe_pair` | 85,218 | 1,005 |
| Philox4x32-10 | 88,979 | 1,049 |
| Threefry2x32-20 | 92,082 | 1,086 |
| Écriture, non PRNG | 96,018 | 1,133 |

[Résumé et provenance](../studies/metal-opt-2026-10-21/results/mac-confirm-reported-20261021T163954Z-22644.json). Threefry est ici **moins rapide que le témoin d'écriture**, et plus rapide que Montmory.

Les essais vectoriels Metal ultérieurs n'améliorent pas ce bilan : `vector4` est autour de 76,6 Go/s contre 84,8 pour `mulhi_pair`. `split_stripe_pair` gagne environ 0,4 %. Ces derniers résultats sont également [rapportés par terminal](../studies/vector-opt-2026-10-21/README.md), sans archive native complète reçue. La vectorisation explicite n'est donc pas retenue comme un progrès automatique.

## CPU : le coût de lancement compte

Sur M3, l'archive initiale confirme l'intérêt du calcul scalaire par paires : 12,487 Go/s à huit workers contre 9,746 Go/s pour la variante indexée, avec sorties identiques. Le gain apparié est d'environ 28 %.

Sur les petits blocs de 65 536 mots, la confirmation terminal du pool de quatre workers rapporte un gain apparié de **30,7 %** par rapport à la création de quatre workers à chaque appel, sur sept seeds. Sur les grands blocs, le gain du pool est beaucoup plus petit. [Sources, protocole et résultats vectoriels](../studies/vector-opt-2026-10-21/README.md).

Sur le VPS x86, le [test de répartition physique](../studies/lanes-study-2026-10-21/README.md) donne 2,02 / 3,87 / 7,38 / 3,35 / 3,09 Go/s avec 1 / 2 / 4 / 8 / 16 workers. Le meilleur réglage de cette session est quatre workers. Cela ne constitue pas une règle pour tous les processeurs et ne modifie pas les lanes logiques.

La [comparaison CPU Random123](../studies/audit-2026-10-21/evidence/benchmark-summary.json) change de classement selon compilation, accès contigus ou dispersés et moteur comparé. Elle n'établit pas de domination universelle de Montmory.

## Qualité statistique et applications

Le [registre TestU01](testu01.md) précise les passes disponibles et leurs limites de provenance. Les [comparaisons TACM](tacm.md) ne montrent aucun avantage statistique de Montmory selon le critère fixé. Les optimisations TPU n'ont modifié aucune constante ni aucun tour du moteur, mais n'ont pas relancé TestU01 ou TACM.

Un petit entraînement avec dropout dispose désormais de mesures natives TPU, présentées ci-dessous. Aucun gain natif sur un calcul VaR complet, une application de tarification ou la consommation énergétique n’est établi. Ces preuves ne chiffrent pas un bénéfice commercial.

Un [premier benchmark applicatif JAX](benchmark-applicatif.md) est désormais disponible : uniformes, normales et estimation Monte-Carlo avec le consommateur dans le même `jit`. Son contrôle court CPU est conservé séparément. Il n'apporte pas encore de mesure applicative native TPU ni de confirmation d'un gain sur une application cliente.

## Dropout : entraînement natif TPU du 24 octobre 2026

Le [rapport détaillé](../applications/dropout/validation/tpu-20261024/README.fr.md)
conserve 1 800 mesures, les sources exactes, les contrôles et les courbes d’un MLP
à 21 252 paramètres sur TPU v5 lite. Pallas réduit le temps d’étape de Montmory
par rapport à Montmory JAX de **5,76 %, 5,10 % et 6,27 %** sur les trois graines.
Les 900 paires de pertes JAX/Pallas sont numériquement identiques ; le programme
rapporte aussi un écart nul entre les poids finaux, non exportés dans l’archive.

Montmory Pallas reste proche de Threefry JAX : +0,75 %, −0,89 % et −0,98 % de temps
par étape. Threefry Pallas prend environ 0,3–0,5 % de temps supplémentaire face à
Threefry JAX. Le passage à Pallas change l’implémentation, pas l’algorithme Threefry ;
le PRNG matériel TPU n’est pas utilisé ici.

La précision finale moyenne vaut 82,55 % pour Montmory et 83,01 % pour Threefry.
Une session, trois graines et une tâche synthétique ne démontrent ni supériorité
ni équivalence statistique de qualité. Le rapprochement avec l’ancien contrôle
CPU donne environ 1,48× pour Montmory Pallas, mais reste indicatif et ne mesure
pas un gain propre au PRNG. Aucun entraînement GPU comparable n’est disponible.

Le [prototype Sobol32](../applications/sobol/README.fr.md) et ses notebooks TPU
sont distincts. Leur compilation native et leurs performances TPU restent à valider.
