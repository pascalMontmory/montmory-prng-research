# Antériorités vérifiées — 21 octobre 2026

Question de Pascal : existe-t-il déjà un moteur massivement parallélisable,
produisant les mêmes bits sur ARM, x86, CUDA et Metal, avec succès aux batteries
SmallCrush, Crush et BigCrush ?

**L'affirmation générale « cela n'existe pas » ne tient pas.** Random123 constitue
un précédent direct, avec notamment Philox et Threefry. Il faut cependant
distinguer l'algorithme, ses ports, les tests de conformité et les rapports de
tests statistiques propres à chaque exécutable.

## Random123 : précédent principal, antérieur à l'archive Montmory de 2025

- Le [papier SC11 de 2011](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  décrit le calcul direct à partir d'un compteur et d'une clé pour les machines
  massivement parallèles, notamment les familles Philox et Threefry.
- Le [README figé](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/README.md)
  rapporte les succès SmallCrush, Crush et BigCrush et liste, dans ses plateformes
  testées, ARM, x86, CUDA et Metal. Le résumé initial CPU/CUDA/OpenCL ne mentionne
  pas toute la couverture détaillée plus bas : s'arrêter à ce résumé aurait
  conduit à manquer Metal.
- Le [test Metal](https://github.com/DEShawResearch/random123/blob/d8a0c25e49992685c0b9d91b553c1d04f81b6b4b/tests/kat_metal_kernel.metal)
  est présent dès le commit du dépôt daté du 17 janvier 2021. C'est une borne
  d'antériorité vérifiée, pas une affirmation sur la toute première publication
  historique de son support Metal.
- La bibliothèque contient des tests CUDA, Metal et CPU utilisant des
  [vecteurs de résultats connus](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors).
  L'identité des mots se vérifie à clé et compteur identiques, en conservant les
  mêmes conventions d'ordre et de conversion.
- L'[exemple CUDA de calcul de pi](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/examples/pi_cuda.cu)
  lance réellement un kernel sur une grille de blocs et utilise les identifiants
  des threads pour produire les tirages. Il s'agit d'une implémentation parallèle
  concrète, pas seulement d'une possibilité théorique de parallélisation.

**Précision pour le M3 : toutes les variantes ne sont pas activées en Metal.**
Dans la révision examinée, `include/Random123/features/metalfeatures.h` définit
`R123_USE_64BIT` à zéro par défaut et `tests/kat_metal.m` le force également à
zéro. Les tests Metal de cette révision ne valident donc pas Threefry4x64.
La remarque du fichier sur l'absence de 64 bits date de février 2019 ; elle ne
doit pas être appliquée comme une description des capacités actuelles du M3.
[Apple documente aujourd'hui les entiers Metal 64 bits](https://developer.apple.com/documentation/metal/mtldatatype/long).
Un port/une activation de Threefry4x64 demanderait sa propre compilation,
ses vecteurs de conformité et ses mesures sur le Mac. Rien ici ne démontre
son impossibilité sur M3, ni une validation native déjà accomplie.

Ces éléments établissent un précédent au cahier des charges général, notamment
avec les variantes 32 bits. Je n'ai
pas relancé les trois batteries sur chaque architecture de Random123 pendant cet
audit, ni récupéré une matrice complète de journaux bruts pour tous ses ports.
L'identité bit à bit du moteur entier ne signifie pas que toute API externe,
toute distribution dérivée et tout programme utilisateur ont des sorties identiques.

Depuis cette recherche, l'utilisateur a exécuté le kit comparatif sur son M3 :
les versions officielles Threefry2x32-20 et Philox4x32-10 compilent et s'exécutent
sous Metal, passent les vecteurs officiels et les comparaisons GPU/CPU, puis
produisent les résultats avec 8 192 et 65 536 threads. Le retour terminal reçu
apporte donc une confirmation native dans la machine cible pour ces variantes,
sans constituer un nouveau passage des batteries TestU01. Voir
[le résultat M3](README.md#premier-résultat-natif-apple-m3).

## Vérification des trois batteries pour les variantes mesurées

Nouvelle vérification documentaire le 21 octobre 2026, à la demande de Pascal.
Le README officiel cité ci-dessus rapporte explicitement le succès des trois
batteries. La [documentation des CBRNG](https://random123.com/releases/1.09/docs/CBRNG.html)
précise que les nombres de tours par défaut incluent une marge au-delà des
paramètres minimaux ayant passé SmallCrush, Crush et BigCrush.

| Variante du benchmark M3 | Paramètre par défaut officiel | Statut documentaire |
|---|---|---|
| Philox4x32-10 | Philox4x32_R<10> | Trois batteries rapportées réussies |
| Threefry2x32-20 | Threefry2x32_R<20> | Trois batteries rapportées réussies |

Les nombres de tours sont vérifiés dans les headers officiels figés, puis
dans les appels de `mac-metal-benchmark-v3/src/reference.hpp` et les kernels.
Voir aussi les documentations [Philox](https://random123.com/releases/1.09/docs/group__PhiloxNxW.html)
et [Threefry](https://random123.com/releases/1.09/docs/group__ThreefryNxW.html).
Le papier SC11 établit le précédent de ces familles ; la documentation des
variantes est nécessaire pour identifier précisément Threefry2x32-20, absent
du tableau de performances du papier consulté.

Le nombre de tours compte : les notes des sources signalent des résultats
suspects de Philox4x32 à sept tours dans des SimpPoker plus longs que ceux de
BigCrush, et un échec de Threefry2x64 à treize tours dans un WeightDistrib
prolongé. Ces notes ne concernent pas les deux variantes mesurées ici.
Le papier décrit aussi son protocole, dont une conversion utilisant 64 bits
d'entrée pour les tests ClosePairs ; il ne faut pas présenter ses expériences
comme une reproduction de chaque API et de chaque conversion utilisateur.

Cette vérification confirme des résultats publiés. Elle n'ajoute aucun nouveau
passage local des trois batteries pour Philox/Threefry sur le M3. Le kit local
en vérifie les vecteurs officiels et la concordance CPU/GPU ; le lanceur des
batteries actuellement exécuté est réservé à Montmory. Le succès TestU01 seul
ne constitue donc pas un avantage exclusif de Montmory, ni une garantie portant
sur toutes les seeds, toutes les répartitions ou tous les ports d'une famille.

## MLX : autre exemple concret pour Apple Silicon et CUDA

Les sources consultées au commit `52a6c234ab35ddf46c8bd1517b849b59b610d078`
contiennent le même cœur Threefry2x32 à 20 tours, avec les mêmes constantes et
opérations entières, pour [CPU](https://github.com/ml-explore/mlx/blob/52a6c234ab35ddf46c8bd1517b849b59b610d078/mlx/backend/cpu/threefry.cpp),
[Metal](https://github.com/ml-explore/mlx/blob/52a6c234ab35ddf46c8bd1517b849b59b610d078/mlx/backend/metal/kernels/random.metal)
et [CUDA](https://github.com/ml-explore/mlx/blob/52a6c234ab35ddf46c8bd1517b849b59b610d078/mlx/backend/cuda/random.cu).
Le [port CUDA initial](https://github.com/ml-explore/mlx/commit/7c4eb5d03e8ebfa2d7531e92db322e1b201f9127)
date du 10 juin 2025. Le fichier CPU cite expressément Threefry/Random123.
Le kernel Metal contient explicitement cinq groupes de quatre tours, soit
Threefry2x32-20. Cette variante produit deux mots de 32 bits par bloc, donc
64 bits au total : son nom n'indique pas une sortie limitée à 32 bits.

Ce constat vient de la lecture du code ; il n'est pas présenté comme une
campagne matérielle indépendante de MLX. Je n'ai pas trouvé dans les sources
consultées une certification TestU01 complète de chaque port MLX. Les succès
de la famille Threefry ne doivent pas être automatiquement attribués à toutes
les interfaces d'un framework, qui peuvent changer l'ordre ou les conversions.

## Conséquence pour Montmory

Le travail reste un générateur implémenté, accompagné de vrais artefacts de test,
dont une reproduction réussie sur x86. Il ne peut pas être distingué des travaux
existants par la seule combinaison parallélisme/portabilité/TestU01.

Une comparaison utile doit fixer le contrat des sorties et mesurer : le débit
sur CPU et GPU, le coût d'initialisation, la pression sur les registres et la
mémoire, la stabilité sous changement de partition, et le comportement Monte
Carlo sur plusieurs seeds et répartitions. Ces avantages ne sont pas démontrés
par le temps CPU d'un passage BigCrush.

Les premières mesures CPU sont désormais dans [README.md](README.md), avec leurs
options de compilation et leurs limites. Les comparatifs GPU de la sauvegarde
OVH ne constituent pas une mesure fiable contre Philox standard : leur version
Philox diverge dès le deuxième bloc et aucun journal de débit n'est inclus.

Les annonces publiques de Pascal retrouvées dans les résultats de recherche
ne sont pas utilisées comme confirmation indépendante des propriétés du moteur.
Cette recherche établit des précédents concrets ; elle n'est pas un inventaire
exhaustif de toute la littérature ni une analyse juridique de brevetabilité.
