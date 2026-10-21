# Audit du PRNG Montmory reçu du Mac — 21 octobre 2026

## Verdict

Le moteur reçu est un véritable générateur à compteur, avec trois tours Feistel
et un mélange de sortie. Il se prête au calcul parallèle et à une spécification
exacte commune aux architectures. Les résultats statistiques retrouvés sont
substantiels et une première reproduction indépendante réussit.

La combinaison « massivement parallèle + mêmes bits sur plusieurs architectures
+ SmallCrush/Crush/BigCrush » possède cependant des précédents : Random123,
notamment Philox et Threefry, y compris un support Metal documenté. Les nouvelles
mesures CPU montrent des avantages dans certaines configurations, mais aucune
supériorité générale. Les anciens comparatifs CUDA utilisent un Philox incorrect
au-delà du premier bloc et ne permettent pas de conclure. Voir ci-dessous et
[ANTERIORITE.md](ANTERIORITE.md).

Le contrat pertinent est : **même version, même seed, même indice logique et même
définition des flux ⇒ mêmes bits**, quelle que soit l'architecture ou la répartition
physique du travail. Un seed seul ne décrit pas le flux GPU historique.

Le premier retour natif **Apple M3** est désormais reçu : les contrôles de
conformité passent, et les médianes de débit en volume placent Philox et Threefry
devant Montmory. Le tableau et sa portée sont précisés plus bas. Il s'agit du
retour terminal transmis par l'utilisateur ; les fichiers de mesure détaillés
restent à recevoir.

## Ce qui a été reçu et figé

L'utilisateur a déclaré le transfert terminé. L'inventaire complet contient
729 fichiers, 22 616 142 octets, avec leurs empreintes dans
[evidence/import-complete.json](evidence/import-complete.json).
La copie originale est dans `../mac-import/CUDA-PRNG/` et n'a pas été modifiée.
Les 38 fichiers utiles figés dans `snapshot/` ont été comparés à la copie finale :
ils sont identiques. Le binaire archivé est un exécutable Mach-O ARM64.

Une seconde copie, provenant de `ovh_gpu_backup/montmory_cuda` sur le Mac,
est ensuite arrivée dans `../mac-import/CUDA-OVH/` : **23 fichiers, 3 668 999 octets**,
dont le kernel CUDA, son wrapper, trois comparatifs GPU et les trois journaux
TestU01. Elle est figée dans `snapshot/cuda-ovh/`, inventoriée avec SHA-256 dans
[evidence/cuda-import.json](evidence/cuda-import.json). Le cœur C est identique
octet pour octet au cœur C de l'archive Metal. Le binaire est ELF x86-64.

Le GPU OVH n'est plus disponible. Les journaux mentionnent l'hôte `a10-45-gra9`
et un commentaire de benchmark indique A10 ; ces éléments ne suffisent pas à
certifier le modèle matériel, ni à confirmer qu'il s'agissait d'un H100.

## Résultats historiques et nouveaux contrôles

| Élément | Résultat et portée |
|---|---|
| Metal, SmallCrush | Journal complet : 15 statistiques, toutes acceptées |
| Metal, Crush | Journal complet : 144 statistiques, toutes acceptées |
| Metal, BigCrush | Journal complet : 160 statistiques, toutes acceptées |
| CUDA, SmallCrush/Crush/BigCrush | Trois journaux complets : 15/144/160 statistiques, toutes acceptées |
| P-values imprimées CUDA contre Metal | Identiques pour les trois batteries : respectivement 15, 186 et 254 valeurs imprimées |
| Journal BigCrush CUDA du dépôt antérieur | Identique à la sauvegarde OVH après remplacement du seul nom du générateur |
| Archive Metal | Les 4 empreintes des sources/binaire correspondent au manifeste |
| Journal BigCrush Metal du dépôt antérieur | Identique à l'archive Mac après remplacement du nom du générateur |
| Deux noms de générateur CPU anciens | SmallCrush et Crush réussis dans les journaux ; la variante et ses constantes doivent rester identifiées |
| Reproduction x86 du flux Metal, seed 42, T=2048, D=4096 | SmallCrush réussi ; les 15 p-values imprimées correspondent exactement au journal Mac |
| Autres SmallCrush nouveaux | CPU seeds 0, 1, 42 ; bits bas 32, bits hauts 32 ; flux Metal T=7,D=11 : tous réussis |
| Modèle entier Python contre C compilé | 3 072 cas identiques, y compris valeurs limites |
| Corps du kernel Metal adapté syntaxiquement en C++ | 9 536 mantisses identiques à la référence ; ce n'est pas une exécution GPU |
| Corps des kernels CUDA/Metal et référence entière | 251 904 mantisses identiques, y compris débordements modulaires ; adaptation C++ sur CPU |
| Répartition réelle OpenMP | 3 600 108 mots comparés, identiques avec 1, 2 et 4 workers et ordres de calcul différents |
| Portabilité du C | 1 600 vecteurs de référence produits avec UBSan, sans diagnostic |
| Exécution Apple M3 rapportée par l'utilisateur | 1 600 vecteurs ARM64/x86 identiques ; 909 312 mots Metal/références comparés, tous moteurs confondus ; six vecteurs officiels sur CPU puis GPU : PASS |
| Benchmark CPU avec Random123 officiel | 24 vecteurs officiels et 262 336 comparaisons du cœur Montmory vérifiés avant chaque série |

Les nombres 15/144/160 désignent des **statistiques**, pas autant de tests
indépendants. Les sept nouveaux SmallCrush ont été exécutés avec TestU01 1.2.3
recompilé depuis l'archive du site des auteurs. Les sources `bbattery.c/.tex` et
`unif01.c/.tex` reçues du Mac sont identiques à celles de cette version officielle.
À ce stade initial, Crush et BigCrush n'avaient pas été relancés. Les nouvelles
exécutions natives M3 sont consignées plus bas : Crush réussi et BigCrush
démarré, bilan final attendu. Les durées historiques de ces batteries ne sont
pas des benchmarks du débit du générateur.

## Propriétés mathématiques et limites

Les démonstrations et certificats sont dans [MATHEMATIQUES.md](MATHEMATIQUES.md).

1. Le compteur de chaque lane a exactement la période 2^64.
2. Les trois tours Feistel constituent une permutation de 64 bits.
3. Le mélange final appelé `pcg_dxsm` dans le code est personnalisé et n'est pas
   le DXSM de référence de PCG. Il n'est pas injectif : quatre entrées distinctes
   donnent exactement le mot 7. Cela ne contredit pas les succès TestU01.
4. Même la sortie tronquée à 53 bits n'est pas parfaitement équilibrée sur le
   cycle complet : la mantisse zéro possède 2 066 antécédents au lieu de 2 048.
   L'excès absolu pour cet événement est 18/2^64 ; ce seul certificat ne démontre
   pas une erreur Monte Carlo importante.
5. Les lanes d'un seed partagent les mêmes clés et réordonnent le même espace
   d'états. Des pas impairs distincts ne prouvent pas l'indépendance des lanes.
6. Le flux historique dépend des paramètres logiques `threads` et `draws`.
   Ceux-ci peuvent être figés dans le contrat et séparés des threads physiques.
   Le moteur n'a pas besoin d'être remplacé pour rendre cette séparation explicite.

## Parallélisation et identité des sorties

Pour la position globale zéro-indexée `k`, le kernel historique réalise :

```
lane = floor(k / D) mod T
counter = seed + (k + 1) * (0x9E3779B97F4A7C15 + 2*lane) mod 2^64
word = mixer(feistel_seed(counter))
mantissa = word >> 11
```

Cette formule permet de traiter directement n'importe quelle position, sans
état partagé entre workers. Le contrôle OpenMP conserve T et D comme paramètres
**logiques**, puis change réellement le nombre de workers physiques. Les sorties
restent identiques. Utiliser systématiquement lane=0 permet aussi de répartir un
flux canonique unique en segments disjoints ; ce choix change le flux Metal
historique et demanderait une nouvelle version explicite et sa validation.

Le kernel reçu écrit `out[lane*D+i]`. À `i` fixé, les threads voisins écrivent
loin les uns des autres. C'est une piste d'optimisation de l'accès mémoire, pas
un défaut statistique démontré. Une distribution par indice global permettrait
des écritures contiguës tout en conservant la formule logique. Le gain doit être
mesuré sur les GPU réels. Le rapatriement dans un tampon CPU sert au wrapper
TestU01 ; il ne mesure pas un usage où les tirages restent dans le calcul GPU.

## Défauts concrets trouvés dans la sauvegarde CUDA

Le wrapper lance `ceil(T/256)*256` threads, alors que le kernel écrit sans vérifier
`tid<T`. Pour T=257 et D=3, **765 mots sont écrits au-delà de l'allocation prévue**.
Le témoin est reproduit en allouant volontairement une zone sentinelle assez
grande, sans provoquer de comportement indéfini dans le vérificateur. Les T des
trois batteries historiques, 2048/4096/8192, sont des multiples de 256 : ce défaut
ne remet pas en cause ces exécutions archivées.

Une correction séparée est dans `reviewed-source/cuda/` : borne du kernel,
définition commune de `Params`, contrôles de dimensions et de taille d'allocation,
libération lors d'une réinitialisation. Le cœur mathématique reste inchangé.
L'émulation vérifie les mêmes 251 904 sorties pour le kernel corrigé, et la
préservation des sentinelles pour cinq configurations auparavant défectueuses.
Le wrapper est relu statiquement ; **cette copie n'a pas encore été compilée par
nvcc ni exécutée sur GPU**. Elle ne reçoit donc aucune nouvelle certification CUDA.

Les trois anciens programmes `bench_montmory_vs_philox_*.cu` ont d'autres défauts :

- Les tours Philox écrasent le compteur persistant et la clé. L'appel suivant
  repart de la sortie précédente incrémentée et d'une clé modifiée, au lieu du
  compteur suivant avec la clé initiale. Le premier bloc correspond à Random123,
  le deuxième diverge : 48 cas vérifiés, sur les trois formats.
- Le calcul affiché en GB/s divise par `ms*1e9`, au lieu de `ms*1e6` pour des
  gigaoctets décimaux par seconde : valeur affichée 1 000 fois trop petite.
- Montmory écrit 8 Gio et Philox 16 Gio, avec une seule mesure chronométrée par
  générateur, sans contrôle des erreurs CUDA dans ces benchmarks.
- Le checksum des doubles lit un `double` par un pointeur `uint64_t*`, ce qui
  enfreint les règles d'aliasing du C++ ; utiliser `memcpy` pour lire ces bits.

La copie reçue contient les sources et exécutables de ces comparatifs, **aucun
journal de leurs résultats**. Ils ne démontrent donc pas un avantage de débit
GPU. Les succès TestU01 de Montmory restent distincts de ces comparatifs.
Tous les témoins arithmétiques sont dans
[evidence/cuda-exact.json](evidence/cuda-exact.json).

## Avantage de débit : mesures CPU nouvelles

Machine : AMD EPYC-Milan sous KVM, GCC 12.2.0, VPS partagé. Huit variantes
officielles Random123 à nombre de tours standard, commit
`9545ff6413f258be2f04c1d319d99aaef7521150`, sont comparées au cœur récupéré.
Leurs fichiers sont vérifiés contre les empreintes Git amont.

Chaque cas utilise 8 388 608 mots de 64 bits, 11 seeds, 1 puis 4 workers physiques,
un ordre de passage mélangé, un échauffement, et deux modes : calcul avec checksum,
ou écriture des résultats en mémoire. Les clés sont initialisées hors chronométrage.
Les checksums concordent entre modes, nombres de workers et options de compilation.
Les tableaux suivants donnent les **médianes ns/mot avec écriture, 1 worker** :
une valeur plus petite est meilleure. Les 1 760 mesures et les huit références
figurent dans [evidence/benchmark-summary.json](evidence/benchmark-summary.json).

**Ces chiffres sont exclusivement x86, pas Metal ni Apple Silicon M3.**
Threefry2x32-20 a une implémentation Metal concrète dans MLX. En revanche,
Threefry4x64-20 est désactivé dans le port Metal par défaut de la version Random123
examinée (`R123_USE_64BIT=0`). Sa présence dans le tableau CPU ne constitue pas
une validation Metal. Voir la distinction et les sources dans
[ANTERIORITE.md](ANTERIORITE.md). Un premier classement Metal sur le M3 est
maintenant disponible dans la section suivante, avec une portée distincte.

| Génération en volume, tous les mots de chaque bloc consommés | `-O3 -march=native` | `-O3` générique |
|---|---:|---:|
| Montmory historique | 3,758 | 5,615 |
| Philox4x32-10 | 4,374 | 3,620 |
| Philox4x64-10 | 3,013 | 2,895 |
| Threefry2x32-20 | 3,103 | 5,222 |
| Threefry4x64-20 | 1,633 | 2,985 |

Avec les options natives, le gain apparié médian face à Philox4x32 est **1,169×**,
Montmory gagnant les 11 comparaisons : environ 266 contre 229 millions de mots/s.
Mais ce gain disparaît avec les options génériques ; même en prenant le meilleur
des deux réglages pour chaque moteur, il ne subsiste pas dans ce test en volume.
Threefry4x64 est nettement plus rapide. À quatre workers, le VPS introduit plus de
variabilité : les résultats complets sont conservés, sans en déduire un facteur
universel de passage à l'échelle.

Un second usage demande **un seul mot par indice aléatoire isolé**. Les indices
sont précalculés hors chronométrage ; leur lecture et l'écriture du résultat sont
incluses. Les autres mots d'un bloc Random123 ne sont alors pas demandés. Cela
mesure un autre usage, qui ne doit pas être confondu avec la production en volume.

| Accès à des indices isolés | `-O3 -march=native` | `-O3` générique |
|---|---:|---:|
| Montmory historique | 3,682 | 13,391 |
| Philox2x32-10 | 6,714 | 6,305 |
| Philox4x32-10 | 12,834 | 13,499 |
| Threefry2x32-20 | 3,148 | 5,268 |
| Threefry4x64-20 | 14,256 | 14,307 |

Le gain natif apparié face à Philox4x32 atteint ici **3,431×**, mais Threefry2x32
reste plus rapide que Montmory. La forte dépendance aux options de compilation
interdit une annonce générale. Une variante Montmory sans branche a aussi été
mesurée : elle n'améliore pas uniformément les résultats et n'est pas adoptée
dans le cœur corrigé. Le pilote avec une autre initialisation des clés reste
archivé séparément et n'entre pas dans ces chiffres.

**Intérêt établi :** moteur à accès direct, parallélisable, reproductible sous
un contrat logique explicite, avec de bons résultats statistiques observés et
des performances compétitives dans certains cas. **Avantage exclusif ou général :
non établi.** Les mesures CPU ci-dessus ne permettent aucune extrapolation du
débit CUDA, Metal ou ARM natif, ni de l'énergie consommée. Le premier retour
Metal natif est présenté séparément ci-dessous.

## Premier résultat natif Apple M3

L'utilisateur a transmis le retour terminal du lancement
`20261021T124458Z-8820`, sur **Apple M3**, pour 4 194 304 mots de 64 bits par
lancement (32 Mio), avec 11 répétitions. Compilation ARM64 et Metal réussies ;
1 600 vecteurs ARM/x86, 909 312 mots GPU/références tous moteurs confondus et
les six vecteurs officiels Random123 sur CPU puis GPU sont annoncés PASS.
Le programme termine également ses contrôles d'échantillons, de sentinelles et
de checksums pendant les mesures.

La [transcription reçue](evidence/metal-m3-user-console.log) et son
[extraction structurée](evidence/metal-m3-transcript-summary.json) sont conservées.
Les CSV bruts et métadonnées d'état thermique n'ont pas encore été reçus : aucun
intervalle de confiance ni ratio apparié par répétition n'est calculé à ce stade.

Les meilleurs débits médians affichés parmi les configurations testées sont :

| Moteur | Mots bruts 64 bits, GB/s | Mantisses 53 bits stockées sur 64 bits, GB/s |
|---|---:|---:|
| Montmory | 53,709 (indexé, 65 536 threads) | 54,996 (historique, 8 192 threads) |
| Threefry2x32-20 | 81,567 (65 536 threads) | 84,272 (65 536 threads) |
| Philox4x32-10 | 85,155 (65 536 threads) | 82,944 (65 536 threads) |

Pour le format 53 bits, Threefry et Philox ont respectivement environ **53 %** et
**51 %** de débit supplémentaire par rapport au meilleur Montmory affiché.
Pour les mots bruts, les écarts sont d'environ **52 %** et **59 %**. Ce sont des
rapports de médianes affichées et arrondies, pas des statistiques appariées.

La réorganisation de Montmory ne montre pas de gain dans cette campagne :
53,544 GB/s en 53 bits au meilleur réglage indexé, contre 54,996 GB/s pour
l'historique. L'écart n'est que d'environ 2,7 % en débit en faveur de l'historique ;
les données brutes sont nécessaires pour juger sa stabilité. À 8 192 threads
pour les deux versions, les médianes sont 50,702 contre 54,996 GB/s.

**Ce résultat confirme le fonctionnement natif et les identités sur les cas
testés, mais ne démontre aucun avantage de vitesse de Montmory sur ce cas M3.**
Il confirme aussi l'exécution concrète des deux concurrents en parallèle sous
Metal. Ce test porte sur la génération en volume, une taille de tampon et les
deux répartitions physiques du kit. Il ne classe pas tous les usages possibles,
ne mesure pas l'énergie et ne relance aucune batterie TestU01.

## Deuxième résultat natif M3 : optimisations exactes

Les résultats M3 ci-dessus comparent bien **trois kernels GPU Metal**. Le C
sert de référence hors chronométrage ; les tableaux x86 précédents sont une
campagne CPU distincte. Le Montmory historique est déjà parallèle : 8 192
threads GPU dans ce test. Les threads physiques ne sont pas des cœurs matériels.

L'égalité du niveau d'optimisation des moteurs n'est pas établie. Le premier
kit ne permet pas de déclarer un classement entre implémentations optimales.
Un [deuxième kit](mac-metal-benchmark-v2/README.md) ajoute trois candidats :
décomposition du calcul en mots de 32 bits, avance du compteur par additions,
et combinaison des deux. Les noyaux historiques et Random123 restent inclus
pour comparaison dans la même campagne. Les constantes, tours et sorties
du générateur ne sont pas modifiés ; les
[identités exactes](mac-metal-benchmark-v2/EXACT.md) sont documentées.

Sur le VPS, 60 883 200 mots de calcul Metal adapté en C++ concordent avec les
références, avec UBSan, ainsi que 7 432 282 comparaisons de composants.

Le retour terminal du lancement natif `20261021T131521Z-8925` a maintenant été
reçu : compilation ARM64/Metal réussie, 1 600 vecteurs ARM/x86 et 4 153 344 mots
GPU/références tous moteurs confondus annoncés PASS, six vecteurs officiels sur
CPU et GPU, puis 11 répétitions terminées. Le volume reste 4 194 304 mots de
64 bits par lancement (32 Mio), pour 43 configurations. Les variantes utilisent
8 192, 65 536 et 262 144 threads physiques avec le même contrat logique.

La [transcription v2](evidence/metal-m3-v2-user-console.log) et son
[extraction structurée](evidence/metal-m3-v2-transcript-summary.json) sont
conservées. Les 43 configurations attendues sont présentes et les unités
ns/mot et GB/s concordent dans la précision d'affichage. L'archive brute n'a
pas encore été reçue : les rapports suivants utilisent les médianes arrondies,
sans statistique appariée par répétition ni intervalle de confiance.

| Moteur | Meilleur débit brut 64 bits, GB/s (threads) | Meilleur débit 53 bits, GB/s (threads) |
|---|---:|---:|
| Montmory indexé initial | 53,841 (65 536) | 53,859 (65 536) |
| Montmory historique | — | 55,332 (8 192) |
| Montmory 32 bits indexé | 61,161 (262 144) | 61,413 (262 144) |
| Montmory 64 bits avec avance par additions | 55,378 (65 536) | 55,665 (65 536) |
| Montmory 32 bits avec avance par additions | **62,969 (65 536)** | **62,983 (262 144)** |
| Threefry2x32-20 | 85,680 (262 144) | 83,219 (65 536) |
| Philox4x32-10 | 83,599 (262 144) | 81,123 (8 192) |

Le meilleur Montmory 53 bits gagne **13,8 % de débit face au kernel historique**
de cette même campagne. Face au meilleur Montmory indexé initial, le gain
atteint **16,9 % en 53 bits et 17,0 % en brut 64 bits**. À nombre de threads
identique, les gains restent visibles : en brut à 65 536 threads, la seule
réécriture 32 bits donne 60,943 contre 53,841 GB/s (+13,2 %), et l'avance par
additions seule donne 55,378 GB/s (+2,9 %). Leur combinaison atteint 62,969 GB/s.
L'essentiel du gain observé vient donc de la réécriture arithmétique 32 bits,
sans que ces mesures constituent un profil des instructions ou registres.

**Il existe maintenant un gain M3 observé de l'implémentation optimisée sur
l'implémentation précédente, mais toujours aucun avantage de débit sur les
deux références Random123 dans ce test.** En 53 bits, Philox et Threefry gardent
respectivement 28,8 % et 32,1 % de débit supplémentaire sur le meilleur
Montmory ; en brut, les écarts sont 32,8 % et 36,1 %.

Le nombre maximal de threads n'est pas systématiquement le meilleur réglage :
pour Montmory 32 bits avec additions en brut, 65 536 threads donnent 62,969 GB/s
contre 62,292 à 262 144 threads. Cette petite différence reste à examiner avec
la dispersion. Elle n'établit pas une limite générale de parallélisation.
Le témoin d'écriture est explicitement non aléatoire ; ses débits ne constituent
ni un classement de PRNG ni une borne matérielle théorique.

Les identités testées portent sur les mêmes flux et les mêmes paramètres
logiques ; **aucune batterie SmallCrush, Crush ou BigCrush n'a été relancée dans
ce benchmark**. Les propriétés et limites mathématiques du générateur sont
conservées par les réécritures exactes ; ce retour n'établit pas un nouveau
passage statistique du binaire natif.

Le kit original et son archive restent inchangés. La nouvelle archive est
`mac-metal-benchmark-v2-kit.tar.gz`, à extraire dans son répertoire distinct
`montmory-metal-m3-v2`. Les résultats sont archivés avec le préfixe
`montmory-metal-v2-results-`.
Le deuxième kit livré est également figé : ses documents internes conservent
l'état de préparation, et le présent rapport enregistre les résultats reçus.

## Troisième kit : lots, spécialisation et diagnostic

À la demande de poursuivre l'optimisation, un
[kit v3](mac-metal-benchmark-v3/README.md) a été préparé séparément. Il conserve
les moteurs déjà mesurés et ajoute cinq candidats : lots de deux ou quatre
sorties indépendantes, calcul alternatif des retenues par produit haut, et
deux formes de spécialisation des rotations dérivées de la seed. La fonction
génératrice et son contrat logique sont préservés ; les
[identités](mac-metal-benchmark-v3/EXACT.md) sont documentées.

Les contrôles préalables avec UBSan comparent **199 824 000 mots complets**,
**73 967 040 réductions par thread** et **9 713 754 résultats de composants**,
sans différence. Les nouveaux cas comprennent des lanes d'un seul ou de deux
mots, des threads inactifs et des bases proches du débordement. L'adaptation
CPU représente les rotations spécialisées par des variables ; elle ne valide
pas le compilateur de spécialisation Metal. Un nouveau SmallCrush de
`spec_dual` adapté sur CPU réussit et reproduit les quinze p-values imprimées
historiques. Preuves dans `mac-metal-benchmark-v3/generated/`.

Le retour natif `20261021T134922Z-9053` annonce la compilation ARM64/Metal
réussie, **1 600 vecteurs ARM/x86**, **4 835 328 mots complets** et
**4 004 592 réductions individuelles** conformes, tous moteurs confondus,
ainsi que six vecteurs officiels sur CPU et GPU. Les onze répétitions sont
terminées. Le protocole comporte 114 configurations par seed : 105 mesures
en volume et neuf diagnostics. Les tailles de groupes de 64, 128 et 256 sont explorées pour les
principaux moteurs. Les coûts de préparation des pipelines spécialisés sont
enregistrés séparément et exclus du débit GPU : un éventuel avantage de ces
variantes dépendra aussi du nombre de sorties produites par seed et du cache.

Un diagnostic calcule toutes les sorties mais ne livre que leur XOR par thread.
Il modifie l'accumulation, les ressources et les écritures ; il ne mesure pas
le calcul pur et ses temps ne peuvent pas être soustraits pour isoler exactement
le coût mémoire. Les unités distinguent les mots générés des octets écrits.
Une capture GPU facultative peut être ouverte dans Xcode pour approfondir le
diagnostic, sans être confondue avec une campagne de débit ordinaire.

Le kit fournit aussi un lanceur natif SmallCrush/Crush/BigCrush, avec la source
officielle TestU01 figée et une compilation locale ARM64. Il utilise la géométrie
statistique du SmallCrush historique (seed 42, 2 048 lanes, 4 096 sorties/lane)
pour les trois batteries, distincte de la géométrie de performance et des
anciens Crush/BigCrush. Le benchmark ne lance pas ces longues batteries
automatiquement. **Le benchmark natif v3, SmallCrush et Crush natifs de
`mulhi_pair` ont réussi selon les retours terminaux ; BigCrush reste en attente.**

La [transcription v3](evidence/metal-m3-v3-user-console.log) et son
[extraction structurée](evidence/metal-m3-v3-transcript-summary.json) sont
conservées. L'affichage sélectionne les 25 meilleurs réglages par moteur et
format parmi les médianes, auxquels s'ajoutent neuf diagnostics. Les unités
ns/mot et GB/s concordent à la précision d'affichage. Les CSV, les métadonnées
et les coûts de préparation n'ont pas encore été reçus : les comparaisons
suivantes portent sur ces médianes arrondies, sans analyse de dispersion ni
confirmation sur une campagne distincte.

| Moteur | Meilleur débit brut 64 bits, GB/s | Meilleur débit 53 bits, GB/s |
|---|---:|---:|
| Montmory historique | — | 55,234 |
| Montmory indexé initial | 53,698 | 53,914 |
| Montmory 32 bits avec avance par additions (v2) | 62,988 | 62,456 |
| Montmory par lots de deux | 66,814 | 66,417 |
| Montmory par lots de quatre | 66,488 | 66,259 |
| Montmory `mulhi_pair` | **72,309** | **72,786** |
| Montmory `spec_pair` | 71,209 | 71,851 |
| Montmory `spec_dual` | 71,736 | 72,951 |
| Philox4x32-10 | 84,885 | 84,653 |
| Threefry2x32-20 | 88,417 | 88,359 |

**Le candidat retenu provisoirement pour les batteries natives est
`montmory_mulhi_pair`.** Il est le plus rapide des Montmory affichés en brut
64 bits et ne dépend pas d'une spécialisation de pipeline par rotations de
seed. En 53 bits, `spec_dual` le dépasse seulement de 0,23 %, avec une taille
de groupe différente ; cet écart n'est pas statistiquement établi. La
compilation de ces variantes spécialisées fonctionne sur ce Mac, mais leur
intérêt pratique dépend aussi des coûts de préparation absents du tableau.

Dans cette même campagne, `mulhi_pair` gagne **31,8 % de débit** sur
l'historique 53 bits et **16,5 %** sur le meilleur réglage v2 53 bits. Le gain
sur le Montmory indexé initial est de 35,0 % en 53 bits et 34,7 % en brut.
Le passage des lots de deux ordinaires au calcul par produit haut améliore
les meilleurs débits observés de 9,6 % en 53 bits et de 8,2 % en brut. Ces
comparaisons associent les réécritures à des réglages sélectionnés ; elles
n'isolent pas le coût d'une instruction ni la pression sur les registres.
Les lots de quatre et la spécialisation n'apportent pas de gain convaincant
sur le candidat dans cette campagne.

Le diagnostic avec XOR par thread atteint 10,809 Gmots générés/s pour
`mulhi_pair`, contre 8,644 pour la variante v2, 11,679 pour Philox et 13,061
pour Threefry. Le gain Montmory reste donc observable lorsque les écritures
du flux complet sont supprimées. La taille de groupe demandée est 256 et
le nombre de threads 65 536 ; les tailles effectives restent à vérifier
dans les CSV. Ce diagnostic ne fournit pas une décomposition exacte entre
calcul et mémoire et utilise parfois d'autres réglages que les meilleurs
débits en volume ci-dessus.

**Aucun avantage de débit sur ces deux références Random123 n'est établi.**
En 53 bits, Philox et Threefry produisent respectivement 16,3 % et 21,4 % de
débit supplémentaire par rapport à `mulhi_pair` ; en brut, 17,4 % et 22,3 %.
Ce résultat concerne un volume et un Apple M3, sans extrapolation CUDA.
Les contrôles d'identité natifs réussis confortent les réécritures exactes
sur les cas vérifiés. Ces contrôles seuls ne constituent pas un passage
statistique ; les résultats SmallCrush et Crush obtenus séparément sont décrits ci-dessous.

Archive : `mac-metal-benchmark-v3-kit.tar.gz`, répertoire d'extraction distinct
`montmory-metal-m3-v3`. Les trois kits livrés et leurs empreintes restent
inchangés ; leurs documents internes décrivent leur état de préparation.
Le présent rapport conserve les résultats reçus ensuite.

La première tentative du lanceur TestU01 natif (`20261021T140817Z-9132`)
s'est arrêtée au premier essai de liaison C de `configure`, avant le wrapper
et les batteries. La détection ARM fonctionne ; le détail du `config.log`
reste à recevoir. Un [complément de lancement](mac-metal-testu01-repair-v1/README.md)
séparé sélectionne explicitement le SDK macOS, isole les options de compilation
et rend les erreurs visibles. Il conserve tous les sources du kit v3 et crée
une installation locale neuve. Archive : `mac-testu01-repair-v1.tar.gz`.

Le retour suivant du Mac confirme le programme C ARM64 préalable, puis la
construction de TestU01 et du wrapper, puisque la batterie native s'exécute.
**SmallCrush natif Metal de `montmory_mulhi_pair` annonce « All tests were
passed » pour ses 15 statistiques.** Les quinze p-values imprimées sont
identiques à la référence historique conservée, dans le même ordre. Le flux
testé utilise la seed 42, 2 048 lanes logiques et 4 096 sorties par lane, en
uniformes 53 bits. Le temps CPU annoncé est de 4,74 s ; il ne mesure pas
le débit GPU ni le temps total de construction et de vérification.

Les [extraits du terminal](evidence/metal-m3-v3-native-smallcrush-user-extract.log)
et leur [vérification structurée](evidence/metal-m3-v3-native-smallcrush-summary.json)
sont conservés. Le passage confirme la construction native avec le nouveau
lanceur, mais n'isole pas la cause du premier échec : celui-ci sélectionne
le SDK et nettoie aussi les options héritées. L'archive brute et l'identifiant
de cette exécution restent à recevoir. Après SmallCrush, un nouveau contrôle
de conformité annonce PASS ; le résumé Crush reçu ensuite figure ci-dessous.
L'égalité des p-values imprimées ne prouve pas à elle seule l'identité des flux,
qui fait l'objet des contrôles exacts distincts.

Un extrait ultérieur confirme le démarrage de **Crush**, avec 49 p-values
imprimées reçues jusqu'au premier test `sknuth_Run`. Aucune de ces valeurs
affichées ne sort de l'intervalle par défaut [0,001 ; 0,999] utilisé par
TestU01 pour signaler les résultats suspects. À ce stade intermédiaire, le résumé
final n'était pas encore reçu ; ce nombre de valeurs ne doit pas être confondu avec le
nombre de statistiques du résumé de la batterie.

**Correction de protocole : le terme « flux historique » utilisé précédemment
pour les trois batteries était trop large.** Les en-têtes CUDA archivés donnent
2 048 × 4 096 pour SmallCrush, 4 096 × 8 192 pour Crush, et 8 192 × 16 384
pour BigCrush, toujours seed 42. Le wrapper v3 garde 2 048 × 4 096 pour chacune
des trois batteries. Les 49 p-values Crush reçues diffèrent de celles des
anciens journaux Metal/CUDA ; la comparaison directe de ces flux différents
n'est pas une vérification d'identité valable. Le lancement actuel teste les
trois batteries sur un même contrat logique et doit être mené à terme.
Les anciens paramètres seraient nécessaires pour un rejeu de leurs journaux.
L'[extraction et les références](evidence/metal-m3-v3-native-crush-prefix-summary.json)
consignent cette distinction ; aucune modification du lancement en cours
ni des kits livrés n'a été faite.

**Le résumé final de Crush natif Metal reçu ensuite annonce « All tests were
passed » pour 144 statistiques**, avec `montmory_mulhi_pair`, TestU01 1.2.3.
Le flux conserve la seed 42 et la géométrie fixe 2 048 × 4 096, en uniformes
53 bits. Le temps CPU annoncé est de 15 min 35,72 s ; il ne constitue pas une
mesure isolée du GPU ni un comparatif de débit avec les anciennes campagnes.
Le [résumé reçu](evidence/metal-m3-v3-native-crush-user-summary.log) et son
[extraction vérifiée](evidence/metal-m3-v3-native-crush-summary.json) sont conservés.
Un nouveau contrôle de conformité PASS suit ce résumé, conformément à
l'enchaînement vers BigCrush du lanceur `all`. Aucun résultat final BigCrush
n'a encore été reçu. L'archive brute et ses métadonnées restent à examiner.

**Le retour suivant confirme le démarrage effectif de BigCrush**, avec la
ligne `battery=big seed=42 lanes=2048 draws=4096`. Il contient les deux
premiers résultats SerialOver et neuf résultats CollisionOver, soit onze
p-values imprimées comprises entre 0,06 et 0,98. Le dernier bloc est tronqué
après sa p-value, dans le tableau d'occupation. Cette progression ne constitue
pas un bilan final. L'[extraction du début de BigCrush](evidence/metal-m3-v3-native-bigcrush-prefix-summary.json)
conserve les paramètres et résultats visibles, sans assimiler cet extrait à
une archive brute ni à une observation en direct du processus Mac.

## Ce qui reste à établir sur le matériel

- Recevoir les résultats bruts des trois campagnes M3 pour vérifier la
  dispersion, les métadonnées et la correspondance avec les kits, au-delà des
  retours terminaux déjà reçus.
- Compiler la correction CUDA et exécuter `tools/check_cuda.cu` sur GPU.
- Confirmer les gains v3 sur une campagne distincte, examiner les coûts de
  préparation et recevoir le résultat natif BigCrush ainsi que
  l'archive du lancement statistique de `montmory_mulhi_pair`.
- Mesurer le débit CPU ARM et CUDA et, pour élargir le constat M3, d'autres
  volumes/usages et les ressources face à Philox/Threefry, en distinguant
  génération avec accumulation, écriture du flux complet et transfert vers le CPU.

Le script de conformité seul [tools/check_macos.sh](tools/check_macos.sh) reste
disponible. Le retour M3 reçu provient du kit de benchmark ci-dessous, qui
effectue ses propres contrôles ; il ne doit pas être attribué à ce script distinct.

Un [kit de comparaison Metal pour le M3](mac-metal-benchmark/README.md) est
également préparé dans `mac-metal-benchmark-kit.tar.gz`. Il vérifie les vecteurs
ARM/x86, les vecteurs officiels Random123 et les sorties GPU, puis mesure les
trois moteurs en volume, avec conservation des mesures brutes. L'émulation CPU
préalable de ses kernels vérifie 17 047 296 mots. Sa première compilation et
exécution natives sur le M3 sont maintenant rapportées comme réussies par
l'utilisateur. Le kit livré et ses empreintes sont conservés sans modification.

L'audit fournit une référence et des contrôles reproductibles. Il ne constitue
pas une nouvelle version du générateur déjà certifiée sur quatre architectures.
Voir [REPRODUIRE.md](REPRODUIRE.md).
