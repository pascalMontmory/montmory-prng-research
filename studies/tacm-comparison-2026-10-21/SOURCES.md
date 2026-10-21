# Générateurs étudiés — sources primaires consultées le 21 octobre 2026

Cette revue accompagne une campagne TACM ciblée. « Exécuté » désigne les moteurs réellement utilisés par `run_campaign.py`, sur le CPU du VPS. Aucun chiffre publié n'est présenté comme une mesure réalisée ici.

| Famille / variante | Intérêt pour notre comparaison | Travail effectué |
|---|---|---|
| Philox4x32-10 | Générateur à compteur ; candidat direct pour GPU et accès par indice | Exécuté, code Random123 et vecteurs officiels |
| Threefry2x32-20 | Compteur, opérations entières 32 bits ; variante du benchmark Metal | Exécuté, code Random123 et vecteurs officiels |
| Threefry4x64-20 | Variante par blocs de quatre mots 64 bits | Exécuté, tous les mots du bloc utilisés |
| Philox4x64 NumPy | Autre largeur et autre initialisation ; ne pas confondre avec Philox4x32 | Exécuté via NumPy 1.24.2 |
| PCG64 / PCG64DXSM | État récurrent, sortie mélangée ; comparaison avant/après une amélioration documentée des flux parallèles | Les deux exécutés via NumPy 1.24.2 |
| xoshiro256** | Petit état et opérations simples ; références CPU et fonctions de saut | Exécuté ; sortie comparée au code officiel |
| SFC64 | Autre famille à état, disponible dans NumPy | Exécuté via NumPy 1.24.2 |
| MT19937 | Référence historique à grand état, utile pour situer la portée des diagnostics | Exécuté ; deux sorties 32 bits assemblées en un mot 64 bits |
| Montmory | Cœur récupéré, géométrie logique 2048 × 4096, indexation déterministe | Exécuté ; équivalence vérifiée sur plus d'un million de mots avec le cœur original |
| MRG32k3a, XORWOW, MTGP32 | Autres approches du parallélisme disponibles dans cuRAND | Documentation étudiée ; pas exécutés dans cette campagne |

Les variantes Philox et Threefry à nombre de tours recommandé ont déjà passé SmallCrush, Crush et BigCrush selon la documentation des auteurs. Le calcul `CBRNG(compteur, clé)` est déterministe et sans état mutable ; le parallélisme et la reproductibilité par indice ne sont donc pas exclusifs à Montmory. Les blocs complets sont bijectifs à clé fixée ; cela ne signifie pas que chaque mot extrait de ces blocs est unique. [Documentation Random123](https://random123.com/releases/1.09/docs/CBRNG.html), [article de Salmon et al., 2011](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).

Les en-têtes Random123 embarqués proviennent du commit `9545ff6413f258be2f04c1d319d99aaef7521150`. Leur licence est conservée dans `deps/random123/LICENSE`. Le moteur ici n'utilise pas l'initialisation propre à cuRAND : son algorithme et son protocole de clés sont documentés dans `src/engines.cpp`. [Dépôt officiel](https://github.com/DEShawResearch/random123/tree/9545ff6413f258be2f04c1d319d99aaef7521150).

NumPy rapporte une faiblesse de PCG64 qui peut apparaître avec des millions de flux longs et explique la correction PCG64DXSM. Cela justifie d'étudier les relations entre flux, mais nos 2 048 seeds et nos courts prélèvements ne reproduisent pas ce régime. Un résultat sans alerte ici ne réfute pas cette faiblesse documentée. Le nom `pcg_dxsm` présent dans Montmory désigne un mélangeur personnalisé, pas l'algorithme standard PCG64DXSM. [Explication officielle de NumPy](https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html).

Les générateurs NumPy ont des initialisations et possibilités de saut propres. Une même valeur de seed ne produit pas des tirages couplés identiques entre familles. Notre appariement par groupe sert à comparer des budgets identiques ; il ne suppose pas de technique de réduction de variance par nombres aléatoires communs. [Catalogue des moteurs NumPy](https://numpy.org/doc/stable/reference/random/bit_generators/index.html).

La famille xoshiro propose plusieurs sorties différentes. Nous testons précisément `xoshiro256**` ; les défauts de bits faibles des variantes « + » ne doivent pas lui être attribués automatiquement. Les fonctions de saut fournissent aussi un moyen de répartir les séquences. [Page des auteurs](https://prng.di.unimi.it/), [source officielle utilisée](https://prng.di.unimi.it/xoshiro256starstar.c).

NVIDIA documente des générateurs utilisables dans les kernels CUDA, notamment Philox4x32-10, MRG32k3a, XORWOW et MTGP32, avec des mécanismes de sous-séquences ou des contraintes de partage d'état. Leur parallélisme existe donc bien ; ses modalités diffèrent selon le moteur. Aucune exécution CUDA n'a été réalisée sur ce VPS. [Documentation cuRAND](https://docs.nvidia.com/cuda/curand/device-api-overview.html).

Une prépublication de mai 2026, Randompack, décrit également une bibliothèque C reproductible entre langages, compilateurs et architectures CPU, avec prise en charge des simulations parallèles. C'est un élément d'antériorité supplémentaire ; cette source ne certifie pas à elle seule un ensemble de backends Metal/CUDA. Randompack n'a pas été installé pour cette campagne. [Jónasson, Randompack](https://arxiv.org/abs/2605.05099).

Une autre prépublication de mai 2026 examine de nombreux flux avec BigCrush. Son résumé affiche un maximum de 72 % de passages complets ; sa discussion traite aussi les alertes dues aux comparaisons multiples. Il serait trompeur d'en déduire que les moteurs concernés sont défectueux sur 28 % de leurs flux. Sous l'hypothèse simplificatrice de 160 p-values indépendantes et uniformes, la probabilité de rester dans [0,001 ; 0,999] pour toutes est `0,998^160 ≈ 72,6 %`. Cette explication n'est pas un modèle exact de la dépendance entre les tests de BigCrush. [Wartel et Hill, discussion p. 12–13](https://arxiv.org/pdf/2605.18227).

Cette sélection couvre plusieurs références importantes ; elle n'est pas un inventaire exhaustif. Les générateurs cryptographiques et les méthodes quasi-Monte-Carlo demanderaient des critères complémentaires et ne sont pas classés par cette expérience.
