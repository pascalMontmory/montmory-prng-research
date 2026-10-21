# Optimisations Montmory à flux identique — 21 octobre 2026

Une réécriture CPU avec compteur incrémental et calcul par paires améliore le débit observé sur le VPS. Sept variantes Metal équivalentes sont livrées pour les mesures natives M3. La définition du flux reste fixée à T=2048 lanes logiques et D=4096 tirages par lane, comme le TestU01 Mac.

## Retour natif M3 communiqué par l'utilisateur

L'exécution `20261021T162407Z-22542` a terminé la compilation native, les vérifications ARM64/x86 et Metal, les sept seeds de recherche et les cinq nouvelles seeds de confirmation. L'archive a ensuite été reçue et relue : SHA-256 `053d4930cf56681df3afdc242675e8308967b551f2e70ea752ac12c3eaaa0042`. Les 62 empreintes du manifeste correspondent au kit livré. `python3 analyze_mac.py` vérifie les fichiers, les vecteurs, le choix des configurations, les identités d'unités et recalcule les résumés à partir des 144 mesures CPU, 2464 mesures GPU de recherche et 110 de confirmation. Tous les contrôles de cohérence passent. [Analyse structurée](results/mac-analysis-20261021T162407Z-22542/analysis.json) ; [figure](results/mac-analysis-20261021T162407Z-22542/mac-analysis.png). Cette réanalyse utilise les traces natives reçues ; le VPS n'exécute pas Metal.

Sur CPU ARM64, `incremental_pair` passe de 2,689 Go/s à un worker à 12,487 Go/s à huit workers. À huit workers, `indexed` atteint 9,746 Go/s : le gain de débit observé est de 28,1 % (rapport de médianes), soit environ 22,0 % de temps en moins. Les rapports appariés par seed donnent un gain médian de 27,9 % à huit workers et la version par paires gagne 9/9 comparaisons pour chacun des quatre nombres de workers. Ce kit ne compare pas les moteurs concurrents sur CPU.

Sur Metal, en raw64 : `montmory_mulhi_pair` atteint 85,123 Go/s, `montmory_split_stripe_pair` 85,580 Go/s, Philox4x32-10 89,330 Go/s et Threefry2x32-20 92,474 Go/s. Les rapports appariés du compteur scindé face à v3 ont une médiane de 1,00510 en raw64 et 1,00557 en mantissa53, avec 5/5 comparaisons gagnées dans les deux formats. L'effet est faible et cinq seeds ne démontrent pas sa stabilité générale. Philox gagne 5/5 comparaisons raw64 face au compteur scindé (rapport médian 1,04426) ; Threefry gagne 4/5 (1,08101), avec un fort ralentissement ponctuel. La géométrie et le volume diffèrent de la campagne v3 historique : la progression entre campagnes n'isole pas l'effet du code.

La réanalyse met en évidence un fort effet de première utilisation des pipelines spécialisés. Parmi 15 pipelines nouvellement préparés pendant la confirmation, 14 premières mesures sont sous 40 Go/s ; une seule seconde mesure reste sous ce seuil. Le rapport médian seconde/première utilisation vaut 3,90. Les formats lents changent avec l'ordre des jobs. La première/seconde comparaison change aussi de format et parfois de groupe : elle ne suffit pas à isoler une cause matérielle. Les 2574 mesures rapportent un état thermique 0 ; cela ne mesure ni fréquence GPU, ni occupation, ni concurrence. Des ralentissements ponctuels concernent aussi les kernels non spécialisés, Philox et Threefry. Les médianes précédentes ne permettent donc pas d'attribuer les chutes à 21 Go/s au seul calcul mantissa53 ou à une régression intrinsèque des kernels spécialisés. Toutes les observations sont conservées dans l'analyse. Aucune variante n'est promue.

Le programme précédent compile paresseusement les pipelines dans la campagne et ne chronomètre qu'un dispatch après deux échauffements pour chaque job/seed. Le temps de compilation est bien exclu du chronométrage GPU ; ses éventuels effets transitoires ne sont pas maîtrisés. Le `ratio/v3` reste correctement calculé comme médiane de rapports appariés, distincte du quotient des débits médians.

Une confirmation séparée est préparée dans `montmory-metal-confirm-m3` : mêmes kernels, configurations figées avant les premiers résultats de confirmation, sept nouvelles seeds, précompilation de tous les pipelines avant campagne, seize dispatches d'échauffement et cinq sorties chronométrées dans cinq buffers distincts, sur trois blocs par seed. Les 15 temps par seed sont résumés avant de comparer les sept seeds. Les temps d'échauffement restent disponibles et aucune mesure n'est filtrée selon sa vitesse. L'empreinte des kernels reste identique à celle du kit précédent. La nouvelle compilation Objective-C++/Metal et l'exécution native sont encore à effectuer sur le Mac.

### Confirmation native communiquée ensuite — 20261021T163954Z-22644

Le terminal utilisateur rapporte maintenant la compilation native réussie, la conformité et les 21 blocs de mesures terminés. La [transcription structurée](results/mac-confirm-reported-20261021T163954Z-22644.json) conserve les 22 lignes du résumé. L'archive brute de cette seconde campagne reste à recevoir.

En raw64, le résumé donne : témoin d'écriture 96,018 Go/s ; Threefry 92,082 ; Philox 88,979 ; Montmory spécialisé par paires 86,300 ; compteur scindé 85,218 ; référence v3 84,787. Le témoin d'écriture a donc le débit le plus élevé. `ratio/v3` compare chaque moteur à la référence Montmory v3 dans le même format, et représente une médiane de rapports appariés par seed.

Le compteur scindé gagne 7/7 comparaisons face à v3 dans chaque format, avec environ +0,5 % de gain médian apparié. Les spécialisations par paires peuvent afficher une meilleure médiane (jusqu'à +1,8 % raw64), mais ne gagnent que 4/7 comparaisons pour leurs meilleures lignes. Philox et Threefry gagnent 7/7 comparaisons face à v3 dans les deux formats ; leurs débits médians restent supérieurs à ceux des candidats Montmory. Ce résumé ne donne pas leurs victoires par seed face à chaque candidat.

Les médianes anormales à environ 21 Go/s des spécialisations ne figurent plus dans cette confirmation. Cela concorde avec l'effet de première utilisation observé dans la première archive. Le nouveau protocole modifie simultanément précompilation, échauffement, répétitions et buffers : il ne démontre pas une cause matérielle unique et ne garantit pas l'absence de ralentissements parmi les mesures individuelles. Les CSV permettront de vérifier ces points.

L'utilisateur a également communiqué le succès final de BigCrush sur `montmory_mulhi_pair` (160 statistiques, temps CPU 01:44:36.42), après SmallCrush (15) et Crush (144), seed 42, T=2048,D=4096. Le rapport intégral de cette exécution reste à recevoir. Les tests de conformité des variantes optimisées portent sur les sorties comparées ; ils ne constituent pas une nouvelle exécution des trois batteries sur chacune d'elles.

## Résultat CPU mesuré

La recherche utilise neuf seeds, quatre implémentations, 1/2/4/8/16 workers et un ordre mélangé par seed. Les sorties de chauffe et chronométrées sont comparées intégralement : 3 019 898 880 mots. Les temps incluent les écritures de 64 Mio et la création/jonction des threads ; les comparaisons sont hors chronométrage.

Les configurations choisies ont ensuite été figées et comparées sur cinq nouvelles seeds (167 772 160 mots supplémentaires comparés).

| Comparaison de confirmation | Indexé | Incrémental par paires | Rapport médian apparié par seed |
|---|---:|---:|---:|
| Un worker chacun | 2,124 Go/s | 3,069 Go/s | ×1,439 ; plus rapide 5/5 seeds |
| Meilleurs nombres de workers de la recherche | 11,047 Go/s à 16 workers | 17,856 Go/s à 8 workers | ×1,418 ; plus rapide 4/5 seeds |

Les débits correspondent à l'inverse de la médiane des temps ; le rapport apparié est la médiane des cinq rapports de temps pour une même seed. Ces deux façons de résumer ne sont pas interchangeables. La comparaison multicœur est bruitée (rapports de 0,82 à 2,20) : elle ne prouve pas un gain stable de cette ampleur. Le gain à un worker est plus régulier (1,27 à 1,47). Ces chiffres décrivent ce VPS partagé, avec GCC 12.2 et `-O3 -march=native`, pas le M3.

La précédente étude de lanes utilisait OpenMP avec une affinité rapprochée et trouvait quatre workers plus rapides. Cette étude emploie des threads C++ sans affinité imposée et des réécritures différentes, à un autre moment. Le classement précédent était propre à cette campagne ; nous n'attribuons pas la différence à une cause unique non mesurée.

Un contrôle séparé avec UBSan vérifie 66 016 mots aux frontières des lanes, aux extrémités de plages et pour des plages vides. Les anciennes sources C du générateur restent inchangées.

## Travail Metal livré

Le dossier `montmory-metal-opt-m3` contient le kit autonome, son protocole, les identités algébriques, les sources et le script de lancement. Les pistes sont : calcul par quatre avec mulhi, disposition des écritures entre threads, rotations spécialisées selon la seed, compteur en deux mots de 32 bits.

Les sept variantes passent 4256 cas avec 581 663 488 mots comparés à la référence C, plus 155 296 contrôles de composants, avec UBSan et gardes mémoire. Cela vérifie une transcription C++ des corps des kernels. Au moment de la livraison, la compilation Objective-C++/Metal et les mesures sur M3 restaient à faire. Le retour natif communiqué ensuite est décrit ci-dessus ; aucune exécution GPU locale n'a eu lieu sur le VPS.

Sur le Mac, sept seeds servent à choisir les réglages, puis cinq nouvelles seeds servent à les confirmer. Montmory v3, Philox et Threefry bénéficient du même choix de réglages. Le programme archive la sélection avant la confirmation. Les chiffres sont des mesures descriptives ; aucune variante n'est automatiquement promue en version de production.

## Reproduction et provenance

`python3 analyze_cpu.py` régénère les comparaisons CPU à partir des CSV. Le programme `src/cpu_benchmark.cpp` du kit produit la campagne de recherche ; `tools/cpu_confirm.cpp` fige les deux comparaisons choisies pour la confirmation sur x86. Il est compilé avec les mêmes options que la recherche. `tools/verify_cpu.cpp` est compilé avec `-O2 -fsanitize=undefined -fno-sanitize-recover=all`.

`results/environment.json` décrit la machine et les options. `results/interrupted-not-evidence.csv` est une exécution partielle interrompue, exclue des conclusions ; la campagne complète a été exécutée après la fin du vérificateur des kernels. Les autres services du VPS étaient actifs.

Cette étude ne modifie aucun ancien kit, ne relance pas TestU01 et n'affirme pas de nouvelle qualité statistique. L'objectif est de préserver le flux existant en réduisant le coût de son calcul.
