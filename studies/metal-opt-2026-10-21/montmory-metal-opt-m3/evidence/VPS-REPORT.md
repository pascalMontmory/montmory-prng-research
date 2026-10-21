# Optimisations Montmory à flux identique — 21 octobre 2026

Une réécriture CPU avec compteur incrémental et calcul par paires améliore le débit observé sur le VPS. Sept variantes Metal équivalentes sont livrées pour les mesures natives M3. La définition du flux reste fixée à T=2048 lanes logiques et D=4096 tirages par lane, comme le TestU01 Mac en cours.

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

Les sept variantes passent 4256 cas avec 581 663 488 mots comparés à la référence C, plus 155 296 contrôles de composants, avec UBSan et gardes mémoire. Cela vérifie une transcription C++ des corps des kernels. **La compilation Objective-C++/Metal et les mesures sur M3 restent à faire.** Aucun gain GPU supplémentaire n'est revendiqué à ce stade.

Sur le Mac, sept seeds servent à choisir les réglages, puis cinq nouvelles seeds servent à les confirmer. Montmory v3, Philox et Threefry bénéficient du même choix de réglages. Le programme archive la sélection avant la confirmation. Les chiffres sont des mesures descriptives ; aucune variante n'est automatiquement promue en version de production.

## Reproduction et provenance

`python3 analyze_cpu.py` régénère les comparaisons CPU à partir des CSV. Le programme `src/cpu_benchmark.cpp` du kit produit la campagne de recherche ; `tools/cpu_confirm.cpp` fige les deux comparaisons choisies pour la confirmation sur x86. Il est compilé avec les mêmes options que la recherche. `tools/verify_cpu.cpp` est compilé avec `-O2 -fsanitize=undefined -fno-sanitize-recover=all`.

`results/environment.json` décrit la machine et les options. `results/interrupted-not-evidence.csv` est une exécution partielle interrompue, exclue des conclusions ; la campagne complète a été exécutée après la fin du vérificateur des kernels. Les autres services du VPS étaient actifs.

Cette étude ne modifie aucun ancien kit, ne relance pas TestU01 et n'affirme pas de nouvelle qualité statistique. L'objectif est de préserver le flux existant en réduisant le coût de son calcul.
