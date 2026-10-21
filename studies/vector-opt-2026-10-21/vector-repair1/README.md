# Reprise du kit vectoriel — correctif 1

À extraire dans le répertoire `montmory-vector-m3`, puis lancer `bash vector-repair1/run.sh`.

Le programme de mesures Objective-C++ utilisait `auto` pour le retour `id` de `mutableCopy`. Clang Apple rejette cette déclaration avec `-Werror,-Wauto-var-id`. Le type explicite `NSMutableDictionary*` corrige cette ligne. Les options strictes du compilateur sont conservées.

Le correctif compile le programme de mesure, relance seulement le contrôle de conformité CPU, puis reprend les vérifications et mesures Metal. Il ne relance pas les campagnes CPU et ne modifie pas les fichiers du kit original. Il vérifie les manifestes original et correctif, produit un nouveau dossier de résultats et une nouvelle archive, y compris en cas d'échec. Les résultats CPU du lancement utilisateur `20261021T170347Z-22732` sont également copiés dans la nouvelle archive s'ils sont présents.

Une correction du vérificateur rend aussi ses cas aléatoires identiques entre compilateurs : l'expression `test(random(), random()>>1, random()%8192)` ne fixait pas l'ordre d'évaluation des arguments en C++17. Le GCC du VPS prenait le premier tirage comme taille ; Clang sur le Mac prenait le troisième. Cela explique les comptes respectifs de 7 484 008 et 7 353 208 mots. Les deux exécutions passaient leurs cas respectifs. Trois déclarations successives fixent maintenant l'ordre seed, base, taille ; le compteur attendu est 7 353 208 sur les deux plateformes. Cette correction concerne les entrées des tests, pas la définition ou l'implémentation du PRNG.

Les sources CPU du moteur, tous les kernels Metal, les seeds de performance, les réglages, les chauffes et les calculs de ratios restent ceux du kit initial. Aucun nouveau résultat natif Metal n'est revendiqué avant l'exécution sur le Mac. Aucun nouveau TestU01/TACM.
