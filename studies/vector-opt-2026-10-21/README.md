# Optimisation conservant le flux — 21 octobre 2026

Demande : poursuivre l'optimisation CPU/Metal sans perdre les propriétés déjà observées du générateur. Les kits précédents restent figés. Le nouveau kit est dans `montmory-vector-m3`.

## Résultat mesuré sur le VPS

Le pool de quatre workers réutilisables améliore les appels sur 65 536 mots. Lors de la recherche : médiane 5,290 Go/s contre 3,306 avec quatre threads recréés par appel ; ratio apparié médian 1,600, neuf seeds sur neuf plus rapides.

La confirmation séparée sur sept nouvelles seeds, avec des bases décalées franchissant les frontières de lanes, donne un ratio apparié médian **1,705** face aux quatre threads recréés, sept seeds sur sept plus rapides. Face au scalaire par paires sur un seul worker, ratio **2,014**, sept sur sept. Il s'agit de petits blocs et du temps appelant ; la construction initiale du pool est exclue. Le réveil et la synchronisation sont inclus. Résultats propres à ce VPS virtualisé et à ces campagnes.

Sur les grands blocs, le résultat du pool dépend du nombre de workers ; aucun gain général n'est retenu. Les variantes SIMD essayées sont plus lentes que le scalaire par paires sur x86. Elles sont conservées comme candidats expérimentaux ARM/Metal, sans sélection automatique ni gain annoncé sur ces matériels.

## Vérifications effectuées

- Arithmétique SIMD, formats, frontières, débordements intentionnels, plages rejetées et découpages : 7 484 008 mots.
- Composants : 120 000 comparaisons, dont produits hauts, rotations et mélange final.
- Deux blocs historiques consécutifs pour trois seeds : 50 331 648 mots comparés à l'état C récupéré.
- Partitionnement par 1, 2, 3, 7, 8 workers et sous-blocs impairs : 655 445 mots.
- 500 appels de pools réutilisés, avec tailles irrégulières et workers inactifs : 1 918 268 mots.
- Ces contrôles passent sous UBSan avec compilation pour le CPU local. La passe ASan+UBSan x86 générique passe avec la détection de fuites désactivée : LeakSanitizer ne peut pas terminer sous le ptrace de cet environnement ; aucun succès LeakSanitizer revendiqué.
- Campagne CPU complète : 360 sorties chronométrées et leurs chauffes, **3 043 491 840 mots comparés**. Les résumés et ratios ont été recalculés depuis les CSV.
- Confirmation du pool : 315 sorties chronométrées, **20 643 840 mots comparés**, plus les chauffes contrôlées.
- Transcription C++ SIMD des corps Metal générés : **65 526 272 mots**, 1204 cas, sous UBSan. Elle couvre les écritures contiguës/entrelacées, les threads inactifs et les réglages irréguliers. Ce n'est pas une exécution Metal native.

Les décomptes sont des comparaisons d'implémentations, pas des échantillons statistiques indépendants. Aucun nouveau TestU01 ni TACM exécuté. Les fonctions et constantes originales restent inchangées ; l'API d'indice global fixe la géométrie T=2048,D=4096 et permet de changer le découpage sans changer le flux.

## Livrable et limites

`montmory-vector-m3/run.sh` vérifie les 1600 vecteurs ARM64/x86, les candidats CPU sous UBSan, mesure deux volumes CPU, confirme séparément le pool, puis effectue la recherche Metal et sa confirmation sur de nouvelles seeds. Toute sortie chronométrée est comparée. Les archives sont locales, aucune transmission automatique.

Au moment de la livraison, les compilations et mesures natives ARM64/Metal restaient à exécuter sur le Mac de l'utilisateur ; les retours natifs sont consignés ci-dessous. Aucun matériel CUDA disponible pour cette nouvelle campagne. La dernière archive brute de confirmation Metal précédente et l'archive complète BigCrush ne sont toujours pas présentes au début de ce travail.

Sources, contrat et identités : `montmory-vector-m3/src`, `montmory-vector-m3/EXACT.md`. Preuves d'exécution, CSV et analyse : `results`. Résultats initiaux SIMD seuls conservés dans `results/initial-vector-only`.

## Retour Mac et correctif de reprise

Le terminal utilisateur `20261021T170347Z-22732` rapporte les vecteurs ARM64/x86 et les contrôles CPU natifs PASS, puis toutes les comparaisons des mesures CPU PASS. La confirmation des petits blocs de 65 536 mots donne un ratio médian apparié de **1,307** pour quatre workers réutilisés contre quatre recréés, avec sept victoires sur sept nouvelles seeds. Face au calcul par paires séquentiel : 3,454. Sur les grands blocs, le pool à huit workers atteint 12,203 Go/s, ratio apparié 1,013, huit victoires sur neuf ; le gain est donc beaucoup plus petit. Données arrondies conservées dans `results/mac-reported-20261021T170347Z-22732.json`, archive brute non reçue à ce stade.

La phase Metal s'arrête avant exécution : Clang Apple rejette `auto result=[jobJSON(j) mutableCopy]` sous `-Werror,-Wauto-var-id`. `vector-repair1` fournit le type explicite `NSMutableDictionary*`, sans modification des kernels ou mesures. Il reprend les contrôles et la campagne Metal en conservant les résultats CPU déjà produits. Le kit initial reste figé.

L'écart entre les comptes de conformité CPU sur GCC et Clang provenait de l'ordre d'évaluation non imposé de trois appels au générateur des cas de test dans les arguments d'une même fonction. Le correctif séquence ces tirages. La vérification corrigée passe sous GCC+UBSan avec les mêmes 7 353 208 mots de plages que sur le Mac ; ni la définition du PRNG ni ses sorties ne sont modifiées. La compilation et l'exécution natives Metal corrigées restent à effectuer sur le Mac.

Archive de reprise : `/home/pascal/prng-recovery/mac-montmory-vector-repair1.tar.gz`. Intégrité de l'archive, manifestes et syntaxe du script vérifiés ; détails dans `results/repair1-delivery.json`.

## Retour natif du correctif — 20261021T170953Z-22809

Le terminal utilisateur confirme la compilation du programme corrigé, les contrôles CPU et la conformité Metal : 8 540 160 mots et 4 004 592 réductions, six vecteurs officiels CPU et GPU. Les cinq seeds de recherche puis sept nouvelles seeds de confirmation passent les comparaisons de toutes les sorties chronométrées. Le compteur CPU corrigé vaut bien 7 353 208 mots de plages sur ARM64 comme sur le VPS. Ce retour clôt l'incertitude de compilation native du correctif ; il ne remplace pas la vérification de l'archive brute, qui reste à recevoir.

| Moteur | raw64, Go/s | mantissa53, Go/s | Ratio médian apparié face à v3 | Seeds plus rapides que v3 |
|---|---:|---:|---:|---:|
| Montmory v3, paires | 84,782 | 84,787 | 1,000 | référence |
| Montmory, compteur scindé | 85,218 | 85,177 | 1,004 | 7/7 dans les deux formats |
| Montmory vector4 | 76,576 | 76,608 | 0,904 | 0/7 |
| Montmory vector4 entrelacé | 76,729 | 76,736 | 0,905 | 0/7 |
| Philox4x32-10 | 89,048 | 89,073 | 1,050 / 1,051 | 7/7 |
| Threefry2x32-20 | 92,500 | 92,405 | 1,090 / 1,090 | 7/7 |
| Témoin d'écriture, sans PRNG | 96,369 | 96,786 | 1,137 / 1,142 | 7/7 |

Décision : les deux variantes vectorielles Metal sont correctes dans les cas exécutés mais ne sont pas retenues pour la performance. Le compteur scindé retrouve un petit gain régulier, environ 0,4 % en ratio médian apparié. La différence avec le quotient des débits médians, environ 0,5 %, provient des agrégations différentes ; les valeurs arrondies ne permettent pas de recalculer exactement les ratios appariés.

Les réglages observés du compteur scindé sont 262 144 threads et groupes de 64 pour raw64, 131 072 threads et groupes de 128 pour mantissa53, toujours T=2048,D=4096. Ils sont conservés dans `results/m3-observed-settings.json` comme réglages provisoires de cette charge et de cette machine, sans modification d'une bibliothèque déployée ou promesse d'optimalité.

Sur raw64, les débits médians affichés placent Philox environ 4,5 % et Threefry environ 8,5 % devant le meilleur Montmory de cette campagne. On ne peut pas déduire leurs comparaisons seed par seed face au compteur scindé à partir des seuls ratios face à v3. Aucun avantage de vitesse sur ces deux concurrents n'est démontré ici.

La cause matérielle du recul des variantes vectorielles n'a pas été profilée ; pression sur les registres ou ordonnancement des instructions restent des hypothèses. Aucun nouveau test statistique n'a été exécuté. L'identité des sorties contrôlées confirme la conservation du flux pour les exécutions testées, pas une indépendance statistique universelle des lanes.

Les données du terminal sont dans `results/mac-repair1-reported-20261021T170953Z-22809.json`. L'archive utilisateur attendue est `montmory-vector-repair1-results-20261021T170953Z-22809.tar.gz` ; elle contient aussi les résultats CPU du lancement précédent lorsque le répertoire initial est présent.
