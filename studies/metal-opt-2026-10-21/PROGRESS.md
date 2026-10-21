# État du travail

- Trois variantes CPU incrémentales ajoutées à la version indexée ; protocole T=2048,D=4096 inchangé.
- Campagne CPU complète : 180 sorties chronométrées plus leurs chauffes, 3 019 898 880 mots comparés, PASS.
- Confirmation sur cinq nouvelles seeds : 20 sorties, 167 772 160 mots comparés, PASS. Gain médian apparié à un worker : 43,9 %, candidat plus rapide sur les cinq seeds.
- Cas limites CPU : 66 016 mots, gardes et UBSan, PASS.
- Sept kernels Metal candidats écrits ; contrôles portables : 581 663 488 mots, 155 296 contrôles de composants, PASS.
- Comparatif natif avec recherche puis confirmation sur nouvelles seeds préparé. Exécution M3 attendue après les tests déjà en cours sur le Mac.
- Aucun changement de la définition du flux ; aucune promotion automatique et aucun nouveau PASS BigCrush revendiqué.

## Retour utilisateur — exécution M3 20261021T162407Z-22542

- Le terminal rapporte la compilation native réussie, 1600 vecteurs ARM64/x86 conformes, 7 716 864 mots et 4 004 592 réductions conformes, six vecteurs officiels CPU/GPU, puis toutes les sorties de recherche et confirmation conformes.
- CPU : `incremental_pair`, huit workers, 12,487 Go/s contre 9,746 pour `indexed`, soit +28,1 % de débit dans cette exécution. Pas de comparaison CPU contre Philox/Threefry dans ce kit.
- Metal raw64 : référence v3 85,123 Go/s ; compteur scindé 85,580 ; Philox 89,330 ; Threefry 92,474. Le nouveau candidat n'apporte qu'environ 0,5 % face à v3 dans ce résumé ; aucune supériorité sur les concurrents démontrée.
- Les variantes spécialisées présentent des ralentissements à examiner. Le calcul du ratio apparié a été relu dans `src/benchmark.mm` ; sa différence avec un quotient de médianes est attendue et ne suffit pas à conclure à une erreur de mesure.
- Transcription structurée : `results/mac-reported-20261021T162407Z-22542.json`. CSV bruts, métadonnées et archive native non reçus à ce stade. Sources du kit livré inchangées.
- BigCrush a entre-temps été annoncé PASS par le terminal utilisateur pour `montmory_mulhi_pair`, seed 42, T=2048,D=4096 : 160 statistiques, 01:44:36.42 CPU. SmallCrush et Crush avaient aussi été annoncés PASS. Attente de l'archive complète ; aucune extrapolation à toutes les seeds ou variantes.

## Archive native reçue et réanalysée

- Archive optimisation reçue : 143 726 octets, SHA-256 `053d4930cf56681df3afdc242675e8308967b551f2e70ea752ac12c3eaaa0042`. Extraction conservée sous `received/mac-20261021T162407Z-22542`. L'archive complète BigCrush n'est pas présente à la racine de récupération.
- Les 62 empreintes du kit, les vecteurs et les copies de protocole concordent. Les 144 lignes CPU, 2464 lignes GPU de recherche et 110 de confirmation sont complètes ; résumés, unités et sélection recomputés avec `analyze_mac.py`.
- Les traces rapportent 2 415 919 104 comparaisons de mots CPU et 43 184 553 984 pour la campagne GPU, auxquelles s'ajoute la conformité initiale. Ce sont des comparaisons répétées, pas autant de mots statistiques distincts.
- CPU par paires : gagne 9/9 comparaisons à 1, 2, 4 et 8 workers. À 8 workers, rapport médian apparié 1,27882 ; rapport de médianes 1,28127.
- GPU compteur scindé : gagne 5/5 comparaisons dans chaque format, mais gain médian apparié limité à 0,51–0,56 %. Aucun classement général de qualité/performance revendiqué.
- Effet de première utilisation constaté : 14/15 premières mesures de pipelines spécialisés sous 40 Go/s, contre 1/15 secondes mesures. Rapport médian seconde/première 3,90. Le format et parfois le groupe changent entre ces mesures : cause précise à isoler. Les états thermiques valent tous 0 mais ne renseignent pas la fréquence GPU. Certains moteurs non spécialisés ralentissent ponctuellement aussi.
- Cette observation corrige l'interprétation provisoire des spécialisations : les seules médianes de confirmation ne justifient pas d'attribuer le ralentissement au format mantissa53 ni d'écarter les kernels pour leur coût intrinsèque. Aucune observation n'est supprimée de l'analyse.
- Confirmation séparée préparée dans `montmory-metal-confirm-m3` : kernels inchangés, réglages figés, sept nouvelles seeds, précompilation préalable, seize échauffements puis cinq chronométrages par bloc, trois blocs par seed, contrôle intégral de chaque sortie chronométrée. Mesures natives attendues.
- Kit de confirmation livré : `/home/pascal/prng-recovery/mac-montmory-confirm-kit.tar.gz`, 128 403 octets, SHA-256 `0f8ee315a4072636f3cc807370196f5c224c10814659532e834ba5a0a7182506`. Les 50 fichiers du manifeste ont été revérifiés dans l'archive. Contrôles C++ portables sous UBSan, six KAT officiels CPU, correspondance des 22 réglages figés, séparation des sept nouvelles seeds, identité des kernels et syntaxe bash : PASS. Compilation Objective-C++/Metal native non effectuée sur le VPS.

## Confirmation native rapportée — 20261021T163954Z-22644

- Terminal utilisateur : conformité PASS et 21 blocs de chronométrage terminés. Données arrondies conservées dans `results/mac-confirm-reported-20261021T163954Z-22644.json` ; archive brute de confirmation non reçue à ce stade.
- Compteur scindé : 85,218 Go/s dans les deux formats ; ratio médian apparié 1,005 et 7/7 victoires face à v3. Le petit gain est retrouvé sur ce second jeu de seeds.
- Meilleure médiane Montmory raw64 : `const_pair`, 86,300 Go/s, mais seulement 4/7 victoires face à v3. Philox raw64 88,979 Go/s, Threefry 92,082 ; chacun gagne 7/7 face à v3.
- Le témoin d'écriture raw64 atteint 96,018 Go/s : débit supérieur à Threefry. La colonne ratio est relative à Montmory v3.
- Les médianes de spécialisation à environ 21 Go/s ne réapparaissent pas dans le résumé. Inspection des mesures individuelles et des échauffements encore nécessaire ; cause matérielle unique non isolée. Aucune nouvelle batterie TestU01 exécutée par cette campagne.

## Suite séparée : optimisation à flux inchangé

- Travail poursuivi dans `/home/pascal/prng-recovery/vector-opt-2026-10-21` ; aucun kit précédent modifié.
- API CPU à indice global, géométrie logique fixée, SIMD expérimental ARM/x86 et pool réutilisable ; deux candidats vectoriels Metal.
- Contrôles CPU, frontières de lanes, découpage et pool conformes ; transcription des nouveaux corps Metal conforme sous UBSan. Native ARM/Metal à exécuter sur le Mac.
- Sur le VPS, SIMD plus lent que le scalaire par paires ; aucune promotion x86. Le pool de quatre workers améliore les petits blocs de 65 536 mots : confirmation séparée sur sept nouvelles seeds, ratio médian apparié 1,705 contre quatre threads recréés, sept victoires sur sept. Aucune extrapolation aux grands blocs, au M3 ou aux concurrents.
- Les résultats TestU01 historiques gardent leur périmètre. Aucun nouveau test statistique exécuté.
