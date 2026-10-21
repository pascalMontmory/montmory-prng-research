# État au 21 octobre 2026

- Copie Mac complète inventoriée ; original conservé, instantané de travail figé.
- Sources C/Metal et journaux historiques retrouvés ; empreintes de l'archive vérifiées.
- Modèle exact, inverse du Feistel et certificat de non-injectivité du mélange vérifiés.
- Sept nouveaux SmallCrush réussis sur x86 ; celui reproduisant le flux Metal
  retrouve les quinze p-values imprimées du journal Mac.
- 3 072 cas C/Python, 9 536 valeurs de kernel émulé et 3 600 108 mots sous
  différentes répartitions OpenMP concordent.
- Le contrat logique et l'ordre des sorties sont explicités ; aucune modification
  du générateur historique ni publication n'a été effectuée.
- Antériorité confirmée : Random123 inclut Metal au moins depuis 2021 et documente
  les trois batteries ; le port Threefry CUDA de MLX est daté de juin 2025.
- Sauvegarde CUDA OVH reçue ensuite : 23 fichiers figés, cœur C identique au
  cœur Metal, journaux SmallCrush/Crush/BigCrush réussis. Leurs p-values imprimées
  correspondent à celles du Metal ; le journal BigCrush du dépôt est le même
  après renommage du générateur.
- 251 904 mantisses CUDA/Metal/référence concordent après adaptation sur CPU.
- Débordement du kernel CUDA pour T non multiple de 256 reproduit ; correction
  séparée avec borne et paramètres communs, sorties conservées et sentinelles
  vérifiées. Wrapper relu mais pas encore compilé par nvcc ni exécuté sur GPU.
- Les trois anciens comparatifs CUDA réutilisent un compteur et une clé Philox
  altérés : écart au standard dès le deuxième bloc, 48 cas reproduits. Unités
  GB/s erronées, aucun journal de débit retrouvé dans cette sauvegarde.
- Comparatifs CPU officiels Random123 : 1 760 mesures, 11 répétitions, huit
  variantes, génération en volume et accès isolés, options natives/génériques,
  1/4 workers. Gain conditionnel face à Philox4x32 ; Threefry dispose de variantes
  plus rapides. Aucun avantage général établi.
- Contrôles natifs ARM/Metal exécutés par l'utilisateur sur Apple M3, retour
  terminal PASS reçu. Le contrôle CUDA corrigé reste à exécuter sur GPU.
- Précision M3 : Threefry2x32-20 est implémenté en Metal dans MLX ; les variantes
  64 bits sont désactivées dans le port Metal par défaut de Random123 examiné.
  Les mesures Threefry4x64 rapportées sont CPU x86, pas des résultats Metal.
- Kit M3 : 17 047 296 valeurs vérifiées préalablement en adaptation C++ avec
  UBSan, puis 1 600 vecteurs ARM/x86 et 909 312 mots GPU/références rapportés PASS
  sur le Mac. Six vecteurs officiels CPU et GPU réussis, 11 répétitions achevées.
- Médianes M3 reçues pour 32 Mio par lancement : en 53 bits, Montmory historique
  54,996 GB/s, Threefry 84,272 GB/s, Philox 82,944 GB/s (meilleurs réglages testés).
  Aucun gain observé pour la version Montmory réorganisée. Les concurrents sont
  plus rapides dans cette campagne ; pas d'avantage M3 démontré pour Montmory.
- Archive brute M3 non encore reçue ; analyse de dispersion et métadonnées en
  attente. Débit CUDA, énergie et avantage commercial non établis.
- Deuxième kit Metal candidat préparé séparément : calcul décomposé sur 32 bits,
  compteur avancé par additions, ou combinaison des deux ; même fonction et
  même ordre des sorties. Identités exactes documentées, références inchangées.
- 60 883 200 mots et 7 432 282 comparaisons arithmétiques supplémentaires passent
  sur l'adaptation C++ avec UBSan.
- Retour natif du deuxième kit reçu (`20261021T131521Z-8925`) : compilation
  ARM64/Metal, 1 600 vecteurs ARM/x86, 4 153 344 mots GPU/références et six
  vecteurs officiels CPU/GPU PASS. Onze répétitions, 43 configurations.
- Meilleur Montmory v2 : 62,969 GB/s brut, 62,983 GB/s en 53 bits. Gain observé
  de 13,8 % sur l'historique 53 bits et d'environ 17 % sur le meilleur indexé
  initial dans la même campagne. En 53 bits, Philox et Threefry gardent environ
  29 % et 32 % de débit supplémentaire. Rapports de médianes affichées ; aucune
  analyse de dispersion sans les fichiers bruts, encore absents du VPS.
- Kits v1/v2 et leurs empreintes conservés sans modification. Les optimisations
  passent les contrôles natifs d'identité ; les batteries TestU01 n'ont pas été
  relancées par ce benchmark. Aucun avantage général sur les concurrents établi.
- Kit v3 préparé : cinq candidats supplémentaires (lots de deux/quatre sorties,
  produit haut, rotations spécialisées), tailles de groupes comparées, diagnostic
  avec XOR par thread, coûts de compilation séparés et capture Metal facultative.
- Vérifications v3 sur CPU avec UBSan : 199 824 000 mots, 73 967 040 réductions,
  9 713 754 comparaisons de composants concordent. SmallCrush de `spec_dual`
  adapté sur CPU réussi ; quinze p-values imprimées historiques reproduites.
- Lanceur des trois batteries natives livré avec la source TestU01 figée.
  Son état de compilation et ses résultats natifs sont précisés ci-dessous.
- Retour du benchmark natif v3 reçu (`20261021T134922Z-9053`) : compilation
  ARM64/Metal et spécialisation réussies, 1 600 vecteurs ARM/x86, 4 835 328 mots
  complets, 4 004 592 réductions et six vecteurs officiels CPU/GPU annoncés PASS.
  Onze répétitions achevées ; 105 configurations en volume et neuf diagnostics
  prévus par seed. Le terminal ne donne que les meilleurs réglages en volume.
- `montmory_mulhi_pair` retenu provisoirement pour les batteries natives :
  72,309 GB/s brut et 72,786 GB/s en 53 bits, sans spécialisation par rotations.
  Dans cette campagne, gain de 31,8 % sur l'historique 53 bits et de 16,5 % sur
  la variante v2. Les 0,23 % supplémentaires de `spec_dual` en 53 bits ne sont
  pas statistiquement établis. Lots de quatre sans gain convaincant.
- Diagnostic avec XOR par thread : 10,809 Gmots/s pour `mulhi_pair`, 8,644 pour
  la variante v2, 11,679 pour Philox et 13,061 pour Threefry. Le gain Montmory
  apparaît aussi avec peu d'écritures ; ce mode ne mesure pas le calcul pur.
- En volume 53 bits, Philox et Threefry gardent respectivement 16,3 % et 21,4 %
  de débit supplémentaire sur `mulhi_pair`. Aucun avantage M3 sur ces références
  démontré. Transcription v3 et extraction conservées ; archives brutes des trois
  campagnes encore absentes. Les trois kits livrés restent inchangés.
- Première tentative TestU01 native v3 (`20261021T140817Z-9132`) arrêtée au
  premier essai de liaison C pendant `configure`, avant le wrapper et les
  batteries. Détection ARM réussie ; résumé `C compiler cannot create executables`.
  Le détail du `config.log` n'est pas encore reçu. Complément séparé
  `mac-metal-testu01-repair-v1/` : SDK macOS explicite, environnement de
  compilation isolé, extraction neuve, programme C ARM64 préalable et diagnostics
  visibles. Cinq scénarios d'orchestration vérifiés sur Linux avec outils simulés ;
  état de préparation avant le retour natif décrit ci-dessous. Générateur inchangé.
- Retour natif du lanceur corrigé : programme C ARM64 préalable PASS, puis
  exécution réelle du wrapper Metal/TestU01. Conformité de 4 835 328 mots et
  4 004 592 réductions PASS ; SmallCrush annonce le succès de ses 15 statistiques
  pour `montmory_mulhi_pair`, seed 42, géométrie historique 2 048 × 4 096.
  Les quinze p-values imprimées concordent avec la référence historique.
  Temps CPU annoncé : 4,74 s, sans interprétation comme mesure de débit GPU.
  Contrôle préalable au processus suivant PASS ; résultats suivants consignés
  ci-dessous. Extraits et comparaison conservés ; archive brute et
  identifiant de ce lancement non encore reçus. La cause exacte du premier
  échec reste non isolée malgré le succès du nouveau lanceur.
- Premier extrait Crush natif : 49 p-values intermédiaires reçues, toutes dans
  l'intervalle par défaut de TestU01 [0,001 ; 0,999], avant réception du résumé final.
  Correction d'une description trop large du flux : le wrapper v3 utilise
  2 048 × 4 096 pour toutes les batteries, ce qui reproduit le paramétrage
  historique SmallCrush seulement. Les en-têtes CUDA archivés indiquent
  4 096 × 8 192 pour Crush et 8 192 × 16 384 pour BigCrush. Les anciennes
  p-values ne sont donc pas directement comparables aux nouvelles pour ces
  deux batteries. Le lancement en cours reste utile et n'est pas interrompu.
- Résumé final Crush natif reçu : les 144 statistiques passent avec
  `montmory_mulhi_pair`, seed 42, géométrie fixe 2 048 × 4 096, uniformes 53 bits.
  Temps CPU annoncé : 15 min 35,72 s. Résumé et extraction conservés ;
  aucune égalité de p-values avec l'ancien flux Crush n'est revendiquée.
  Nouveau contrôle de conformité PASS avant la batterie suivante ; BigCrush,
  l'archive brute et l'identifiant de cette exécution restent attendus.
- Extrait ultérieur : démarrage BigCrush confirmé explicitement, seed 42,
  2 048 lanes et 4 096 draws. Onze p-values imprimées (deux SerialOver,
  neuf CollisionOver), de 0,06 à 0,98 ; le dernier bloc est tronqué après
  sa p-value. Extraction conservée. Bilan final BigCrush et archive brute
  toujours attendus ; le résumé Crush répété n'est pas une seconde exécution.
