# Montmory : optimisation à flux inchangé

Ce kit expérimental compare les implémentations ; il ne remplace pas les versions déjà validées. L'objectif est de conserver les valeurs et leur ordre, puis de mesurer si une nouvelle manière de les calculer est plus rapide.

## Ce qui change

- CPU : quatre mots traités ensemble en SIMD, ou deux groupes de quatre. Backend NEON sur ARM64 ; backend x86 séparé. Le calcul scalaire par paires reste le témoin.
- CPU : pool de workers réutilisables, afin d'éviter leur création et leur jonction à chaque petit bloc. Temps de réveil, synchronisation et génération inclus ; construction initiale du pool exclue et explicitement signalée. Aucune affectation forcée à un cœur physique.
- Metal : deux variantes `uint4`, à écritures contiguës ou entrelacées. Recherche du nombre de threads et de la taille des groupes, puis confirmation des réglages retenus sur de nouvelles seeds.

Une instruction SIMD traite plusieurs mots ; elle ne crée pas de nouvelles lanes logiques du générateur. Elle ne garantit pas un gain. Les premières mesures x86 de ce travail sont défavorables aux variantes SIMD ; leur intérêt éventuel sur ARM et Metal reste à mesurer.

## Contrat conservé

T=2048 lanes logiques, D=4096 mots par lane dans chaque bloc historique de 8 388 608 mots. Les constantes, les trois tours Feistel et le mélange final restent inchangés. `raw64` conserve les 64 bits. `mantissa53` vaut exactement `raw64 >> 11`, stocké dans un entier de 64 bits.

L'API CPU `src/stream.hpp` utilise un indice **global** :

```
lane(k) = floor(k / 4096) mod 2048
gamma(k) = 0x9E3779B97F4A7C15 + 2*lane(k)
counter(k) = seed + (k+1)*gamma(k) mod 2^64
```

Les blocs GPU de cette campagne gardent la géométrie historique. L'API CPU autorise des sous-blocs arbitraires sans réinitialiser les lanes : le caller doit transmettre leur indice global. Sa limite de plage interdit le bouclage de cet indice, tout en conservant les débordements intentionnels du compteur. Changer T ou D serait un changement de flux, et ne fait pas partie de cette optimisation.

## Exécution

Sur le Mac : `bash run.sh`. Le script vérifie les fichiers et les vecteurs x86/ARM64, exécute les contrôles natifs avec UBSan, mesure le CPU, confirme séparément le pool, puis mesure Metal. Il crée une archive locale de résultats, même en cas d'échec. Aucun envoi automatique.

Les essais CPU portent sur 65 536 mots et 8 388 608 mots ; le petit bloc est un préfixe du même flux, avec le même D. Les nombres de workers sont 1, 2, 4 et 8, dans la limite rapportée par le système. Les pools inactifs restent endormis.

La campagne Metal mesure Montmory v3, le compteur scindé précédent, les deux nouveaux candidats, Philox4x32-10, Threefry2x32-20 et le témoin d'écriture. Même volume, même format, même grille de recherche. Tous les pipelines de cette campagne sont préparés avant les mesures. Seize chauffes précèdent chaque bloc ; toutes les mesures sont conservées, sans retrait d'observations. Chaque sortie chronométrée est comparée entièrement après les chronométrages. Les réglages sont figés avant les sept seeds de confirmation. Les ratios finaux sont appariés par seed ; ils ne constituent pas une garantie pour toutes les machines ou tous les usages.

## Portée de la validation

Les contrôles vérifient l'arithmétique, les limites de plages, les passages entre lanes, le découpage en sous-blocs, les workers et les sorties natives. Ils ne prouvent pas l'indépendance statistique entre toutes les lanes.

Le terminal utilisateur a rapporté SmallCrush, Crush et BigCrush PASS pour `montmory_mulhi_pair`, seed 42, via les uniformes de 53 bits. L'archive complète de cette exécution n'a pas encore été reçue au début de ce travail. Ce kit ne relance pas ces batteries, et ne prétend pas avoir testé directement tous les bits de `raw64` avec elles.

Si toute la suite et son adaptateur sont identiques, les données fournies à TestU01 le sont aussi. Les identités arithmétiques sont décrites dans `EXACT.md`. Les comparaisons finies vérifient les cas exécutés et ne constituent pas une preuve universelle des compilateurs.

Les valeurs de débit du VPS concernent un CPU x86 virtualisé ; elles ne prédisent pas celles du M3. Les candidats plus lents ne sont pas adoptés. Aucun changement de paramètres pour améliorer artificiellement les p-values, aucune couche de Markov et aucune revendication cryptographique.
