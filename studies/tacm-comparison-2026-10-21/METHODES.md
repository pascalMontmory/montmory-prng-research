# Hypothèses, calculs et limites

Cette expérience utilise la révision mathématique [TACM v1.2](../../FondamentalWork-maths-audit-20261019/Math/PRNG/WIP/tacm-diagnostic-formulas-v1-2.md) pour les idées de variance et de calibration, avec le protocole autonome conservé ici. Les conclusions sont expérimentales ; aucune indépendance des sorties déterministes d'un PRNG n'est démontrée.

Sous le modèle de référence U,V indépendants et uniformes continus sur [0,1), les quantités exactes sont :

| Fonction | Espérance | Variance d'une observation | Nombre d'observations par répétition |
|---|---:|---:|---:|
| U | 1/2 | 1/12 | N |
| U² | 1/3 | 4/45 | N |
| sin(2πU) | 0 | 1/2 | N |
| 1{U > 0,99} | 1/100 | 99/10000 | N |
| UV | 1/4 | 7/144 | N/2 |
| 1{U > 0,99 et V > 0,99} | 1/10000 | 9999/100000000 | N/2 |

Les identités suivent de l'intégration, de `E[U^r]=1/(r+1)`, de l'indépendance et de la variance d'une Bernoulli. Pour UV : `E[(UV)^2]=1/9`, d'où `1/9−1/16=7/144`. Sous indépendance des observations, la variance de leur moyenne est σ²/m. Une meilleure qualité d'un PRNG ne fait pas mécaniquement baisser cette variance idéale commune à tous les moteurs IID ; une réduction particulière pourrait provenir de dépendances favorables à une fonction, sans avantage universel.

Les couples sont `(u[0],u[1]), (u[2],u[3]), …`. Aucun chevauchement n'est utilisé. Les trois histogrammes ont 16, 64 et 16 cases. Les références de Pearson sont respectivement χ² à 15, 63 et 15 degrés de liberté ; ce sont des limites asymptotiques, pas des lois finies exactes. Le test inter-lanes utilise 64 cases sur 1 024 couples de lanes × 16 positions, soit 16 384 observations par seed. Sous le modèle IID les cases ont toutes probabilité 1/64.

La grille réelle des uniformes est de taille M=2^53. Même si les mots bruts étaient exactement uniformes, `E[U]=(M−1)/(2M)` et `Var(U)=(M²−1)/(12M²)`. Pour le seuil flottant Python 0,99, la probabilité exacte sur cette grille est `90071992547409/9007199254740992`, soit environ `0,01 − 1,0214×10^-16`. Les références continues et binomiales nominales sont donc des approximations à l'échelle de la grille. Cet écart est négligeable devant l'incertitude de cette expérience, mais n'est pas présenté comme exactement nul. L'uniformité exacte de Montmory lui-même n'est pas supposée démontrée.

Dans un groupe de R=16 estimations, `s²` désigne la variance empirique avec diviseur R−1. La statistique `(R−1)s²/(σ²/m)` suit exactement χ²(R−1) si les estimations sont normales indépendantes et de variance σ²/m. Ici les quatre moyennes continues sont seulement approximativement normales par le théorème central limite sous le modèle IID ; le PIT de variance est donc approximatif. Les deux fonctions rares sont exclues de ce PIT.

Pour un effectif rare K sous le modèle nominal Bin(m,p), le PIT est `F(K−1)+A·P(K)`, avec A uniforme indépendant. Son uniformité est exacte dans ce modèle idéal, y compris lorsque K est discret. L'implémentation emploie un auxiliaire reproductible PCG64DXSM sur 53 bits, avec namespace propre au moteur et à la fonction ; son calcul est explicite dans `analyze.py`. Un simple mid-p ne serait pas interchangeable avec cette construction.

Pour chaque fonction et chaque référence, on calcule dans chacun des 128 groupes la différence entre les MSE moyennes de Montmory et de la référence. Un test t unilatéral approximatif sur ces différences recherche une baisse de MSE. Les 54 p-values sont multipliées par 54 et plafonnées à 1. La règle d'avantage exige aussi un rapport global <0,90 et un rapport <1 dans chacune des deux moitiés de 64 groupes.

Les intervalles de rapport exportés sont descriptifs et approximatifs : méthode delta sur le logarithme du rapport, variance estimée à partir des groupes appariés, quantile t à 127 degrés de liberté et correction bilatérale pour les 54 comparaisons. Ils ne donnent pas une preuve d'équivalence. La procédure principale et les intervalles ont des approximations différentes ; nous rapportons les deux sans prétendre à une couverture finie exacte.

Les 93 diagnostics agrégés sont : 40 PIT de variance (10×4), 20 PIT binomiaux (10×2), 30 distributions de p-values d'histogrammes (10×3), et 3 distributions de p-values inter-lanes. Chaque distribution est comparée à l'uniforme par KS. Une famille de Bonferroni de taille 93 est appliquée. Cette correction ne nécessite pas l'indépendance des tests entre eux si chaque p-value est valide ; elle ne corrige toutefois pas une mauvaise calibration marginale. Les références de Pearson et de variance restent approximatives.

La calibration empirique séparée concerne les quatre diagnostics structurels, avec deux sources de référence. Les bornes de Clopper–Pearson à 95 % sont individuelles, pas des bornes simultanées certifiant toute la famille. La borne de taux d'alerte fixée à 3 % est un garde-fou, pas un test d'égalité exacte à 1 %. Les contrôles injectés sont volontairement forts : intervalle divisé par deux, duplication de valeurs consécutives, duplication de lanes adjacentes. Ils démontrent une sensibilité à ces défauts précis, pas à toute faiblesse d'un PRNG.

Une seed est un entier de 64 bits issu de SHA256 d'une chaîne documentée ; les 2 048 seeds fonctionnelles sont distinctes dans le fichier effectivement utilisé. Cette construction facilite le rejeu et évite leur sélection en fonction des résultats. Elle n'est pas une preuve que les sorties de deux seeds sont indépendantes. Les seeds initialisent chaque famille selon son adaptateur ; les versions NumPy passent par leur SeedSequence interne.

Le budget de mots est identique pour chaque moteur. Six fonctions réutilisent les mêmes tirages, ce qui induit des dépendances entre leurs diagnostics et justifie notamment une gestion explicite de la multiplicité. Les temps de campagne incluent les calculs de statistiques et ne sont pas des débits de génération ; seuls `timings.csv` et `timing_summary.csv` concernent le benchmark séparé.

Le contrôle de 16 positions par lane utilise les indices globaux `lane×4096+j×256`, j de 0 à 15. Le champ historique du CSV `duplicate_selected_lane_prefixes` désigne en réalité l'égalité de ces **signatures échantillonnées**, pas des préfixes contigus de 16 tirages. Le mot « lane » représente la décomposition logique fixée ; aucune mesure de montée en charge sur 2 048 threads physiques n'est réalisée ici.

Ce protocole n'explore ni des millions de flux longs, ni toutes les configurations de lanes, ni les très grandes dimensions, ni les bits individuellement sur toutes les positions. Il ne relance pas TestU01 et ne certifie pas la sécurité cryptographique. Un moteur connu pour échouer à un autre diagnostic peut donc ne produire aucune alerte ici, notamment MT19937. Les fonctions de faible dimension et les tailles choisies limitent aussi la portée de toute conclusion sur les applications réelles.
