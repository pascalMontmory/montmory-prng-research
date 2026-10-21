# Identités et conditions

Pour un entier de 64 bits, écrire x = lo + 2^32 hi. Tous les mots lo et hi sont non signés et calculés modulo 2^32.

## Opérations à deux moitiés

Addition : le nouveau lo vaut `(a.lo+b.lo) mod 2^32`; le nouveau hi vaut `a.hi+b.hi+carry mod 2^32`, où `carry = (lo<a.lo)`. Cette identité couvre aussi le débordement modulo 2^64.

Multiplication par c = c_lo + 2^32 c_hi :

```
lo' = lo*c_lo mod 2^32
hi' = high32(lo*c_lo) + hi*c_lo + lo*c_hi mod 2^32
```

Le terme hi*c_hi est multiplié par 2^64 et disparaît. Cette décomposition s'applique aux mêmes deux constantes du mélange historique. `high_product` utilise le produit élargi sur ARM64, deux produits de positions paires sur x86, ou `metal::mulhi` dans Metal.

Pour 0<s<32, le décalage `x >> s` a pour moitié basse `(lo>>s)|(hi<<(32-s))` et pour moitié haute `hi>>s`. Les XOR par décalage du mélange sont donc inchangés.

Les rotations Feistel issues de l'initialisation sont impaires dans [1,63]. Leur décalage modulo 32 est dans [1,31]. Une rotation de plus de 32 échange les rôles de lo/hi, puis applique les mêmes deux décalages. Le code ne décale aucun mot de 32 bits de 32 positions.

La rotation finale vaut `hi >> 27`, donc appartient à [0,31]. Pour zéro, les termes croisés sont masqués ; pour 1..31, ils reconstruisent exactement la rotation droite de 64 bits. Les deux moitiés sont alors assemblées avant le décalage de sortie éventuel de 11 bits.

## SIMD

Les opérations de `wide_math.inc` sont appliquées composante par composante à quatre entrées. Aucune réduction, permutation ou communication entre composantes n'intervient. `wide_round` conserve le même échange des moitiés et le même XOR que chaque tour Feistel historique. Par composition de ces identités, chaque composante produit le même mot que le calcul scalaire sous les conditions de rotation ci-dessus.

Le fichier arithmétique est partagé par les backends CPU et Metal. Les opérations de comparaison sont normalisées en masques tout-zéro/tout-un dans chaque backend. Les contrôles natifs restent nécessaires pour vérifier ces implémentations et leur compilation.

## Compteurs, lanes et workers

Pour une lane logique fixée de pas gamma, `c(k+s)=c(k)+s*gamma mod 2^64`. Le code CPU arrête ses boucles à chaque frontière de 4096 mots, puis recalcule gamma avec l'indice global. Les opérations SIMD ne franchissent donc pas une frontière de lane sans traitement explicite de la fin du segment.

À indice global fixé k, le mot dépend seulement de k et de la seed. Découper une plage en intervalles disjoints, puis écrire chaque mot à son indice, conserve la suite. Le pool utilise quotient et reste pour partager chaque plage sans trou ni chevauchement ; aucune graine n'est choisie par un worker.

Pour Metal, à l'intérieur d'une lane et pour W threads par lane, les positions contiguës sont `j=4*(m*W+t)+b`, avec 0<=t<W et 0<=b<4. La division euclidienne de j par 4 puis par W donne une attribution unique. Les positions entrelacées sont `j=t+W*(4*m+b)` ; la division par W puis par 4 donne la même propriété. Le contrôle j<D supprime uniquement les positions hors plage. Le fallback d'un nombre de threads irrégulier utilise `i=tid+m*P`, également unique.

Les blocs historiques ont N=2048*4096 et leurs bases sont des multiples de N. Dans ce cas, `floor((base+i)/4096) mod 2048 = floor(i/4096)` pour 0<=i<N : le contrat global CPU reproduit exactement les rechargements de l'adaptateur TestU01 historique. Il ne faut pas appliquer cette égalité à une base arbitraire sans tenir compte de la lane globale.

## Limites

Ces identités conservent le générateur ; elles ne démontrent ni l'indépendance entre lanes, ni une sécurité cryptographique, ni une amélioration statistique. Le succès des comparaisons finies n'est pas une démonstration générale de correction du compilateur ou du matériel. Les anciens résultats TestU01 gardent leur périmètre : version, seed, ordre et adaptateur de sortie.

Référence des primitives ARM : [Arm NEON Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/). Les noms historiques des fonctions du code récupéré, dont `pcg_dxsm`, sont conservés ; ce mélange personnalisé ne doit pas être assimilé au générateur standard PCG64DXSM.
