# Pourquoi ces variantes calculent le même flux

Le contrat conservé est celui du bloc TestU01 actuel : T=2048, D=4096, N=T·D, base=0. Les contrôles explorent aussi d'autres géométries et des bases près du débordement. Chaque mot est défini par sa position i dans le bloc, avec lane ℓ=floor(i/D) et compteur

`c(i) = seed + (base+i+1)·γℓ (mod 2^64)`, où `γℓ=0x9E3779B97F4A7C15+2ℓ`.

La sortie reste le même mélange de la même permutation à trois tours. Le format mantissa53 reste `raw64 >> 11`. La fonction nommée `pcg_dxsm` dans le code récupéré est celle de Montmory ; ce nom ne l'identifie pas au générateur PCG64DXSM standard.

## Compteurs incrémentaux

Dans une même lane, pour un pas s, `c(i+s)=c(i)+s·γℓ (mod 2^64)`. Remplacer une multiplication répétée par cette addition est donc une identité, y compris lors du débordement non signé. Les variantes CPU coupent leurs intervalles aux frontières des lanes et réinitialisent le pas lorsque ℓ change.

## Deux mots de 32 bits pour un compteur

Pour a=a₀+2³²a₁ et b=b₀+2³²b₁, le mot bas de a+b est `lo=(a₀+b₀) mod 2^32`. Le mot haut est `a₁+b₁+[lo<a₀] mod 2^32`. C'est exactement l'addition modulo 2^64. `opt_add` applique cette formule ; le débordement haut est volontaire.

## Rotations spécialisées

Pour chaque x de 32 bits, le facteur de la fonction de tour vaut 3 si x est impair, 1 sinon. Le produit est reconstruit avec son mot bas et `mulhi`, puis la clé est ajoutée avec sa retenue. La rotation choisie vaut ro ou re suivant la même parité. Déplacer ce choix après les deux expressions de rotation conserve donc la valeur sélectionnée. Cela permet au compilateur Metal de connaître séparément les deux rotations après spécialisation de la seed ; un gain de vitesse n'en découle pas mathématiquement.

Les rotations produites par l'initialisation sont impaires entre 1 et 63. Leur résidu modulo 32 est donc entre 1 et 31 : les décalages de `mmc_rot_low32` ne valent jamais 32. Les variantes ne réduisent ni le nombre de tours ni le mélange final.

## Attribution des positions aux threads

Soit W le nombre de threads associés à une lane et B le nombre de mots calculés ensemble. Dans la disposition contiguë, le thread de rang t écrit les positions `r=(mW+t)B+b`, avec 0≤t<W, 0≤b<B et m≥0. La division euclidienne de r par B puis de floor(r/B) par W donne une décomposition unique.

Dans la disposition entrelacée, il écrit `r=t+W(mB+b)`. La division euclidienne de r par W puis de floor(r/W) par B donne également une décomposition unique. La condition r<D élimine les positions hors de la lane. Pour toute position valide, la base de sa boucle est inférieure ou égale à r et reste donc inférieure à D : elle est effectivement parcourue.

Ainsi chaque mot valide a exactement un écrivain dans ces deux dispositions. Les écritures de threads voisins sont consécutives à b fixé dans la seconde disposition. Aucun résultat d'un autre thread n'est lu. Le chemin général utilise `i=tid+kP`, où P est le nombre de threads ; la division euclidienne par P assure la même propriété.

## Portée des vérifications

Ces identités justifient les transformations algébriques et l'attribution des indices sous les préconditions du benchmark : tailles contrôlées, T et D puissances de deux, compteurs non signés, rotations issues de l'initialisation. Elles ne certifient pas le compilateur ni tous les détails de son exécution native.

Le vérificateur Linux exécute une transcription C++ des kernels avec UBSan, compare chaque sortie à la référence C originale et contrôle les gardes mémoire. Le kit Mac vérifie ensuite les sorties natives et les vecteurs officiels Random123. Les spécialisations des rotations sont des variables dans la transcription CPU ; leur compilation Metal reste à vérifier sur le Mac.

Une égalité bit à bit pour toute la séquence soumise au même adaptateur conserve les données des tests statistiques. Les comparaisons finies fournies contrôlent l'implémentation sur leurs cas ; elles ne constituent pas une preuve universelle d'absence de bug, ni une nouvelle exécution de SmallCrush, Crush ou BigCrush.
