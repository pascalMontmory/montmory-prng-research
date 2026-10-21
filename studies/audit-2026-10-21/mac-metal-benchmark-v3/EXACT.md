# Identités des variantes Montmory des kits v2 et v3

Ces réécritures portent sur l'implémentation. Le générateur de référence reste
celui de `src/mmc_core.c` et du kernel original `src/mmc_kernel.metal`, conservés
à l'identique. Tous les calculs ci-dessous sont non signés ; une opération sur
32 ou 64 bits est réduite modulo la puissance de deux correspondante.

## Décomposition exacte du Feistel

Posons B = 2^32. Les trois tours ne transmettent à `mmc_f` que des demi-mots
x compris entre 0 et B−1. Pour x pair, t = x. Pour x impair, t = 3x. Dans ce
dernier cas, l'addition x + 2x donne exactement :

    t_lo = (x + ((2x) mod B)) mod B
    t_hi = floor(x / 2^31) + [t_lo < x].

Le premier terme de t_hi est la retenue du décalage 2x ; le second est celle
de l'addition. Le masque `0 - (x & 1)` sélectionne ce calcul pour x impair et
le couple (x, 0) pour x pair. L'ajout de la clé k = k_lo + B k_hi donne :

    lo = (t_lo + k_lo) mod B
    hi = (t_hi + k_hi + [lo < t_lo]) mod B.

On obtient ainsi exactement t + k modulo 2^64. Les rotations r proviennent
de l'initialisation historique : elles sont toujours impaires dans [1, 63].
Avec s = r mod 32, on a donc 1 ≤ s ≤ 31. Pour r < 32, les 32 bits faibles de
la rotation gauche sont `(lo << s) | (hi >> (32-s))`. Pour r ≥ 32, les moitiés
sont échangées dans cette formule. Aucun décalage de 32 bits n'est effectué.

Le XOR final de `mmc_f` est tronqué à 32 bits dans la référence. Il suffit
donc de calculer x >> 1 pour la branche impaire et 5x modulo B pour la branche
paire. Cette décomposition prouve l'égalité de chaque fonction de tour.
Par substitution dans les trois tours inchangés, le Feistel complet est égal.

## Décomposition exacte du mélange 64 bits

Pour x = a + B b et c = d + B e, la multiplication modulo 2^64 donne :

    (x c)_lo = (a d) mod B
    (x c)_hi = (floor(a d / B) + b d + a e) mod B.

`metal::mulhi(a,d)` fournit la partie haute du produit non signé 32 × 32.
Le terme B² b e s'annule modulo 2^64. Cela justifie `mmc_mul32` pour les deux
constantes historiques, sans perte de bits.

Pour chacun des décalages s = 25, 28, 27, le couple de x XOR (x >> s) est :

    lo' = lo XOR ((lo >> s) | (hi << (32-s)))
    hi' = hi XOR (hi >> s).

Le début du mélange `(x >> 32) XOR x` devient `(lo XOR hi, hi)`.
La rotation finale r = x >> 59 est dans [0, 31]. Pour r > 0, la rotation
droite recombine les deux moitiés par les décalages r et 32−r. Pour r = 0,
les termes croisés sont masqués à zéro et les deux moitiés sont inchangées.
Le code borne chaque décalage à [0, 31], y compris dans ce cas limite.
Toutes les étapes sont donc identiques au mélange original, avec ses mêmes
propriétés et limites, y compris sa non-injectivité déjà documentée dans l'audit.

## Compteurs et affectation des threads

Pour un bloc de N = L D sorties, L lanes logiques, une seed S et une base b,
la sortie d'indice i = l D + j dépend du compteur :

    gamma_l = 0x9E3779B97F4A7C15 + 2l
    c(i) = S + (b + i + 1) gamma_l     modulo 2^64.

Le mode indexé recalcule c(i) pour chaque indice. Le mode par sous-séquences
affecte W threads à chaque lane, où W est une puissance de deux. Le thread
(l, u), 0 ≤ u < W, écrit les positions :

    i(t) = l D + u + t W, tant que u + t W < D.

Chaque position j possède une unique décomposition j = u + t W par division
euclidienne. Il n'y a donc ni omission ni écriture concurrente d'une position.
Si W > D, les threads u ≥ D ne font aucune écriture. Le compteur vérifie :

    c(i(t+1)) = c(i(t)) + W gamma_l    modulo 2^64.

Une multiplication initiale et des additions suffisent donc. Le débordement
non signé conserve l'identité, même lorsque la base est proche de 2^64−1.
Les nombres de threads qui ne se décomposent pas en L × W avec W puissance de
deux utilisent le mode indexé de repli, couvert par les mêmes vérifications.
Le contrat du kit impose L et D puissances de deux et N dans [2^16, 2^25]
pour les mesures ; les contrôles couvrent aussi des blocs plus petits.

## Portée de la vérification

Ces identités établissent l'égalité dans le modèle d'arithmétique non signée.
Les essais avec UBSan confrontent en plus le code généré à des références,
sur des cas limites et de nombreuses valeurs. Ils ne remplacent pas la
compilation Apple ni son exécution sur GPU. Le kit interrompt les mesures
si ses contrôles natifs d'identité échouent.

À L, D, seed, base et adaptation de sortie identiques, changer le nombre de
threads physiques ne change pas le flux. Modifier L ou D définit en général
un autre flux : le nouveau découpage physique ne supprime pas cette contrainte.
Pour reproduire les p-values historiques, il faut également conserver
l'adaptation TestU01, la progression entre blocs et les paramètres des batteries.

## v3 : produit haut et deux expressions de la fonction de tour

Dans `mmc_f_mulhi`, a = 1 + 2(x mod 2) vaut 1 pour x pair et 3 pour x impair.
Le produit a x est inférieur à 3B. Sa partie basse est `(a*x) mod B` et sa
partie haute est exactement `mulhi(x,a)`. L'ajout de la clé utilise la même
retenue que la démonstration précédente. La rotation et la correction finales
sont celles de la branche historique correspondante.

Dans `mmc_f_dual`, les deux expressions pour x pair et impair sont calculées
séparément, avec les retenues exactes, puis la parité sélectionne le résultat.
Cela ne suppose aucune propriété statistique de la parité.

La variante `spec_pair` conserve directement la fonction de tour 64 bits
historique et décompose le mélange final sur 32 bits. Les trois rotations sont
fournies au compilateur Metal comme constantes de fonction. `spec_dual` combine
cette spécialisation avec l'expression précédente des deux branches.
La factory CPU dérive toujours ces constantes de `mmc_init(seed,0)` et cache
les pipelines par triplet ordonné (r0,r1,r2), moteur et mode. Les clés, la seed,
la base et la géométrie restent des paramètres d'exécution. Une seed différente
ayant les mêmes rotations peut réutiliser le pipeline avec ses propres clés.
L'identité nécessite que le triplet du pipeline corresponde à celui des
paramètres, ce que la factory garantit. La compilation native reste à vérifier.

## v3 : plusieurs sorties adjacentes par thread

Soit Bc = 2 ou 4 le nombre de sorties par lot (distinct de B = 2^32).
Le thread (l,u) de la lane l écrit désormais :

    j = Bc (u + t W) + b,  0 ≤ b < Bc, avec j < D.

Pour chaque j, la division par Bc donne un unique couple (q,b). La division
de q par W donne un unique couple (t,u). Les positions sont donc couvertes
exactement une fois, sans conflit. Si D est plus petit que Bc ou W Bc, les
positions et threads inactifs ne sont pas écrits. Les calculs d'un lot sont
indépendants ; entre les lots, le compteur progresse de W Bc gamma_l.
À l'intérieur d'un lot, son élément b ajoute b gamma_l. Toutes les valeurs
réalisent ainsi le même c(i) que la référence.

Les indices de boucle de ces candidats sont limités à 32 bits. Le contrat
validé impose N ≤ 2^25 et threads physiques ≤ 2^20 : même les incréments
intermédiaires de boucle restent inférieurs à 2^32. Les compteurs PRNG, la
seed et la base restent sur 64 bits et conservent leur débordement modulo 2^64.

## v3 : diagnostic avec moins d'écritures

Le mode de réduction produit, pour chaque thread, le XOR de toutes ses sorties,
ou zéro si ce thread n'a aucune sortie. L'identité du XOR global suit de la
couverture unique démontrée ci-dessus et de l'associativité du XOR. La référence
CPU vérifie en plus chaque XOR individuel avec une formule indépendante
d'affectation : `owner = l*W + (floor(j/Bc) mod W)`.

Pour Philox, les deux mots de chaque bloc sont consommés, et son propriétaire
est `(floor(i/2) mod threads)`. Pour Threefry et les chemins indexés de repli,
le propriétaire est `i mod threads`.

Ce mode n'est pas une interface de génération du flux complet. Il conserve
un contrôle sur tous les calculs mais introduit une accumulation XOR et modifie
les ressources utilisées. Une différence de temps n'est pas une mesure exacte
du coût des seules écritures. Les sommes et XOR du benchmark ne remplacent
pas les contrôles mot par mot ni les identités arithmétiques.
