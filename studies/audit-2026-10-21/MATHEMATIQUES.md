# Propriétés exactes du moteur récupéré

Notation : M=2^64, g=0x9E3779B97F4A7C15, a=seed, l=lane.
Toutes les sommes et tous les produits de mots sont pris modulo M.
Le code de référence est `snapshot/montmory_gpu_test/montmory_gpu_test/src/mmc_core.c`.
La sortie est `H(P_a(s))`, où P est le Feistel et H le mélange final.

## P1 — Période du compteur

Le compteur après n tirages vaut s_n=a+n(g+2l). Puisque g+2l est impair,
il est inversible modulo M. L'égalité s_n=s_0 équivaut donc à n=0 modulo M.
Le compteur a exactement la période M. Cette proposition concerne l'état ; elle
ne suffit pas, à elle seule, à établir la période minimale de la suite observée
après une fonction de sortie non injective.

## P2 — Inversibilité des tours Feistel

Un tour est (L,R) → (R,L xor F(R)), sur deux mots de 32 bits. Son inverse est
(U,V) → (V xor F(U),U), sans aucune hypothèse d'injectivité sur F.
La composition de trois tours est donc bijective, pour chaque seed et même pour
des rotations ou clés répétées. Les clés sont fixes pendant la trajectoire.

Cela démontre une permutation finie. Cela ne démontre ni imprédictibilité ni
indépendance statistique. La branche de F inspirée de Collatz ne dépend pas de la
convergence de la dynamique classique de Collatz.

## P3 — Inversion complète du mélange final et contre-exemple

Écrivons H(x)=ROTR(B(x), x>>59), où B est la composition :

```
x ^= x >> 32
x ^= x >> 25
x *= 0xA24BAED496EE9B5F
x ^= x >> 28
x *= 0x9FB21C651E98DF25
x ^= x >> 27
```

Chaque multiplicateur est impair, donc inversible modulo M. Pour un décalage h,
l'inverse de y=x xor (x>>h) est
x=y xor (y>>h) xor (y>>(2h)) xor ... jusqu'à épuisement des 64 bits.
B est donc une bijection explicitement inversible.

Pour un mot de sortie z donné, tous les antécédents de H, et seulement eux,
s'obtiennent en examinant r=0,...,31 :

```
x_r = inverse_B(ROTL(z,r))
retenir x_r si (x_r >> 59) == r
```

Il s'agit d'une description exhaustive des fibres, pas d'une recherche limitée
dans les 2^64 entrées. Chaque r impose un unique candidat par bijectivité de B.

Pour z=7, les quatre candidats retenus sont :

```
0x189da44d5bb5734c
0x63408dd79b3fd590
0x8abd642d2d15ccec
0xf3f0c11c862f9315
```

Le programme `verify_exact.py` vérifie H(x)=7 dans le modèle entier et en appelant
directement le mélange du C original compilé. Il inverse ensuite P pour le seed
42 et retrouve les indices du compteur correspondants. Les certificats sont dans
`evidence/exact.json`. Ainsi H et H∘P ne sont pas injectifs.

## P4 — Non-uniformité exacte de la mantisse sur le cycle

La mantisse zéro est produite précisément lorsque H(P(s)) appartient à
{0,...,2047}. La procédure exhaustive de P3 donne

```
sum_{z=0}^{2047} |H^{-1}({z})| = 2066.
```

Comme P et la trajectoire du compteur sont bijectifs sur leurs ensembles de M
états, ce nombre est valable pour chaque seed et chaque lane sur un cycle entier.
Une distribution exactement uniforme des 2^53 mantisses donnerait 2048 états
par mantisse. L'uniformité exacte est donc réfutée. Pour la seule mantisse zéro,
la différence de probabilité vaut 18/2^64, environ 9,76×10^-19. On ne doit pas
transformer ce contre-exemple en affirmation non mesurée sur l'erreur des
intégrales usuelles ou sur un échec futur obligatoire de BigCrush.

## P5 — Relation exacte entre lanes

Les clés et rotations dépendent du seed seulement. Pour deux lanes l et h,
soit n'=n(g+2l)(g+2h)^(-1) modulo M. Alors s_l(n)=s_h(n') et les sorties coïncident.
Chaque lane parcourt le même ensemble de compteurs dans un ordre différent.
Les lanes l et l+2^63 ont même pas ; elles sont identiques dès le début pour l'API
64 bits, même si cet alias dépasse les identifiants de thread 32 bits du kernel.
Les lanes 0 et 1 coïncident à l'indice n=2^63.

Pour seed 42, le premier tirage de lane 0 égale celui de lane 1 à
n'=15941120965150899. Cet exemple lointain n'est pas une preuve de corrélation
pratiquement détectable sur les préfixes testés. Il réfute seulement une garantie
globale de non-recouvrement dérivée de la simple distinction des pas.

## P6 — Formule globale et indépendance de l'ordonnancement physique

Le wrapper incrémente base de T*D après chaque remplissage. Pour un bloc b, une
lane l et un indice local i (0≤i<D), la position k=b*T*D+l*D+i vérifie :

```
lane(k) = floor(k/D) mod T
s(k) = seed + (k+1)*(g+2*lane(k)) mod M
```

La sortie dépend uniquement de seed,T,D,k. Aucune sortie précédente n'est requise.
Répartir les mêmes k entre workers dans n'importe quel ordre conserve les sorties.
Le contrôle OpenMP vérifie cette propriété en exécution, après sa dérivation.

Changer T ou D modifie en revanche la fonction lane(k). Pour seed=42 et k=2 :
la mantisse est 8314571741591251 pour T=2,D=4, et 4905313308424108 pour T=4,D=2.
Ce n'est pas une divergence d'arithmétique entre architectures. C'est une
différence de définition du flux ; les paramètres logiques doivent être fixés.

## P7 — Conversion en double

Le mot mantisse=word>>11 est dans [0,2^53). Sa conversion vers un binary64 IEEE
est exacte, puis la multiplication par 2^-53 est exacte également. La division
par 2^53 du C et la multiplication du wrapper représentent donc la même valeur
sur les architectures concernées sous ce contrat. Le kernel Metal calcule des
entiers 53 bits ; la conversion en double est effectuée côté CPU.

Le raisonnement porte sur les uniformes ainsi définis. Des transformations
ultérieures faisant intervenir log, sin, des réductions flottantes ou des
échantillonnages avec rejet ont leurs propres exigences de reproductibilité.

## Portée des preuves et contrôles

Les preuves P1/P2/P5/P6/P7 sont algébriques et conditionnées aux sémantiques de
mots non signés et de binary64 indiquées. P3/P4 disposent d'énumérations exactes
exhaustives de petites fibres et de certificats vérifiés par le C original.
Les comparaisons finies entre implémentations ne remplacent pas la validation des
compilateurs et du matériel. Aucune sécurité cryptographique ni originalité
mathématique n'est déduite de ces propriétés.
