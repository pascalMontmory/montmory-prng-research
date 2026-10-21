# Contrat du flux et propriétés exactes

[Français](contract.md) · [English](en/contract.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Référence C](../src/reference/mmc_core.c) · [Modèle entier Python](../src/reference/reference.py)

## Définition

Le moteur documenté est le **Montmory 3R récupéré**, employé par les variantes Metal v3 et les portages TPU. « v3 » peut désigner un kit Metal ; cela ne signifie pas qu'un nouvel algorithme a été créé à chaque optimisation.

Pour la sortie d'indice global `k ≥ 0`, avec `T > 0` lanes logiques et `D > 0` sorties consécutives par lane :

```text
lane(k)  = floor(k / D) mod T
gamma(l) = (0x9E3779B97F4A7C15 + 2*l) mod 2^64
state(k) = (seed + (k+1)*gamma(lane(k))) mod 2^64
word(k)  = mixer(feistel_3R(state(k), seed))
mant53   = word(k) >> 11
u01      = mant53 * 2^-53
```

L'état est avancé **avant** la sortie. Le `n` du modèle par lane commence donc à 1 ; le `k` de l'interface globale commence à 0. Les paramètres et opérations exacts sont donnés dans les deux sources ci-dessus et dans l'[audit mathématique détaillé](../studies/audit-2026-10-21/MATHEMATIQUES.md).

La géométrie principale du travail récent est `T=2048, D=4096`. Le benchmark TPU final commence à `k=2^32−4097`, afin de traverser une frontière du compteur. Les campagnes statistiques ne doivent pas être confondues avec cette fenêtre de mesure.

Une sortie doit être identifiée par `(version, seed, k, T, D, format)`. Modifier `T` ou `D` peut modifier le flux. Modifier la répartition physique du travail peut préserver le flux si chaque sortie conserve le même indice global et cette même formule.

## Trois notions différentes de parallélisme

| Terme | Signification dans ce travail | Modifie nécessairement le contrat ? |
|---|---|---|
| Lane logique | Identifiant participant au calcul de `gamma` | Oui si sa correspondance avec `k` change |
| Worker CPU / thread Metal / groupe | Organisation du calcul et des écritures | Non si les indices restent identiques |
| Tuile Pallas | Bloc de sorties traité ensemble | Non si les indices restent identiques |

`pallas_carry_64` est bien **une implémentation Montmory sur TPU**. `64` désigne 64 lignes de 128 sorties, soit 8 192 mots par tuile, et non 64 cœurs. `carry` utilise une identité de retenue pour calculer exactement le même résultat entier. Les trois tours et les constantes du moteur sont inchangés. Les [sources des noyaux](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/kernels.py) restent celles de la campagne.

## Ce qui est exact, et ce qui ne l'est pas

Pour une lane fixée, l'incrément est impair : le **compteur d'état** a une période exacte de `2^64`. Le Feistel à trois tours est une permutation des 64 bits. Cela ne démontre ni une sécurité cryptographique, ni la période minimale des sorties après leur mélange final.

Le mélangeur final applique une rotation dont le sélecteur dépend de son entrée. Il **n'est pas injectif**. Ces quatre entrées distinctes ont toutes la sortie entière `7` :

```text
189da44d5bb5734c  63408dd79b3fd590
8abd642d2d15ccec  f3f0c11c862f9315
```

Le certificat calcule exhaustivement les candidats des 32 rotations possibles, puis vérifie chacun. La première case de mantisse, `mant53=0`, possède **2 066 préimages**, au lieu de 2 048 pour une répartition exactement uniforme sur un cycle d'états de `2^64`. L'excès de masse pour cette seule case est `18/2^64`. Ce constat exact ne mesure pas à lui seul une erreur pratique de Monte Carlo.

Les lanes de même seed parcourent le même espace d'états selon des ordres différents. Leur indépendance ou l'absence de recouvrement ne sont pas garanties par construction. Les lanes 0 et 1 atteignent notamment le même état au pas `2^63`. Les [certificats](../studies/audit-2026-10-21/evidence/exact.json) et `scripts/check_reference.py` rendent ces limites vérifiables.

Le nom historique `pcg_dxsm` dans le code désigne un mélangeur spécifique à ce projet ; il **ne faut pas l'assimiler au PCG64DXSM standard**.

## Portée du « bit à bit »

Le contrat porte sur les mots entiers `raw64` et les mantisses entières `mantissa53`. La conversion prescrite en binary64 est exacte. Les transformations applicatives ultérieures (`log`, `sin`, normales, sommes parallèles, valorisations) ne deviennent pas automatiquement identiques entre architectures.

Les vecteurs historiques, contrôles de frontières et empreintes de sorties chronométrées fournissent une conformité expérimentale substantielle. Ils ne couvrent pas toutes les seeds et tous les indices. Aucun nouveau passage statistique n'est déduit d'une simple réussite de conformité.
