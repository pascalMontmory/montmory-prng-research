# Références et antériorités

[Français](references.md) · [English](en/references.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md)

Les travaux Montmory sont évalués dans le contexte des PRNG existants. L'accès déterministe par compteur et la portabilité des sorties ne sont pas revendiqués comme des nouveautés générales.

| Source primaire | Utilisation dans le dossier |
|---|---|
| [Salmon et al., Parallel Random Numbers: As Easy as 1, 2, 3 (2011)](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf) | Philox, Threefry et génération parallèle à compteur |
| [Random123, commit 9545ff6](https://github.com/DEShawResearch/random123/tree/9545ff6413f258be2f04c1d319d99aaef7521150) | Sources C/C++ des comparateurs ; vecteurs connus et licence conservés |
| [TestU01, documentation des auteurs](https://www-labs.iro.umontreal.ca/~simul/testu01/guideshorttestu01.pdf) | Batteries statistiques et paramètres ; version archivée 1.2.3 |
| [JAX : benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) | Séparation compilation/exécution et synchronisation des mesures |
| [JAX : Pallas TPU PRNG](https://docs.jax.dev/en/latest/pallas/tpu/prng.html) | Génération pseudo-aléatoire sous Pallas et portée des options TPU |
| [JAX : configuration des PRNG](https://docs.jax.dev/en/latest/jax.random.html#advanced-rng-configuration) | Noms d'implémentation, choix d'API et limites de reproductibilité |
| [JAX : détails TPU](https://docs.jax.dev/en/latest/pallas/tpu/details.html) | Organisation des tuiles et mémoire ; distinction avec les lanes logiques du PRNG |
| [NumPy : PCG64DXSM](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html) | Distinguer le moteur standard du nom historique spécifique utilisé par Montmory |
| [Blackman et Vigna : xoshiro256**](https://prng.di.unimi.it/xoshiro256starstar.c) | Comparateur TACM ; attribution et notice présentes dans la source |

Les liens documentaires JAX peuvent évoluer. Les résultats du dépôt restent rattachés à **JAX/jaxlib 0.10.2** et aux sources figées, pas à une promesse portant sur toute version future.

Voir aussi l'[analyse d'antériorité archivée](../studies/audit-2026-10-21/ANTERIORITE.md), les [références identifiées par commit](../studies/audit-2026-10-21/evidence/references.json) et les [sources TACM](../studies/tacm-comparison-2026-10-21/SOURCES.md). Aucun endossement par Google, Apple, NVIDIA, D. E. Shaw Research ou les auteurs de TestU01 n'est impliqué.
