# Diagnostics TACM complémentaires : état corrigé

[Français](tacm.md) · [English](en/tacm.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Protocole figé](../studies/tacm-comparison-2026-10-21/protocol.json) · [Méthodes détaillées](../studies/tacm-comparison-2026-10-21/METHODES.md)

Le nom TACM désigne ici le programme de diagnostics complémentaires du projet. La campagne archivée est **un sous-ensemble corrigé et explicitement défini**, pas la validation générale de toutes les idées historiques portant ce nom.

L'objectif est d'examiner le comportement d'estimateurs Monte Carlo et de certaines relations entre sorties, en complément des batteries TestU01. Réussir une batterie générale n'établit pas à lui seul la précision d'une fonction applicative particulière.

## Plan expérimental

| Élément | Valeur |
|---|---|
| Moteurs | Montmory, Philox4x32-10, Threefry2x32-20, Threefry4x64-20, xoshiro256**, NumPy PCG64, PCG64DXSM, MT19937, SFC64 et Philox4x64 |
| Répétitions | 2 048 seeds par moteur, organisées en 128 groupes de 16 |
| Sorties | 65 536 mots par seed ; conversion commune `(raw64 >> 11) × 2^-53` |
| Fonctions | `U`, `U²`, sinus, événement `U>0,99`, produit de deux uniformes, événement rare conjoint |
| Paires | Paires disjointes pour les fonctions à deux coordonnées |
| Comparaisons principales | 9 concurrents × 6 fonctions = 54 |
| Règle d'avantage | Réduction de MSE supérieure à 10 %, p-value corrigée Bonferroni < 0,05, et même direction d'effet dans les deux moitiés prédéfinies des groupes |

La campagne produit **1 342 177 280 mots fonctionnels** et **122 880 estimations**. Les seeds sont déterministes et distinctes selon les espaces de noms du protocole ; cela ne démontre pas leur indépendance mathématique.

À partir de l'indice zéro, 65 536 sorties ne couvrent que 16 des 2 048 lanes logiques Montmory. Un contrôle distinct porte sur 64 seeds, 2 048 lanes et 16 positions par lane, pour trois moteurs. Il ne faut pas présenter la première expérience comme une exploration exhaustive de toutes les lanes.

## Calibration : conserver aussi le pilote échoué

La calibration initiale comprend 512 essais par source de référence. Son critère global n'est pas atteint. Cet échec est conservé dans les données.

Un [addendum fixé avant la comparaison fonctionnelle](../studies/tacm-comparison-2026-10-21/calibration_addendum.json) prévoit une extension unique à 4 096 essais par source, sans changer les seuils. Cette extension satisfait le critère : les huit taux d'alerte observés vont approximativement de 0,781 % à 1,196 % pour un seuil nominal de 1 %. Les trois défauts fortement injectés sont détectés dans les 4 096 essais de chacune des deux sources.

Ces contrôles utilisent deux PRNG de référence. Ils ne constituent pas une source physique IID parfaite, ni une garantie de puissance contre tous les défauts faibles. Plusieurs lois et tests de second niveau restent approximatifs ; ils sont identifiés dans les méthodes.

## Résultat

**Aucune des 54 comparaisons n'établit l'avantage prévu pour Montmory.** Aucune des 93 comparaisons diagnostiques corrigées ne déclenche d'alerte. Cela ne démontre ni une équivalence statistique de tous les moteurs, ni l'absence de différences sur d'autres fonctions ou tailles.

Les MSE Montmory rapportées à la variance nominale IID sont environ 0,978 / 0,987 / 0,995 / 1,016 / 0,966 / 1,003 selon les six fonctions. Des nombres légèrement inférieurs à 1 ne suffisent pas à annoncer un meilleur PRNG : ce sont des estimations soumises à fluctuation.

Une vérification indépendante de 180 estimations avec `math.fsum` et de moments polynomiaux exacts est [archivée](../studies/tacm-comparison-2026-10-21/results/independent_replay.json). Le [résumé machine](../studies/tacm-comparison-2026-10-21/results/summary.json), les CSV et les scripts de recalcul sont inclus.

## Corrections par rapport aux diagnostics historiques

Les anciennes interprétations reposant sur une borne universelle de variance, un `bridge_gap` nul par construction, ou des paires recouvrantes traitées comme indépendantes ne sont pas réintroduites dans cette comparaison. Les [travaux d'audit](../studies/audit-2026-10-21/README.md) documentent le contexte.

« Améliorer la calibration » signifie améliorer la fiabilité des seuils du diagnostic. Cela ne prouve pas que Montmory a un défaut, et ne signifie pas que ses p-values doivent être rapprochées de 0,5. Aucune modification du moteur n'est justifiée par cette seule formulation.
