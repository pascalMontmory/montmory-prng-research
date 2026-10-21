# Méthodologie de validation

[Français](methodology.md) · [English](en/methodology.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Provenance](evidence.md) · [Reproduction](reproduce.md)

## Questions séparées

| Question | Contrôle | Ce qu'une réussite ne démontre pas |
|---|---|---|
| Le programme implémente-t-il le flux défini ? | Référence C, modèle entier, vecteurs, frontières, comparaisons CPU/GPU/TPU | Qualité statistique ou vitesse |
| Le flux présente-t-il les défauts recherchés ? | TestU01 et diagnostics TACM corrigés | Absence de tout défaut ou sécurité cryptographique |
| Une réalisation est-elle plus rapide ? | Mesures répétées avec sorties contrôlées, périmètre chronométré explicite | Supériorité sur d'autres matériels ou dans une application complète |

Les optimisations étudiées conservent le moteur. Aucun ajout d'une chaîne de Markov n'a été adopté dans les implémentations confirmées.

## Conformité avant et pendant les mesures

Les sources de référence et les kits sont figés par SHA256. Les contrôles couvrent des seeds ordinaires et extrêmes, les passages de retenue, le découpage du travail et les vecteurs historiques. Les sorties de chaque appel chronométré sont contrôlées **hors de la fenêtre de mesure**.

Pour la dernière campagne TPU, les 3 420 appels des moteurs à référence CPU fixe utilisent 240 empreintes préétablies. Les 40 empreintes Montmory ont ensuite été recalculées depuis une compilation neuve du C figé. Les 288 appels RBG vérifient seulement une répétabilité sur le même backend. Ces deux catégories ne sont pas fusionnées en une garantie multiarchitecture.

Les tableaux produits ne sont pas intégralement conservés : les empreintes et les rapports le sont. Le décompte de mots avec répétitions n'est pas un nombre d'entrées distinctes. La campagne TPU compte 10 406 854 656 mots Montmory contrôlés avec répétitions, mais 167 772 160 couples seed/indice distincts.

## Recherche et confirmation TPU

Le [programme de campagne figé](../studies/tpu-opt-2026-10-21/montmory-tpu-opt/campaign.py) définit quatre volumes, 26 configurations et le calendrier pseudo-aléatoire des mesures.

1. **Recherche** : trois seeds, trois répétitions, quatre volumes, 26 configurations ; 936 observations.
2. **Sélection figée** : deux candidats Montmory et un comparateur Threefry optimisé sont sélectionnés sur la recherche. Le JSON de sélection est haché avant la confirmation.
3. **Confirmation** : sept nouvelles seeds, trois blocs de trois répétitions, onze finalistes par volume ; 2 772 observations.
4. **Adoption** : le candidat principal remplace l'ancien Pallas `32×128` seulement si son gain apparié atteint au moins 3 % et s'il est plus rapide sur les sept seeds. Sinon la référence est conservée.

Un secondaire plus rapide dans la confirmation n'est pas promu après coup. À 64 Mio, c'est précisément ce qui évite de sélectionner opportunément `pallas_carry_64` après avoir observé ses résultats.

Les 104 compilations précèdent les mesures correspondantes. Les appels attendent la complétion du résultat. L'audit vérifie les effectifs, l'absence de doublons, l'ordre prévu, la séparation des seeds, la sélection, les décisions et les 44 synthèses. Les sources officielles JAX expliquent pourquoi compilation et exécution asynchrone doivent être distinguées dans un [benchmark](https://docs.jax.dev/en/latest/benchmarking.html).

## Périmètre du chronométrage

| Campagne | Temps mesuré | Exclusions ou particularités |
|---|---|---|
| CPU ARM64 / workers | Temps vu par l'appelant | Le test `pair_reused_workers` exclut la construction initiale du pool ; les variantes `spawn` incluent création/jonction |
| Metal | Horodatage GPU du travail soumis | Ne mesure pas le coût complet d'une application CPU→GPU→CPU |
| TPU JAX/Pallas | Appel depuis l'hôte, allocation et attente de complétion | Compilation, préparation des clés, transfert des résultats et vérification exclus |
| TestU01 | Temps CPU rapporté par la batterie | Ce n'est pas un benchmark du noyau GPU seul |

Il ne faut pas comparer directement « Go/s CPU », « Go/s Metal » et « Go/s TPU » comme s'il s'agissait de mesures ayant la même frontière. Le témoin d'écriture n'est ni un PRNG ni un plafond matériel universel.

## Statistiques, calibration et comparaisons multiples

Les diagnostics complémentaires évaluent des fonctions explicites avec des moments de référence connus. Ils emploient des paires disjointes lorsque la loi supposée l'exige, une conversion uniforme commune et des familles de comparaisons annoncées à l'avance. Le détail des hypothèses, approximations, corrections de multiplicité et du pilote de calibration échoué est dans [TACM](tacm.md).

Une p-value isolée proche de 0,01 ou 0,99 n'est pas un score de qualité. Chercher à ramener toutes les p-values vers 0,5 serait une mauvaise cible. Les conclusions reposent sur les règles propres aux protocoles et leur portée.

## Incertitude et reproductibilité

Les médianes par seed, dispersions et comparaisons par bloc sont conservées. Les sept seeds d'une même session ne remplacent pas sept sessions matérielles indépendantes. Les résultats TPU actuels demandent encore une répétition sur une nouvelle session et sur d'autres modèles avant une extrapolation.

La CI de ce dépôt vérifie l'intégrité, des certificats exacts et la cohérence des données sur CPU. Elle ne relance pas SmallCrush/Crush/BigCrush, ne possède pas de TPU et n'atteste pas de nouvelles performances natives.
