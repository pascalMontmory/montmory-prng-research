# Analyse indépendante — campagne TPU du 21 octobre 2026 à 18:51:02 UTC

La campagne native est réussie. Montmory conserve son flux de référence et
progresse sur les grands volumes : +15,42 % à 64 Mio et +23,25 % à 128 Mio face
au précédent noyau Pallas 32x128. Threefry2x32 optimisé avec Pallas reste environ
4 % devant Montmory sur ces volumes. Ces résultats concernent un TPU v5 lite,
un périphérique, JAX/jaxlib 0.10.2 et une session Colab.

## Vérification des données

Le script `../../audit_results.py` lit les JSON/CSV reçus sans exécuter les
scripts ou bibliothèques binaires reçus. Il contrôle le manifeste local figé,
les 104 compilations, l'intégralité des 3 708 mesures, les groupes attendus,
l'absence de doublons, l'ordre aléatoire prévu, les seeds disjointes et les
44 lignes de synthèse. Il recalcule la sélection sur les données de recherche,
vérifie le SHA256 de la sélection figée et reproduit la règle d'adoption.

Les 3 420 empreintes chronométrées à référence CPU fixe correspondent aux
240 empreintes préétablies. Les 288 mesures RBG sont traitées séparément et
leurs empreintes restent identiques pour chaque seed/volume entre les blocs.
Les 40 empreintes Montmory ont été recalculées sur ce VPS à partir d'une
compilation neuve du code C local figé : toutes correspondent.

- Conformité historique native : 1 972 720 comparaisons de mots, PASS.
- Conformité native des candidats : 1 684 047 comparaisons, PASS ; ce total
  inclut les comparateurs Threefry. Aucun candidat exclu à la compilation.
- Composant de multiplication spécialisée : 1 048 592 valeurs vérifiées.
- Recherche : 936 mesures ; confirmation : 2 772 mesures.
- Sorties chronométrées Montmory : 1 584 mesures, 10 406 854 656 mots comptés
  avec les répétitions et les variantes. Cela correspond à 167 772 160 couples
  seed/indice distincts, car les petits volumes sont des préfixes des grands.
- Ensemble des références CPU fixes : 22 469 345 280 mots avec répétitions.
- RBG, répétabilité sur le même backend : 1 892 155 392 mots avec répétitions.

Les empreintes sont conservées, pas les tableaux de sortie complets. Les
contrôles des appels de chauffe ne sont pas enregistrés ligne par ligne dans
le CSV. Les vérifications finies ne prouvent pas l'absence universelle de bugs.

## Configurations à conserver

| Volume de sortie | Configuration retenue | Débit Go/s | Gain apparié sur Pallas précédent | Décision |
|---|---|---:|---:|---|
| 512 Kio | `pallas_original_32` | 2,120 | 0 % | Référence conservée |
| 8 Mio | `pallas_original_32` | 15,859 | 0 % | Référence conservée |
| 64 Mio | `pallas_carry_128` | 62,458 | +15,42 % | Nouvelle variante retenue |
| 128 Mio | `pallas_carry_64` | 89,488 | +23,25 % | Nouvelle variante retenue |

À 64 Mio, le gain du candidat retenu est positif pour les sept seeds
(+9,61 à +19,22 %) et les 21 couples seed/bloc. À 128 Mio, il est positif pour
les sept seeds (+19,97 à +25,23 %) et les 21 couples seed/bloc. Ces observations
partagent une session ; les 21 couples ne sont pas 21 sessions indépendantes.

À 512 Kio, le primaire choisi pendant la recherche (`pallas_original_256`)
donne -0,63 % en confirmation et ne gagne que pour deux seeds sur sept.
À 8 Mio, `pallas_carry_128` donne +1,64 % et six seeds sur sept : il ne satisfait
pas le seuil prévu de +3 % avec sept gains sur sept.

Le secondaire `pallas_carry_64` atteint 63,990 Go/s à 64 Mio, davantage que le
primaire. Il n'est pas promu après consultation des résultats de confirmation,
conformément au protocole. Une nouvelle campagne avec sélection préalable
serait nécessaire pour cette décision. Le plus gros bloc n'est pas toujours
le plus rapide : 256 lignes ne gagne pas la confirmation.

## Comparaison avec les autres implémentations

Débits calculés à partir de la médiane des temps médians par seed :

| Implémentation | 512 Kio | 8 Mio | 64 Mio | 128 Mio |
|---|---:|---:|---:|---:|
| Montmory retenu | 2,120 | 15,859 | 62,458 | 89,488 |
| Montmory Pallas précédent | 2,120 | 15,859 | 54,200 | 73,498 |
| Montmory JAX portable | 2,026 | 15,138 | 51,075 | 66,769 |
| Threefry2x32 JAX | 1,979 | 12,982 | 31,954 | 33,364 |
| Threefry2x32 Pallas sélectionné | 2,013 | 15,917 | 65,249 | 93,006 |
| Threefry4x32 JAX | 1,921 | 10,691 | 20,396 | 21,331 |
| Philox2x32 JAX | 1,885 | 10,650 | 20,302 | 20,925 |
| Philox4x32 JAX | 1,664 | 7,867 | 12,193 | 12,463 |
| RBG, même backend uniquement | 2,096 | 18,465 | 86,811 | 90,791 |
| Témoin d'écriture, non PRNG | 2,131 | 17,790 | 91,322 | 151,754 |

Les variantes Threefry Pallas retenues utilisent respectivement 16, 32, 128,
128 lignes. Elles ont vingt tours, et préservent les sorties de l'API JAX
Threefry2x32 utilisée dans cette campagne. Leurs empreintes CPU concordent.

Les rapports appariés Montmory/Threefry2 JAX atteignent 1,926 à 64 Mio et
2,692 à 128 Mio. Face à Threefry Pallas, ils sont 0,960 et 0,961 : Montmory
est donc environ 4 % moins rapide sur cette mesure de débit. Il perd pour
six seeds sur sept à 64 Mio et pour sept sur sept à 128 Mio. La différence
est moins régulière entre blocs : Montmory gagne sept couples seed/bloc sur
21 à 64 Mio et six sur 21 à 128 Mio. Aucune supériorité générale de Montmory
sur Threefry n'est démontrée.

À 128 Mio, Montmory est proche de RBG (rapport apparié 0,994 ; trois gains
sur sept), mais le contrôle RBG ne garantit pas la reproduction du flux entre
architectures. À 64 Mio, RBG est nettement devant (rapport Montmory/RBG 0,716).
Les très grands écarts face aux API JAX Philox ne prouvent pas que Montmory
domine toutes les implémentations optimisées de Philox : aucun noyau Philox
Pallas spécialisé n'est mesuré ici.

Le témoin d'écriture est un point de comparaison, pas un PRNG ni une borne
matérielle universelle. À 8 Mio, RBG le dépasse légèrement ; les instructions,
les lancements et les voies d'exécution des deux programmes diffèrent. Cette
mesure seule ne permet pas d'en attribuer la cause au matériel.

## Interprétation des gains et portée

Le gain apparié est la médiane, sur les sept seeds, de
`temps_reference / temps_candidat`. Les débits du tableau utilisent séparément
la médiane des temps de chaque moteur. Le quotient des débits affichés peut
donc différer du gain apparié. +23,25 % de débit correspond à environ 18,86 %
de temps économisé selon le rapport apparié, pas à 23,25 % de temps économisé.

Les temps médians du Montmory retenu sont 247,27 µs, 528,94 µs, 1 074,47 µs et
1 499,84 µs. Les écarts absolus médians relatifs, calculés par seed puis
résumés par leur médiane, valent environ 3,06 %, 5,70 %, 4,04 % et 3,07 %.
Le temps inclut lancement par l'hôte, allocation et attente de complétion.
Il exclut la compilation, la préparation des clés, les transferts de sortie
et leur contrôle. Il ne s'agit pas d'un profil du noyau seul ni d'une mesure
de calcul VaR complet. Les petites sorties sont compatibles avec un coût fixe
important ; ce constat ne permet pas de séparer précisément ses composantes.

Les deux grands volumes couvrent les 2048 lanes logiques. Les valeurs 64 et
128 dans les noms sont des dimensions de tuile Pallas, pas un nombre de cœurs
physiques ou une modification du nombre de flux logiques. La campagne ne
mesure pas des affinités explicites par cœur physique.

Le moteur mathématique, ses constantes, ses trois tours, la seed et la
correspondance indice/lane sont inchangés. Le calcul de retenue est une
identité arithmétique exacte ; les sorties TPU concordent avec les références
CPU historiques. Aucun nouveau passage natif ARM/GPU ni TestU01/TACM n'a été
effectué pendant cette campagne. La reproductibilité concerne les mots et les
mantisses entières, pas tous les calculs flottants ultérieurs d'une application.

La sélection confirmée est utilisable par `tuned.load_tpu_selection` à partir
du `campaign/report.json` reçu, pour le même type de TPU et les mêmes versions
JAX/jaxlib. `tuned.generate` conserve le chemin portable pour les autres
architectures et tailles. Une nouvelle session doit encore confirmer la
stabilité des performances, en particulier l'écart avec Threefry Pallas.

## Fichiers

- `audit.json` : toutes les vérifications, empreintes des fichiers reçus,
  comparaisons appariées par seed et par bloc.
- `confirmation-summary.csv` : synthèse des 44 configurations confirmées.
- `confirmation-tpu.png` et `confirmation-tpu.pdf` : comparaison graphique.
- Sources de l'audit et du graphique : `../../audit_results.py` et
  `../../plot_results.py`.
