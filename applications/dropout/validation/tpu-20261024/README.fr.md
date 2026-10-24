# Dropout natif TPU — résultats du 24 octobre 2026

[Français](README.fr.md) · [English](README.md)

[Expérience et contrat des masques](../../README.fr.md) · [Rapport original](report.json) · [Audit recalculé](analysis/audit.json)

**Le dropout fonctionne dans un entraînement natif TPU. Pallas réduit le temps
d’étape de Montmory de 5,10 à 6,27 % par rapport à Montmory JAX. Les performances
restent proches de Threefry ; aucun avantage général sur ce comparateur n’est établi.**

## Exécution reçue

Une session Colab démarrée le 24 octobre 2026 à 10:11:28 UTC sur un **TPU v5 lite**.
JAX/jaxlib 0.10.2, libtpu 0.0.42.1, NumPy 2.1.3, précision des multiplications
matricielles `highest`. Un MLP 32 → 128 → 128 → 4, 21 252 paramètres, lots de 64,
dropout de 25 %, SGD à 0,1, 150 mises à jour, graines 7, 19 et 42.
La tâche synthétique utilise 2 048 exemples d’entraînement et 512 de validation.
Les quatre variantes partagent données, ordre des lots et poids initiaux par graine.

Les mesures incluent passage avant, passage arrière, mise à jour, lancement et
synchronisation. Données et lots résident sur l’appareil. Préparation, extraction
du lot, compilation et évaluation de validation sont exclues des temps d’étape.
Toutes les variantes sont compilées et amorcées avant mesure, puis intercalées
dans un ordre aléatoire à chaque étape. La durée totale enregistrée de 17,55 s
comprend aussi les vérifications et la compilation ; ce n’est pas le temps d’une
seule variante d’entraînement.

## Conformité

- Les 19 empreintes du manifeste correspondent au [kit de sources original](source-kit.tar.gz),
  dont les 10 sources d’exécution sont identiques aux fichiers du dépôt à cette publication.
- La reconstruction CPU des données et de l’ordre des lots reproduit leur empreinte.
- Les 1 800 lignes de [mesures](steps.csv) sont complètes, sans doublon ; les médianes
  et ratios du rapport sont recalculés indépendamment du programme d’entraînement.
- Le rapport natif enregistre **49 672 comparaisons d’éléments de masque** et autant
  de contrôles de gradient réussis, deux tailles de tuiles, des fins de blocs,
  quatre graines de conformité et six contrôles d’adressage.
- Les **900 paires de pertes d’entraînement JAX/Pallas** et les 18 paires de points
  de validation sont numériquement identiques, à moteur et graine identiques.
- Le programme rapporte un écart maximal de **zéro entre les poids finaux** des six
  paires moteur/graine. Les tenseurs de poids ne figurent pas dans l’archive : ce
  dernier contrôle est rapporté par le programme et n’est pas refait depuis des poids exportés.

Ces contrôles ne prouvent ni une identité universelle entre architectures, ni une
nouvelle propriété statistique du PRNG. Aucune batterie TestU01 n’a été exécutée ici.

## Temps d’étape

Médiane de 150 observations par variante et graine, en millisecondes :

| Variante | Graine 7 | Graine 19 | Graine 42 |
|---|---:|---:|---:|
| Threefry JAX | 0,2988245 | 0,2885500 | 0,2834850 |
| Montmory JAX | 0,3194850 | 0,3013550 | 0,2994900 |
| Threefry Pallas | 0,3002600 | 0,2893900 | 0,2843500 |
| Montmory Pallas | 0,3010800 | 0,2859750 | 0,2807155 |

Pallas réduit le temps Montmory de **5,76 %, 5,10 % et 6,27 %**, soit environ
15 à 19 µs par étape médiane. Le ratio médian JAX/Pallas vaut 1,0611 :
**6,11 % de débit supplémentaire correspondent à 5,76 % de temps en moins**.

Face à Threefry JAX, Montmory Pallas prend 0,75 % de temps supplémentaire pour
la graine 7, puis 0,89 % et 0,98 % de moins pour les graines 19 et 42.
Face à Threefry Pallas, les écarts de temps sont respectivement +0,27 %, −1,18 %
et −1,28 %. Ces faibles écarts de signe variable ne démontrent pas un avantage
de vitesse général. Trois trajectoires d’une même session ne sont pas trois
sessions de benchmark indépendantes ; aucun intervalle de confiance n’est revendiqué.

[Statistiques détaillées](analysis/timing_summary.csv) · [Ratios et différences appariées](analysis/comparisons.csv) · [Blocs de 50 étapes](analysis/timing_by_block.csv)

## Threefry dans JAX et dans Pallas

Les deux chemins emploient **Threefry2x32 à 20 tours**. Le chemin JAX utilise
`jax.random.bernoulli` dans le graphe ordinaire. Notre chemin Pallas calcule
les mêmes mots aux mêmes indices dans un noyau qui génère et applique le masque.
Sa dérivée personnalisée régénère le masque pour le passage arrière.
Les résultats de cette exécution sont numériquement identiques, et Pallas prend
environ 0,3 à 0,5 % de temps supplémentaire sur l’étape complète Threefry.

Pallas est une [extension de JAX pour écrire des noyaux](https://docs.jax.dev/en/latest/pallas/quickstart.html).
Générer les valeurs dans le noyau consommateur peut réduire les échanges mémoire,
mais ne garantit pas une accélération de l’entraînement.
Le [PRNG matériel accessible depuis Pallas](https://docs.jax.dev/en/latest/pallas/tpu/prng.html#using-the-hardware-prng)
est un autre générateur ; **il n’est pas utilisé dans cette comparaison**.
Aucune économie de mémoire totale n’a été mesurée.

## Apprentissage et comparaison CPU/GPU

| Graine | Précision initiale | Threefry après 150 étapes | Montmory après 150 étapes |
|---|---:|---:|---:|
| 7 | 26,95 % | 84,18 % | 83,20 % |
| 19 | 27,93 % | 82,03 % | 82,81 % |
| 42 | 21,29 % | 82,81 % | 81,64 % |

Les précisions finales moyennes sont 83,01 % pour Threefry et 82,55 % pour Montmory.
Les pertes finales moyennes sont 0,4481 et 0,4504. Les deux apprennent cette tâche ;
ces trois graines ne démontrent ni supériorité ni équivalence statistique de qualité.
L’expérience ne montre pas que Montmory atteint une précision cible en moins d’étapes.

Le [contrôle CPU du 21 octobre](../cpu/report.json) utilise les mêmes sources
d’exécution, données, lots et dimensions. Ses temps médians Montmory JAX sont
0,447087 / 0,4218075 / 0,4179535 ms pour les mêmes graines. Leur rapprochement
avec le TPU donne environ **1,40× pour Montmory JAX** et **1,48× pour Montmory Pallas**,
soit 32,2 à 32,8 % de temps en moins pour ce dernier.
**Cette comparaison est indicative : l’essai CPU du VPS était un contrôle
fonctionnel dans une autre session, pas un benchmark matériel contrôlé.**
Elle ne permet pas d’attribuer l’écart au seul PRNG. Aucun entraînement GPU de ce
modèle n’a été mesuré. [Calculs CPU/TPU](analysis/cpu_comparisons.csv).

![Temps d’étape et courbes d’apprentissage](analysis/dropout-tpu-analysis.png)

La figure utilise la moyenne et l’étendue min–max des trois graines pour les courbes,
sans intervalle de confiance. [Version PDF](analysis/dropout-tpu-analysis.pdf).

## Refaire l’analyse et poursuivre

Depuis la racine du dépôt, avec NumPy installé :

```bash
python3 applications/dropout/analyze.py --out build/dropout-tpu-audit
```

Ajouter `--plots` avec Matplotlib installé pour régénérer les figures. Ce script
vérifie les archives, les sources et les mesures ; il ne lance aucun entraînement TPU.
Les résultats dérivés sont écrits hors de ce dossier de preuves.

Le prochain test prévu par le protocole est `--batch 256 --width 256`, puis des
répétitions indépendantes avec les mêmes conditions pour les quatre variantes.
Le petit modèle actuel ne permet pas d’extrapoler aux grands modèles, à plusieurs
appareils, à la mémoire ou à l’énergie. Un avantage pratique sur Threefry reste à démontrer.

Le [prototype Sobol32 pour TPU](../../../sobol/README.fr.md) est une expérience
distincte de points à faible discrépance, avec ses propres notebooks. Sa conformité
CPU est vérifiée ; sa compilation et ses performances natives TPU restent à mesurer.
Il ne constitue pas une variante du dropout mesuré ici.

## Provenance conservée

[Archive reçue](dropout-tpu.tar.gz), [manifeste reçu](manifest.json), [journal](run.log),
[courbes brutes](learning_curves.csv), [empreintes](SHA256SUMS) et kit original sont
conservés sans modification. Les README embarqués décrivent l’état antérieur à cette
exécution ; ils ne remplacent pas cette analyse datée. Le commit Git n’était pas
disponible dans Colab : la provenance repose sur les empreintes de fichiers.

SHA-256 de l’archive reçue :
`db392503b39399a86c6307a0700d226ccef46cadd40e95e70328d99a0fdd6deb`.
Le seul avertissement du journal concerne les transparent hugepages au démarrage
et à l’arrêt du runtime ; aucun échec n’est rapporté.
