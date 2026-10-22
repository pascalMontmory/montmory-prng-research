# Entraînement d’un petit MLP avec dropout Montmory et Threefry

[Français](README.fr.md) · [English](README.md)

Cette expérience cherche à savoir si un PRNG logiciel optimisé pour TPU améliore une véritable étape d’entraînement. Elle ajoute un dropout fixe de 25 % à un petit classifieur. Elle ne modifie aucune source historique du PRNG, constante, nombre de tours ou géométrie logique Montmory. Développé par Pascal Montmory pour ENINCA CONSULTING ; la [notice de droits](../../LICENSE) du dépôt s’applique.

**Statut : prototype fonctionnel contrôlé sur CPU. La compilation et les performances d’entraînement natives TPU restent à valider.** Les mesures TPU antérieures du générateur seul ne répondent pas à la question de performance de cette application.

[Ouvrir dans Colab — français](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/dropout/notebooks/Montmory_Dropout_TPU.ipynb) · [Open in Colab — English](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/dropout/notebooks/Montmory_Dropout_TPU.en.ipynb). Les notebooks embarquent les sources et leurs empreintes ; l’exécution nécessite un TPU effectivement alloué sur le compte de la personne qui les lance. Les fichiers sont aussi consultables [ici en français](notebooks/Montmory_Dropout_TPU.ipynb) et [en anglais](notebooks/Montmory_Dropout_TPU.en.ipynb).

## Ce qui tourne

- Un MLP 32 → 128 → 128 → 4, ReLU, dropout après les deux couches cachées et SGD ; 21 252 paramètres, lots de 64, 150 mises à jour, graines 7, 19 et 42.
- Une tâche supervisée synthétique : 2 048 exemples d’entraînement et 512 exemples réservés à la validation, étiquetés par un modèle linéaire fixe. Tous les chemins utilisent les mêmes poids initiaux, exemples et ordre des lots pour chaque graine. C’est un contrôle élémentaire d’apprentissage, pas une mesure de qualité sur des données réelles.
- CPU : entraînement des implémentations JAX ordinaires de Threefry et Montmory ; interprétation des noyaux Pallas pour vérifier les masques, gradients et une mise à jour complète. L’interprétation CPU ne mesure pas les performances TPU.
- TPU : entraînement des quatre chemins, en ajoutant le dropout fusionné Pallas Threefry et Pallas Montmory. La tuile par défaut est de 32 × 128 pour les deux ; aucune recherche de paramètres ni sélection du meilleur essai.

Le noyau Pallas génère et applique le masque dans le même appel. Sa dérivée en mode inverse régénère le même masque depuis les paramètres RNG sauvegardés. Les autres parties du graphe peuvent conserver des activations. Aucune réduction de mémoire totale n’est revendiquée sans profilage. Le prototype prend en charge float32, une probabilité fixe, un seul appareil et la différentiation en mode inverse. Il ne fournit pas une couche Flax générique, la différentiation en mode direct ou l’entraînement distribué.

## Contrat exact des masques

Conserver une activation lorsque son mot uint32 est inférieur à `0xC0000000` ; multiplier les valeurs conservées par `1 / 0.75`. Le seuil est exact. Cela n’établit aucune nouvelle garantie de qualité statistique de Montmory.

- **Threefry :** l’implémentation Threefry2x32 à 20 tours du kit figé, avec des indices globaux aplatis inférieurs à 2³². Comparaison à `jax.random.bernoulli(..., p=0.75, mode="low")`, Threefry partitionnable activé. Les clés sont dérivées de la graine, de l’étape et de la couche avec `fold_in` de JAX.
- **Montmory :** flux T=2048, D=4096 inchangé. L’activation `i` utilise low32 du mot `base + floor(i/2)` si elle est paire, high32 sinon. Pour une charge fixe, réserver `ceil(batch × width / 2)` mots par couche. La couche `l` à l’étape `s` commence à `2³² - 4097 + (2s+l) × stride`. Les deux couches cachées ont la même largeur. Les limites du programme maintiennent ces adresses dans uint64. C’est une correspondance applicative explicite, pas un nouvel algorithme de mélange ou une preuve d’indépendance des sous-flux.

Changer les tuiles physiques conserve cette correspondance. Changer la taille du lot ou la largeur change l’intervalle réservé et donc les masques suivants. L’invariance entre formes différentes ou plusieurs appareils n’est pas revendiquée. Chaque paire passage avant/arrière doit utiliser les mêmes paramètres. L’égalité des entiers et des masques se distingue de l’égalité de l’entraînement flottant entre CPU, GPU et TPU. Cette application n’a été exécutée localement que sur CPU pour le moment.

## Résultat local

Le premier essai CPU a passé 49 672 comparaisons d’éléments de masque fusionné et 49 672 contrôles d’éléments de gradient, avec deux tailles de tuiles, des fins de blocs et quatre graines. Les masques Montmory ont été comparés à la référence C scalaire indépendante. Les masques Threefry ont été comparés à l’API Bernoulli de JAX. Une mise à jour complète du MLP correspond exactement à sa référence pour chaque moteur sur ce CPU. Six contrôles supplémentaires de l’adressage couvrent les retenues et les grands numéros d’étape.

Après 150 mises à jour, la précision sur les données réservées atteint 82,0–84,2 % avec Threefry et 81,6–83,2 % avec Montmory, contre 21,3–27,9 % au départ. Les deux apprennent cette tâche simple. Ces trois essais n’établissent ni supériorité ni équivalence statistique des générateurs. La première exécution complète de vérification et d’entraînement a pris environ 18 secondes sur ce VPS, compilation CPU comprise. Ce n’est pas une prédiction de durée TPU.

Le rapport local archivé et les courbes d’apprentissage se trouvent dans [validation/cpu](validation/cpu). Les résultats natifs TPU restent à produire.

## Reproduire

Utiliser Python avec JAX et jaxlib 0.10.2, NumPy et un compilateur C. Pour un environnement CPU neuf :

```sh
python -m pip install 'jax==0.10.2'
python applications/dropout/run.py --backend cpu --out build/dropout-new-cpu
```

Sur un TPU alloué, installer les [dépendances TPU officielles de JAX](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu), avec la version de cette expérience :

```sh
python -m pip install 'jax[tpu]==0.10.2'
python applications/dropout/run.py --backend tpu --out build/dropout-new-tpu
```

Le répertoire de sortie doit être nouveau. L’absence de TPU provoque un échec explicite. `report.json`, `steps.csv`, `learning_curves.csv` et une archive de diagnostic enregistrent l’exécution. Un essai échoué n’est pas un résultat de performance. Pour une seconde charge, toujours petite, utiliser `--batch 256 --width 256` avec un nouveau répertoire de sortie.

Pour Colab, générer les notebooks autonomes français/anglais et le kit de sources :

```sh
python applications/dropout/build_notebook.py --out build/dropout-delivery
```

Importer `Montmory_Dropout_TPU.ipynb` (français) ou `Montmory_Dropout_TPU.en.ipynb` (anglais), choisir un environnement TPU et exécuter les quatre cellules de code. Les empreintes des sources sont contrôlées avant exécution. Télécharger les diagnostics même si la compilation échoue. Générer le notebook ne valide pas ses noyaux natifs TPU.

## Mesure et interprétation

Compiler et amorcer tous les pipelines d’étape complète avant les mesures. Randomiser l’ordre des variantes à chaque étape. Les tableaux résident sur l’appareil ; synchroniser chaque étape chronométrée. Inclure passage avant, passage arrière, mise à jour des paramètres, lancement et synchronisation. Exclure création des données, extraction du lot, compilation et évaluation de validation. C’est un benchmark d’étape d’entraînement avec données résidentes, pas le temps du travail entier incluant l’alimentation en données.

Comparer les suites de pertes et les poids finaux Pallas/JAX séparément pour chaque moteur, avec les tolérances fixées dans le code (rtol=2e-4, atol=2e-5). Des générateurs différents produisent volontairement des masques différents. Présenter les médianes de temps d’étape par graine et leurs ratios ; un ratio supérieur à 1 signifie que la version Pallas est plus rapide. Aucune répétition indépendante au-delà des trois trajectoires, intervalle de confiance, mesure énergétique ou extrapolation aux grands modèles n’est revendiquée. Un très petit modèle peut être dominé par le coût de lancement. Pallas peut aussi empêcher une fusion accessible à JAX ordinaire : un ralentissement serait un résultat valide.

L’implémentation suit l’usage décrit dans la [documentation PRNG Pallas de JAX](https://docs.jax.dev/en/latest/pallas/tpu/prng.html) : générer les valeurs aléatoires dans le noyau qui les consomme. Il reste à mesurer si cela aide ce graphe d’entraînement précis.
