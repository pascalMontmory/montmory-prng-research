# Prototype Sobol32 exact pour JAX et TPU

[Français](README.fr.md) · [English](README.md)

Développé par Pascal Montmory pour ENINCA CONSULTING. Cette expérimentation est
séparée : les moteurs Montmory/Threefry et leurs mesures historiques restent
inchangés. Les contributions originales restent soumises à la [licence du dépôt](../../LICENSE.fr.md).

**État : validation CPU et interprétation Pallas sur CPU uniquement. La compilation
TPU native, son débit, l’exécution GPU et les gains applicatifs restent à vérifier.**
Le pilote CPU est un contrôle fonctionnel, pas une mesure TPU. Aucune supériorité
sur toutes les implémentations Sobol existantes n’est revendiquée.

**Essayer :** [notebook Colab français](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/sobol/notebooks/Eninca_Sobol_TPU.ipynb)
ou [notebook Colab anglais](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/sobol/notebooks/Eninca_Sobol_TPU.en.ipynb).
Le notebook intègre son kit de sources contrôlé. Choisir un environnement TPU ;
l’accès dépend de votre allocation et de votre quota Colab. Le mode par défaut
est un petit pilote.

## Contrat exact

- 128 dimensions, 32 bits de directions, sortie `(points, 128)` en `uint32`.
- Indices logiques : `0 <= start < start+points <= 2**32`, sans bouclage du flux.
- Ordres des dimensions et des bits de direction figés. Le premier point est conservé.
- Directions et brouillage optionnel LMS avec décalage numérique fournis par
  **SciPy 1.17.1**. NumPy **2.4.6** et JAX/jaxlib **0.10.2** sont figés.
- `seed=None` désactive le brouillage ; les graines entières utilisent le LMS avec
  décalage de SciPy. Il ne s’agit pas d’un nouvel algorithme de brouillage. Cette
  préparation est effectuée sur le CPU.
- L’identité durable du flux est la **matrice de directions exportée avec son
  décalage**, pas seulement une graine. Chaque campagne sauvegarde ces paramètres
  en NPZ avec une empreinte SHA-256 canonique : directions en octets little-endian,
  puis décalage. Une mise à jour SciPy/NumPy ne doit pas les redéfinir silencieusement.
- Chaque mot est le XOR du décalage et des directions sélectionnées par le code
  de Gray de l’indice. CPU, GPU et TPU doivent rendre les mêmes entiers à paramètres
  et indices identiques. Seuls CPU et interprétation ont été exécutés ici.
- La conversion optionnelle `uniform24` utilise les 24 bits hauts pour produire
  des valeurs dyadiques float32 exactes dans `[0,1)`. Elle abandonne huit bits et
  **ne correspond pas** à la conversion float64 complète de SciPy. Le benchmark
  ne mesure que les sorties entières raw32.

L’API est expérimentale : le contrôle CPU `reference.check_range` doit précéder
`kernels.make_kernel`. Le chemin rapide `aligned=True` exige aussi un début
divisible par la taille du bloc ; la campagne vérifie cette condition. Le chemin
général traite les débuts arbitraires et les fins partielles par padding aligné
et découpage. Le padding abandonné peut boucler, les indices logiques retournés
ne bouclent jamais. Les allocations d’au moins `2**31` éléments sont hors périmètre.
Le partage entre plusieurs périphériques et les dimensions autres que 128 ne
sont pas implémentés.

## Optimisation étudiée

Pour B puissance de deux, une base b alignée et `0 <= r < B` :

```text
Gray(b+r) = Gray(b) XOR Gray(r)
Sobol(b+r) = common(b) XOR local_table[r]
```

Le décalage intervient une seule fois dans `common`, jamais dans la table locale.
Cette identité de linéarité connue est une technique d’implémentation, pas une
nouvelle construction Sobol. Le bit de Gray de la base en position `log2(B)-1`
est bien conservé. Le noyau partage la contribution de la base entre les points
du bloc au lieu de recalculer 32 termes pour chaque point.

Huit candidats : JAX direct par code de Gray ; JAX par blocs de 128/256/512 points ;
Pallas direct par blocs de 32 ; Pallas optimisé par blocs de 128/256/512. Les
paramètres restent des entrées à l’exécution, sans spécialisation à une graine
durant la compilation. Les tables occupent respectivement 64/128/256 Kio par jeu
de paramètres. Leur préparation et leur transfert sont mesurés séparément.

## Vérification et mesures

`verify.py` compare les sorties publiques SciPy, un vecteur explicite de huit
points, une récurrence séquentielle indépendante, les grands indices, les fins
partielles, les découpages, le dernier point autorisé, les plages interdites et
les limites de conversion float32. Les oracles partagent les données de directions
SciPy : ce n’est pas une reconstruction indépendante des tables Joe–Kuo. Pour les
grands indices, la récurrence CPU évite des milliards d’étapes SciPy fast-forward.

`run.py` interdit le repli de backend et conserve les diagnostics en cas d’échec.
En mode CPU, Pallas est interprété pour vérifier les résultats, **jamais chronométré**.
Le pilote CPU ne mesure que JAX CPU natif. Le mode TPU contrôle et mesure les huit
candidats nativement sur le périphérique annoncé par JAX.

La campagne complète utilise 512, 8 192 et 131 072 points (256 Kio, 4 Mio, 64 Mio).
Trois graines de recherche sélectionnent séparément le meilleur JAX, y compris
le chemin direct, et le meilleur Pallas. Cinq nouvelles graines confirment ensuite
ces choix figés, avec sept répétitions dans un ordre aléatoire. Le mode rapide
utilise deux volumes, une graine de recherche, deux de confirmation et trois
répétitions : il constitue seulement un pilote.

Compilation et échauffement précèdent les mesures. Chaque sortie chronométrée est
comparée intégralement à l’oracle CPU hors chronométrage. Le temps inclut l’appel,
l’allocation, les écritures et `block_until_ready`, mais exclut préparation,
transfert initial, compilation, échauffement, rapatriement et vérification. Il
s’agit donc d’un benchmark **à entrées résidentes et paramètres réutilisés**, pas
du coût d’un appel isolé à froid. `preparation.json` conserve les durées exclues.
La plage d’indices est alignée et varie selon le cas ; tous les candidats d’un
même cas reçoivent la même plage.

Les rapports conservent temps CSV bruts, sélection, ratios appariés par graine,
empreintes des sources et paramètres, versions et nombres de mots comparés.
Un ratio supérieur à un signifie plus rapide que le JAX sélectionné. La médiane
des ratios appariés peut différer du quotient des débits récapitulatifs. Le pilote
n’établit aucun gain universel ou statistiquement significatif.

Le [rapport CPU conservé](validation/cpu/report.json) compte 2 097 152 mots comparés
aux sorties publiques SciPy et 14 314 496 mots JAX/Pallas interprétés contrôlés
pendant la conformité. Les [mesures brutes du pilote](validation/cpu/measurements.csv)
concernent uniquement le CPU ; le chemin par blocs ne gagne pas systématiquement
selon les tailles et les cas. Paramètres et environnement sont conservés avec le rapport.

## Exécution

Utiliser un environnement virtuel séparé, avec une version Python compatible :

```bash
python -m pip install 'jax==0.10.2' 'jaxlib==0.10.2' 'numpy==2.4.6' 'scipy==1.17.1'
python applications/sobol/run.py --backend cpu --quick --out build/sobol-cpu
```

Sur un TPU alloué, installer les mêmes versions NumPy/SciPy et `jax[tpu]==0.10.2`,
puis choisir `--backend tpu`. `--verify-only` lance seulement la conformité ;
retirer `--quick` lance la campagne complète. Le dossier de sortie doit être neuf.
Même en cas d’échec : `report.json`, `error.log` et archive `.tar.gz` adjacente.

Construire les notebooks Colab français/anglais et le kit contrôlé :

```bash
python applications/sobol/build_notebook.py --out build/sobol-delivery
```

Le notebook commence par vérifier l’allocation TPU avec les journaux complets.
Il propose le mode rapide par défaut. Il ne se replie pas sur CPU, ne demande pas
de ressources cloud et ne transmet pas les résultats automatiquement. La dernière
cellule télécharge résultats ou diagnostics de démarrage.

## Périmètre et sources

Sobol est une suite à faible discrépance, **pas un PRNG i.i.d.** Une portion
arbitraire reste un extrait exact mais ne conserve pas nécessairement les propriétés
d’équilibrage d’un ensemble complet de `2**m` points. Aucun passage de TestU01 ni
comparaison d’erreur ou de convergence applicative n’est revendiqué.

- [SciPy Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) : directions, LMS avec décalage et conditions d’équilibrage.
- [Données de directions Joe–Kuo](https://web.maths.unsw.edu.au/~fkuo/sobol/) : source citée par SciPy.
- [JAX Pallas TPU](https://docs.jax.dev/en/latest/pallas/tpu/details.html) : contraintes de mémoire et de blocs propres au backend.
- [Notices des tiers](../../THIRD_PARTY_NOTICES.fr.md) : SciPy/NumPy/JAX restent des dépendances externes. Aucune propriété de Sobol, de ses tables ou des implémentations amont n’est revendiquée.
