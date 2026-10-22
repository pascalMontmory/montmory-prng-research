# Petit modèle génératif : diffusion / flow matching

[Français](README.fr.md) · [English](README.md) · [Accueil](../../README.fr.md)

Par Pascal Montmory pour ENINCA CONSULTING. **Prototype exécuté sur CPU ; compilation native et performances TPU non validées.** Les noyaux Montmory et Threefry historiques restent inchangés. La [notice de droits](../../LICENSE) et les [notices des tiers](../../THIRD_PARTY_NOTICES.fr.md) s’appliquent.

[Ouvrir dans Colab](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/diffusion/notebooks/Eninca_Diffusion_TPU.ipynb) · [Notebook français](notebooks/Eninca_Diffusion_TPU.ipynb) · [English notebook](notebooks/Eninca_Diffusion_TPU.en.ipynb)

## Question étudiée

La génération de bruit et son application dans un même noyau Pallas améliorent-elles le temps d’une étape d’entraînement complète ? Le comparateur principal est le chemin JAX de la même famille, avec la même transformation gaussienne. Le programme conserve aussi `jax.random.normal` comme référence applicative courante. Une régression de performance est un résultat possible et doit être conservée.

Le modèle apprend à transformer un bruit gaussien en images de chiffres de 8 × 8 pixels. C’est un **flow matching rectifié**, proche de l’exemple génératif de Flax ; ce n’est ni une réimplémentation de Stable Diffusion ni une démonstration sur un grand modèle. L’interpolation est `x_t=(1-t)z+t*x`, avec cible de vitesse `x-z`, `t` uniforme dans `[0,1)` et perte quadratique.

## Protocole fixé

- Données `sklearn.datasets.load_digits` : 1 797 images, 1 400 pour apprendre et 397 réservées à la mesure. Pixels ramenés dans `[-1,1]`, permutation fixe et empreinte des données enregistrée. Les données viennent de scikit-learn, pas d’un jeu créé par ENINCA.
- MLP 80 → 128 → 128 → 64 : pixels et 16 caractéristiques sinusoïdales du temps, activations SiLU, Adam à 0,001. Lots de 128. Initialisations, exemples, temps et ordre des lots identiques entre variantes pour chaque seed.
- Pilote `--quick` : 300 mises à jour, seeds 7 et 19. Protocole complet : 1 500 mises à jour, seeds 7, 19 et 42. Aucun choix du meilleur entraînement après observation.
- CPU : Threefry JAX/Box–Muller, Montmory JAX/Box–Muller, Threefry avec normales JAX natives. TPU : ajout des deux chemins Pallas, tuile fixe de 8 × 128 paires.
- Validation sur un bruit et des temps réservés, communs à tous les modèles. La MSE de vitesse ne mesure pas à elle seule la qualité des images. Aucun FID ni avantage statistique d’un moteur n’est revendiqué.
- Génération : intégration d’Euler en 40 pas à partir de 64 bruits initiaux fixes, communs aux variantes. Tous les résultats numériques sont sauvegardés ; le graphique montre les huit premiers, sans sélection. Seul l’affichage est borné à `[-1,1]`.

## Contrat et conformité

Les uniformes suivent le maillage exact float32 `((uint32 >> 9)+0.5)*2^-23`, puis Box–Muller utilise ses deux sorties. La normale obtenue est une approximation discrète à queues limitées. `jax.random.normal` utilise une autre transformation : ses sorties ne sont pas supposées égales à celles de Box–Muller.

Montmory conserve T=2048,D=4096. La paire `i` utilise les moitiés basse et haute du mot `base+i`. À l’étape `s`, la base vaut `2^32-4097+s*(batch*64/2)`. Threefry utilise les positions aplaties `2*i,2*i+1` du flux partitionnable JAX à vingt tours, avec clé dérivée par `fold_in(seed_high)` puis `fold_in(step)`. Changer la forme du lot change le contrat applicatif Montmory. Aucun contrat distribué n’est ajouté.

Le contrôle compare Montmory à la référence C, Threefry à l’API JAX, et les sorties fusionnées à une transformation NumPy float64. Il couvre des fins de blocs, deux tuiles, des seeds extrêmes et des retenues. Une mise à jour Adam complète est comparée entre JAX et Pallas pour chaque moteur. Sur CPU, **Pallas est interprété pour la conformité et jamais chronométré comme un TPU**.

Tolérances : sorties fusionnées `atol=rtol=4e-5`, mise à jour de contrôle `atol=5e-6,rtol=5e-4` ; campagnes TPU, pertes et états entraînés `atol=2e-4,rtol=2e-3`. L’égalité des entiers ne garantit pas des transformations flottantes ou un entraînement bit à bit identiques entre architectures. Le programme mesure la différentiation des poids ; il ne fournit pas une primitive générique différentiable par rapport aux images ou aux paramètres RNG.

## Mesure et résultats disponibles

Toutes les étapes complètes sont compilées et amorcées avant les mesures. L’ordre des variantes est mélangé à chaque étape. Le temps inclut génération du bruit, préparation, passage avant, gradients, Adam et synchronisation. Données résidentes ; préparation des lots, compilation, évaluation et génération d’images exclues. Le noyau Pallas produit aussi la cible de vitesse nécessaire à la perte : aucune réduction de mémoire globale n’est affirmée sans profilage.

[Rapport CPU](validation/cpu/report.json) · [Mesures par étape](validation/cpu/steps.csv) · [Courbes](validation/cpu/learning.csv). Le pilote démontre l’exécution et une baisse de la perte réservée, pas une supériorité commerciale ni une performance TPU. Il n’ajoute pas de batterie statistique PRNG.

![Courbe du pilote CPU](validation/cpu/learning.png)

![Premières images du pilote CPU, non sélectionnées](validation/cpu/samples.png)

## Reproduire

Python 3.11+ et compilateur C, depuis la racine du dépôt :

```bash
python -m pip install -r applications/stochastic/requirements-cpu.txt
python applications/diffusion/run.py --backend cpu --quick --out build/diffusion-new
# Contrôles seuls, sans entraînement : ajouter --verify-only.
```

Sur un TPU effectivement alloué, utiliser le notebook ou installer `jax[tpu]==0.10.2` en plus des dépendances épinglées, puis remplacer `--backend cpu` par `--backend tpu`. Retirer `--quick` pour le protocole complet. Chaque sortie doit être un nouveau dossier. Un échec produit un diagnostic ; aucun repli silencieux sur CPU. Le compte Colab et les quotas utilisés sont ceux de la personne qui lance le notebook. Aucun envoi automatique.

Le [générateur de notebooks](../stochastic/build_notebooks.py) produit les deux langues et un kit autonome contrôlé par SHA-256. `report.json` contient les versions, les sources, l’environnement, le protocole et les résultats. Les durées CPU ne prédisent pas les durées TPU.

## Références

- [Exemple génératif officiel Flax](https://flax.readthedocs.io/en/latest/examples/digits_diffusion_model.html) : contexte du flow matching ; notre MLP n’en reprend pas l’architecture U-Net.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) et [Flow Straight and Fast](https://arxiv.org/abs/2209.03003).
- [Données de chiffres scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), provenant du jeu UCI Optical Recognition of Handwritten Digits.
- [PRNG dans Pallas](https://docs.jax.dev/en/latest/pallas/tpu/prng.html).
