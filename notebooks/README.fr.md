# Notebooks TPU natifs

[Français](README.fr.md) · [English](README.md) · [Documentation FR/EN](../docs/languages.md)

| Campagne | Français | English |
|---|---|---|
| Optimisation, recherche et confirmation sur de nouvelles seeds | [Original](Montmory_TPU_optimisation.ipynb) | [Guide traduit](Montmory_TPU_optimisation.en.ipynb) |
| Validation du port initial et mesures | [Original](Montmory_TPU_validation.ipynb) | [Guide traduit](Montmory_TPU_validation.en.ipynb) |
| Petit réseau avec dropout — validation native TPU en attente | [Notebook](../applications/dropout/notebooks/Montmory_Dropout_TPU.ipynb) | [Notebook](../applications/dropout/notebooks/Montmory_Dropout_TPU.en.ipynb) |
| Diffusion / flow matching de chiffres — validation native TPU en attente | [Notebook](../applications/diffusion/notebooks/Eninca_Diffusion_TPU.ipynb) | [Notebook](../applications/diffusion/notebooks/Eninca_Diffusion_TPU.en.ipynb) |
| Options asiatiques Monte-Carlo / Sobol — validation native TPU en attente | [Notebook](../applications/asian_option/notebooks/Eninca_Asian_Option_TPU.ipynb) | [Notebook](../applications/asian_option/notebooks/Eninca_Asian_Option_TPU.en.ipynb) |

Pour les débits bruts des PRNG, commencer par le notebook d'optimisation. Pour une étape complète d’entraînement, consulter l’[expérience dropout et ses limites](../applications/dropout/README.fr.md), contrôlée sur CPU uniquement à ce stade. Télécharger le notebook choisi, l'ouvrir dans Colab, sélectionner un environnement **TPU** effectif et exécuter ses cellules de code dans l'ordre. L'accès dépend du compte et des quotas. Le notebook embarque le kit exécutable ; aucun identifiant GitHub n'est nécessaire après téléchargement. Conserver séparément chaque nouvelle archive de résultats.

Les éditions anglaises traduisent uniquement les cellules explicatives Markdown. Chaque cellule exécutable, kit embarqué, sortie enregistrée et valeur des métadonnées du notebook est identique à l'original français. Les diagnostics restent donc dans leur langue d'origine. La commande `python3 scripts/check_translations.py`, depuis la racine du dépôt, contrôle cette équivalence sans exécuter la campagne.

Le notebook de validation initiale décrit les connaissances au moment de sa préparation. Sa mention de résultats TPU natifs encore attendus est historique ; le [registre actuel des preuves](../docs/evidence.md) décrit les campagnes natives réalisées ensuite. Les notebooks originaux et les kits figés sont conservés octet pour octet.

Lire le [guide de reproduction](../docs/reproduce.md), les [résultats](../docs/results.md) et le [guide du CSV](../docs/csv-guide.md) avant d'interpréter une exécution. Traduire ces notebooks ne relance ni les benchmarks ni les batteries statistiques et n'accorde aucun nouveau droit d'utilisation.
