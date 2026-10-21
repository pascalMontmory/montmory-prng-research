# Vocabulaire de lecture

[Français](glossary.md) · [English](en/glossary.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Lire le CSV](csv-guide.md)

| Terme | Sens dans ce dépôt |
|---|---|
| PRNG | Algorithme déterministe produisant une suite destinée à des usages pseudo-aléatoires |
| Seed | Paramètre initial ; elle ne décrit pas seule tout le contrat du flux |
| `raw64` | Mot entier non signé de 64 bits |
| `mantissa53` | Les 53 bits hauts d'un mot, conservés comme entier ; multiplication par `2^-53` pour obtenir l'uniforme binary64 prescrit |
| Conformité | Accord entre une implémentation et la référence pour les entrées contrôlées |
| Qualité statistique | Comportement du flux face aux défauts visés par des tests spécifiés |
| Calibration | Vérification du comportement des seuils d'un diagnostic sur des contrôles ; ce n'est pas le réglage d'un PRNG pour obtenir une p-value favorable |
| MSE | Erreur quadratique moyenne d'un estimateur par rapport à sa valeur de référence |
| Biais | Différence entre l'espérance d'un estimateur et la cible ; son estimation finie fluctue |
| Lane logique | Identifiant utilisé par la formule du moteur ; ne désigne pas un cœur physique |
| Worker | Tâche d'exécution CPU ; le nombre de workers peut varier sans changer le flux si les indices sont préservés |
| Pallas | API JAX de noyaux spécialisés ; ici, une réalisation TPU du même calcul entier |
| Tuile | Bloc de sorties traité ensemble par un noyau Pallas |
| Recherche / confirmation | Sélection sur un premier ensemble de seeds, puis évaluation sur des seeds différentes |
| Ratio apparié | Rapport des temps des deux moteurs pour une même seed, ensuite résumé sur les seeds |
| Empreinte SHA256 | Résumé déterministe de données utilisé pour vérifier leur conservation ou leur concordance |
| RBG | Comparateur JAX traité séparément ; contrôle de répétabilité sur le même backend seulement |
| Témoin d'écriture | Programme produisant des écritures mémoire pour comparaison ; il ne génère pas un flux PRNG |

Un test de conformité réussi, une batterie statistique réussie et un gain de débit répondent à trois questions différentes. Aucun des trois ne remplace les deux autres.
