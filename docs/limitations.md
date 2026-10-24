# Limites et conclusions à ne pas dépasser

[Français](limitations.md) · [English](en/limitations.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md)

1. **Aucun PRNG universellement supérieur n'est établi.** Le classement dépend du compilateur, du matériel, de la taille et du chemin d'exécution. Threefry Pallas reste devant Montmory sur les grands volumes TPU confirmés.
2. **Aucune sécurité cryptographique n'est revendiquée.** Un Feistel, une longue période d'état et une réussite à TestU01 ne constituent pas une analyse de sécurité.
3. **L'uniformité exacte est réfutée pour le mélangeur documenté.** Le certificat de non-injectivité et le décompte de la première case de mantisse sont conservés. Leur impact applicatif n'est pas mesuré par ce seul certificat.
4. **L'indépendance des lanes n'est pas prouvée.** Les lanes de même seed sont des réindexations d'un même espace d'états. La reproductibilité ne doit pas être confondue avec l'indépendance statistique.
5. **Les passages TestU01 ont un périmètre précis.** Seeds, géométrie et adaptateur comptent. Les confirmations de code n'ajoutent pas de nouvelles batteries statistiques.
6. **TACM n'établit pas d'avantage statistique.** La campagne corrigée porte sur six fonctions et des tests explicités ; elle ne valide pas tous les diagnostics historiques ni tous les usages Monte Carlo.
7. **La performance TPU provient d'une session sur un périphérique.** Les sept seeds et 21 couples seed/bloc ne sont pas autant de sessions indépendantes. Aucune comparaison énergétique ni multi-TPU native n'est disponible.
8. **Le « bit à bit » concerne le contrat entier.** Les mêmes seeds avec une autre géométrie peuvent changer le flux. Les transformations flottantes et réductions d'une application nécessitent leurs propres contrôles.
9. **Le contrôle des sorties repose en partie sur des empreintes.** Les buffers complets ne sont pas archivés ; les données conservées permettent la recomposition des références et l'audit des empreintes, pas la réinspection de chaque buffer natif original.
10. **Les niveaux de preuve diffèrent.** Une valeur transmise par terminal reste distincte d'un CSV reçu et audité. Les pièces manquantes sont recensées dans le [registre](evidence.md).
11. **Les autres PRNG sont des comparateurs sérieux.** Philox et Threefry sont antérieurs à ce travail ; les PRNG à compteur, les clés séparables et le portage multiarchitecture ne sont pas présentés comme des inventions propres à Montmory.
12. **Le gain d’entraînement mesuré reste limité à un petit MLP synthétique.** Le [dropout natif TPU](../applications/dropout/validation/tpu-20261024/README.fr.md) réduit le temps de Montmory Pallas face à Montmory JAX, sans avantage général établi sur Threefry. Les mesures de génération seules ne prouvent ni réduction d’erreur VaR, ni gain de tarification, ni conformité réglementaire. Aucun gain sur de grands modèles, GPU, mémoire totale ou énergie n’est établi.

Ces limites font partie du résultat de recherche. Elles ne sont pas masquées par les optimisations réussies.
