# Historique des décisions et corrections

[Français](history.md) · [English](en/history.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md)

Ce tableau décrit la filiation des travaux disponibles au 21 octobre 2026. Il n'attribue pas de date d'exécution nouvelle aux journaux anciens récupérés.

| Étape | Travail / constat | Conséquence conservée |
|---|---|---|
| Récupération des sources | Référence C, Metal et CUDA, journaux TestU01 historiques | Sources et empreintes conservées ; filiation des journaux examinée |
| Audit mathématique | Compteur impair, Feistel bijectif ; mélangeur final non injectif | Période d'état distinguée de période de sortie ; pas d'uniformité exacte revendiquée |
| Audit CUDA historique | Garde de borne manquante pour une géométrie irrégulière ; ancien comparateur Philox incorrect et conversion de débit erronée | Anciens chiffres exclus des comparaisons probantes ; correction non présentée comme validée nativement |
| Comparaison CPU | Random123 et vecteurs connus, compilations et modes d'accès distincts | Classement dépendant du contexte |
| TACM corrigé | Pilote de calibration échoué, extension unique documentée et réussie ; aucun avantage MSE établi | Conservation du pilote et des résultats négatifs |
| Metal v3 | Réparation de l'intégration TestU01, conformité native ; trois batteries rapportées réussies | Dernier BigCrush classé « résumé terminal », archive intégrale encore absente |
| Optimisation CPU/Metal | Calcul par paires, workers, variantes de répartition | Gains CPU ; faible gain GPU ; première utilisation de certaines variantes contaminant les mesures initiales |
| Confirmation Metal séparée | Précompilation et chauffe ; nouvelles seeds | Threefry et Philox restent devant ; archive brute de cette confirmation manquante |
| Essais vectoriels | Échec initial Apple Clang sur déduction `auto` de `id`, puis réparation ; vérifications CPU d'ordre d'évaluation explicite | Traces conservées ; `vector4` GPU plus lent, pool CPU utile sur petits blocs |
| Port JAX et premiers essais Colab | Vérifications locales puis erreurs d'environnement sans TPU alloué | Les échecs d'initialisation ne sont pas des échecs du moteur ni des mesures TPU |
| Première campagne TPU 18:06 | Conformité native et 770 observations | Comparaison initiale sans Threefry Pallas spécialisé |
| Optimisation TPU 18:51 | Retenues exactes, tailles de tuile, ajout de Threefry Pallas ; 3 708 observations | Gains Montmory à 64/128 Mio ; Threefry Pallas demeure plus rapide |
| Constitution de ce dépôt | Documentation, provenance et audits reproductibles | Aucun changement de moteur ; pause des développements |

Les erreurs d'environnement, essais rejetés et chemins non adoptés restent accessibles dans les études. Leur présence dans une archive ne signifie pas qu'ils font partie de la configuration recommandée par le protocole.
