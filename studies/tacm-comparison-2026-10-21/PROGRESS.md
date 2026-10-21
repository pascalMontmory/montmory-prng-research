# Journal de campagne — 21 octobre 2026

- Objectif utilisateur : comparer les PRNG existants à Montmory avec TACM et chercher un avantage vérifiable.
- Dossier indépendant créé ; aucune modification des snapshots récupérés, des kits Mac ou du dépôt cryptovar.
- Protocole `TACM-comparison-1` fixé avant mesure. SHA256 : `0d525700d1b15800694c567e03ccce0bb535e7d9dcf9ac15d9895592429d25bf`.
- Construction native réussie. Neuf vecteurs officiels Random123, 1 048 656 mots Montmory contre le cœur récupéré, 1 048 576 mots xoshiro contre le code officiel, 82 240 vérifications de remplissage/indexation. Exécutable de vérification avec UBSan, aucune erreur signalée.
- Début d'exécution enregistré à 15:17:28 UTC. Calibration pilote conservée : deux portes sur huit références ne passent pas avec 512 essais. Aucune comparaison fonctionnelle n'est encore lancée à ce stade.
- Avenant explicite de précision : une seule calibration indépendante de 4 096 essais par source, mêmes seuils ; réussite. Aucune modification du cœur PRNG ou de la règle de comparaison.
- Partie fonctionnelle terminée : 1 342 177 280 mots, dix moteurs, 122 880 estimations. Contrôles de 2 048 lanes sur trois moteurs et benchmark CPU terminés. Dernière phase enregistrée à 15:21:09 UTC.
- Analyse : 0/54 avantages établis, 0/93 alertes après correction, aucune duplication de signature de lane, aucune variance de groupe nulle.
- Bilan : absence d'avantage statistique démontré ; petit écart favorable de débit CPU face à Threefry2x32 dans cette interface, plusieurs références plus rapides. Aucune extrapolation au GPU ou aux propriétés de tous les flux.
- Revue primaire : Random123, NumPy, auteurs de xoshiro, NVIDIA, deux prépublications de mai 2026. Les alertes isolées de BigCrush et les taux de passage globaux sont interprétés avec leur multiplicité.
- Les fichiers d'estimations et les résultats intermédiaires sont conservés. Les mots bruts sont régénérables ; ils ne sont pas tous stockés.
- Contrôle indépendant : 180 estimations régénérées et recalculées par réductions scalaires ; accord à moins de 2×10^-15. Identités polynomiales vérifiées en fractions exactes.
- Rapport, revue des sources, méthodes, figure PNG/PDF et script de reproduction ajoutés ; liens locaux vérifiés. L'archive possède un manifeste SHA256 vérifié sur son contenu.
