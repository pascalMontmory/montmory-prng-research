# État au 21 octobre 2026

- Protocole séparant parallélisme physique et géométrie logique enregistré.
- Campagne CPU : 99 mesures, 11 configurations, 9 seeds, 830 472 192 mots comparés intégralement à la référence C ; PASS.
- Toutes les configurations à géométrie fixe produisent le même flux. Les six autres géométries modifient le flux.
- Meilleure médiane CPU observée : 4 workers, 7,38 Go/s, accélération 3,65 par rapport à 1 worker. Cause du ralentissement à 8/16 non établie.
- Vérification portable des kernels Metal inchangés : 904 cas, 167 444 480 mots, UBSan et gardes ; PASS.
- Kit autonome M3 préparé. Compilation et mesures natives M3 en attente de son exécution, après la batterie BigCrush en cours sur le Mac.
- Aucun changement au moteur, aux anciens kits, ni aux travaux TACM dans cette étude. Aucune revendication de qualité statistique nouvelle.
