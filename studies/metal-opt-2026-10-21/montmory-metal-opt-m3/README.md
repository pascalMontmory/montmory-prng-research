# Montmory : optimisations CPU et Metal à flux fixé

Ce kit ajoute sept variantes Metal et trois variantes CPU au calcul indexé. Il conserve les lanes logiques de ton TestU01 : T=2048, D=4096, soit 8 388 608 mots de 64 bits par bloc. Les seeds, compteurs, tours de permutation, mélange final et formats de sortie ne sont pas modifiés. Il s'installe dans son propre dossier.

Le kit des lanes précédent explore la géométrie de lancement et les lanes logiques. Celui-ci conserve la géométrie logique et explore des réécritures du kernel ainsi que leur répartition sur les threads. Les kernels Philox et Threefry du kit v3 sont conservés comme références.

## Exécution sur Apple Silicon

Attendre la fin de BigCrush et de tout autre benchmark, puis :

```bash
bash run.sh
```

Cela lance successivement les mesures CPU puis Metal. Pour les exécuter séparément : `bash run.sh cpu` ou `bash run.sh metal`. Une archive de résultats et de diagnostics est créée à la fin, même en cas d'échec de compilation. Aucun résultat n'est transmis automatiquement.

## Variantes

- CPU `indexed` : compteur calculé à chaque index. Les variantes `incremental`, `incremental_pair`, `incremental_quad` exploitent la progression arithmétique dans chaque lane et calculent un, deux ou quatre mots ensemble. Les workers sont des threads C++ ; leur placement est laissé au système. La création et la jonction des threads font partie du temps hôte mesuré. Aucune bibliothèque OpenMP à installer sur le Mac.
- Metal `montmory_mulhi_quad` : quatre mots par thread avec les multiplications hautes du kernel v3 `mulhi_pair`.
- `montmory_stripe_pair`, `montmory_stripe_quad` : disposition où les threads voisins écrivent des mots voisins à chaque étape du lot.
- `montmory_const_pair`, `montmory_const_stripe_pair`, `montmory_const_stripe_quad` : rotations spécialisées suivant la seed, appliquées à la fonction de tour utilisant `mulhi`.
- `montmory_split_stripe_pair` : compteur incrémenté en deux mots de 32 bits avec retenue explicite.

L'original `montmory_mulhi_pair`, Threefry2x32-20, Philox4x32-10 et un témoin d'écriture sans génération aléatoire sont mesurés dans les mêmes conditions. Le témoin n'est pas un PRNG.

## Méthode de mesure

Le CPU utilise neuf seeds, quatre implémentations et 1, 2, 4, 8, 16 workers dans la limite du nombre de threads matériels déclaré par le système. Il écrit raw64 ; ses résultats sont descriptifs et ne sélectionnent pas automatiquement une implémentation de production.

Pour Metal, la recherche utilise sept seeds, 8192/32768/65536/131072 threads et des groupes de 64/128/256/512 dans les limites du pipeline. Elle mesure raw64 et mantissa53, toujours stockés en 64 bits. L'ordre des configurations est mélangé à chaque seed. La compilation des pipelines et les références CPU sont hors chronométrage.

La meilleure médiane est choisie séparément pour chaque moteur et format. Les choix sont enregistrés dans `selection-before-confirmation.json` **avant** une confirmation sur cinq nouvelles seeds. Le tableau final affiche les médianes de confirmation. Un ratio/v3 supérieur à 1 signifie un débit supérieur à celui de `montmory_mulhi_pair` réglé selon la même procédure ; le ratio est la médiane des rapports appariés par seed. Les quantiles sont descriptifs, pas des intervalles de confiance.

Chaque sortie de chauffe et chaque sortie chronométrée est comparée intégralement à la référence C originale ou à Random123. Avant la première chauffe de chaque mesure Metal, toutes les destinations sont remplies avec le complément de leur résultat attendu ; un mot non écrit ne peut pas hériter d'un ancien résultat correct. Les gardes mémoire sont contrôlées. Le temps GPU mesure génération et écritures ; le temps hôte de soumission et attente est également enregistré. Les comparaisons CPU sont exclues mais peuvent contribuer à l'échauffement : les états thermiques Metal sont enregistrés.

## Ce qui est vérifié et ce qui reste à mesurer

Les identités de réécriture sont exposées dans `EXACT.md`. La transcription C++ des sept kernels passe 4256 cas représentant 581 663 488 mots comparés à la référence, ainsi que 155 296 contrôles de composants, avec UndefinedBehaviorSanitizer et gardes mémoire.

Le VPS ne dispose pas de Metal : la compilation native et le gain éventuel sur M3 restent à établir. `run.sh` exige les contrôles natifs avant d'accepter les mesures. Une meilleure vitesse n'est pas garantie pour chaque variante. Aucun test SmallCrush, Crush, BigCrush ou TACM supplémentaire n'est exécuté par ce kit ; les données de ces tests sont préservées seulement lorsque le flux complet et son adaptateur sont identiques.

Le protocole Metal est dans `protocol.json`, les sorties individuelles dans `measurements.csv`, les résumés dans `summary.json`. Les équivalents CPU portent le préfixe `cpu-`. `pipeline-preparation.json` contient aussi les limites de groupes et la largeur SIMD annoncées par Metal.

## Reproduction des contrôles Linux

```bash
python3 tools/build_source.py
python3 tools/verify_optimizations.py
```

La bibliothèque `generated/all.metal` est la concaténation de la base v3 inchangée et de `src/optimizations.metal`. Les constantes de fonction Metal sont remplacées par des variables dans la transcription CPU ; l'efficacité et la correction de leur spécialisation par Apple doivent être contrôlées nativement. Les licences Random123 sont conservées dans `vendor/`.
