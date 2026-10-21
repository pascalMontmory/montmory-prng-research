# Registre TestU01

[Français](testu01.md) · [English](en/testu01.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Limites](limitations.md)

Une réussite TestU01 signifie que la batterie n'a pas signalé d'échec selon ses règles pour **le flux et l'adaptateur exécutés**. Elle n'est pas une preuve universelle de hasard, de sécurité cryptographique, d'indépendance des lanes ou de supériorité sur un autre moteur.

## Pièces disponibles

| Exécution | Résultats | Niveau de preuve disponible |
|---|---|---|
| Metal historique, `Montmory-GPU-3R+DXSM` | SmallCrush 15, Crush 144, BigCrush 160 statistiques : passes | Trois journaux complets importés avec les sources historiques |
| CUDA historique | Trois batteries annoncées réussies dans les journaux | Journaux complets importés ; provenance et défauts des anciens benchmarks examinés séparément |
| Reprises CPU de l'audit | Sept SmallCrush réussis, couvrant plusieurs seeds/adaptateurs et une géométrie irrégulière | Sources, métadonnées et journaux locaux conservés |
| Metal M3 natif v3, `montmory_mulhi_pair` | SmallCrush 15 : 4,74 s ; Crush 144 : 15 min 35,72 s ; BigCrush 160 : 1 h 44 min 36,42 s | Extraits/résumés de terminal transmis ; archive complète du dernier BigCrush non reçue |
| Optimisation TPU finale | Pas de nouvelle batterie statistique | Contrôles de conformité et performances uniquement |

Les nombres 15, 144 et 160 sont ceux des **statistiques du résumé**, pas nécessairement le nombre de lignes portant « p-value » dans tout le journal. Les temps sont des temps CPU rapportés par TestU01, pas des temps de noyau GPU.

## Journaux historiques complets

- [SmallCrush Metal](../studies/audit-2026-10-21/snapshot/montmory_gpu_test/montmory_gpu_test/logs/smallcrush_gpu.log)
- [Crush Metal](../studies/audit-2026-10-21/snapshot/montmory_gpu_test/montmory_gpu_test/logs/crush_gpu.log)
- [BigCrush Metal](../studies/audit-2026-10-21/snapshot/montmory_gpu_test/montmory_gpu_test/logs/bigcrush_gpu.log)
- [Journaux CUDA](../studies/audit-2026-10-21/snapshot/cuda-ovh/logs)
- [Métadonnées des reprises SmallCrush CPU](../studies/audit-2026-10-21/evidence/smallcrush-runs.json)

Les journaux historiques CUDA et Metal présentent des p-values imprimées identiques pour les batteries comparées. Les [analyses de filiation](../studies/audit-2026-10-21/evidence/cuda-bigcrush-lineage.json) en détaillent la portée. Deux fichiers portant des hôtes différents ne constituent donc pas automatiquement deux répétitions indépendantes.

La géométrie CUDA historique varie selon la batterie : `2048×4096`, `4096×8192`, `8192×16384`. La géométrie récente v3 reste `2048×4096`. Ces campagnes ne doivent pas être confondues.

## Dernier résultat BigCrush M3 v3

Le [résumé terminal conservé](../evidence/user-reported/m3-v3-bigcrush-summary.txt) annonce explicitement 160 statistiques et « All tests were passed ». Son [enregistrement de provenance](../evidence/user-reported/m3-v3-bigcrush.json) indique que l'archive native complète reste absente du dépôt. Ce texte est une transcription du message transmis, pas un journal reconstruit.

L'adaptateur testé fournit les uniformes issus des mantisses de 53 bits. Un passage par cet adaptateur ne remplace pas l'examen de toutes les projections du `raw64`. L'audit contient aussi des SmallCrush séparés sur des moitiés de 32 bits, mais aucun BigCrush exhaustif de toutes ces projections n'est revendiqué.

Philox et Threefry disposent eux aussi de résultats TestU01 publiés, suivant variantes et nombres de tours. Leur réussite n'est pas un avantage exclusif de Montmory ; voir [Random123 et ses résultats](https://github.com/DEShawResearch/random123).
