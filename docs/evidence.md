# Registre des preuves et traçabilité

[Français](evidence.md) · [English](en/evidence.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Index machine](../evidence/index.json) · [Empreintes importées](../evidence/import-manifest.json)

## Niveaux de preuve

| Niveau | Définition |
|---|---|
| **Données brutes auditées** | CSV/JSON ou journaux reçus, contrôles de structure et recalcul des conclusions avec les sources figées |
| **Journal historique** | Journal complet disponible, mais exécution historique non refaite dans la session actuelle |
| **Résumé terminal transmis** | Valeurs recopiées depuis le terminal de l'utilisateur ; absence de données individuelles pour un recalcul indépendant |
| **Exécution CPU / émulation** | Vérification locale de valeurs, d'un protocole ou d'une transcription ; ce n'est pas une exécution native GPU/TPU |
| **Identité / certificat exact** | Calcul entier démontrant une propriété ou un contre-exemple précis ; sa portée ne dépasse pas cette propriété |

« Audité » désigne ici une relecture et un recalcul par un script distinct dans le même projet. Il ne signifie pas certification indépendante par un organisme tiers.

## Où trouver chaque résultat

| Sujet | Dossier principal | Ce qui est réellement présent |
|---|---|---|
| Moteur et certificats exacts | [audit](../studies/audit-2026-10-21/MATHEMATIQUES.md) | Sources, certificats, vecteurs, identités et contre-exemples |
| TestU01 | [registre dédié](testu01.md) | Journaux historiques complets, reprises CPU, extraits récents séparés |
| Comparaisons CPU Random123 | [données](../studies/audit-2026-10-21/evidence/benchmark-native.csv) | Mesures, compilation, vecteurs connus et implémentations figées |
| TACM corrigé | [étude](../studies/tacm-comparison-2026-10-21/README.md) | Protocole, pilote de calibration, extension, estimations brutes et analyse |
| Workers et lanes CPU | [étude](../studies/lanes-study-2026-10-21/README.md) | Mesures locales x86 et contrôles de conservation du flux |
| Optimisation M3 initiale | [archive extraite](../studies/metal-opt-2026-10-21/received) | Mesures natives et audit ; effets de première utilisation identifiés |
| Confirmation M3 séparée | [résumé](../studies/metal-opt-2026-10-21/results/mac-confirm-reported-20261021T163954Z-22644.json) | Terminal transmis ; archive brute manquante |
| SIMD et pool de workers | [étude](../studies/vector-opt-2026-10-21/README.md) | Données x86 locales ; résultats ARM64/Metal rapportés par terminal |
| Port JAX CPU | [étude](../studies/jax-port-2026-10-21/README.md) | Port entier portable et vérifications CPU |
| Première campagne TPU | [données](../studies/jax-tpu-2026-10-21/20261021T180635Z-colab-tpu) | 770 observations, rapports natifs, audit CPU |
| Optimisation TPU finale | [données](../studies/tpu-opt-2026-10-21/20261021T185102Z-tpu-opt) | 3 708 observations, 104 compilations, sélection figée, 44 synthèses recalculées |
| Dropout TPU du 24 octobre | [résultats et audit](../applications/dropout/validation/tpu-20261024/README.fr.md) | 1 800 mesures ; 900 paires de pertes JAX/Pallas identiques ; archive et kit exact conservés |

## Matrice de conformité native

| Plateforme | Vérifications documentées | Limite |
|---|---|---|
| x86 CPU | C de référence, 1 600 vecteurs, certificats, frontières, découpages, comparaisons et reconstitution des empreintes TPU | Résultats des machines et compilateurs enregistrés |
| ARM64 Apple M3 | 1 600 vecteurs ARM64/x86 et variantes équivalentes ; première archive d'optimisation reçue | Certaines confirmations ultérieures ne sont disponibles que par terminal |
| GPU Metal M3 | Millions de comparaisons natives avec référence CPU ; vérification des sorties chronométrées | Modèle M3 testé ; aucune généralisation automatique à tous les Apple Silicon |
| GPU CUDA historique | Sources et journaux récupérés, transcriptions vérifiées sur CPU | Version corrigée non exécutée sur un GPU NVIDIA dans l'audit récent |
| TPU v5 lite | 1 972 720 comparaisons historiques, 1 684 047 comparaisons de candidats incluant Threefry, empreintes des sorties chronométrées | Un périphérique natif ; aucune preuve native multi-TPU |

Cette matrice ne prétend pas que chaque variante nouvellement écrite a été exécutée nativement sur toutes les architectures. La référence commune relie les campagnes ; les portages et plateformes réellement testés restent identifiés.

## Conservation

Les **884 fichiers importés** sont copiés sans modification et recensés par taille, origine relative et SHA256 dans `import-manifest.json`. Les fichiers de documentation et les scripts de ce dépôt sont distincts de ces imports. Les anciennes archives contenant des fichiers supplémentaires ne sont pas reconstituées artificiellement.

Les caches, environnements virtuels, bibliothèques compilées reçues, exécutables et archives redondantes sont exclus. Les sources nécessaires aux audits actuels sont présentes. Le [registre des exclusions](../evidence/excluded-files.json) conserve la règle et les chemins concernés ; il ne prétend pas fournir une copie intégrale de chaque ancien bundle.

Les manifestes de certains bundles historiques mentionnent donc des binaires absents. Les points d'entrée de `scripts/` sont les vérifications prises en charge pour ce dépôt. Les kits TPU figés, en revanche, conservent intégralement leurs fichiers de manifeste.

L'[archive M3 initiale reçue](../evidence/archives/montmory-metal-opt-results-20261021T162407Z-22542.tar.gz) est aussi conservée : son empreinte est requise par l'audit historique. Les audits ne chargent aucun binaire reçu ; les oracles nécessaires sont recompilés depuis les sources.

## Pièces encore manquantes

L'archive intégrale du dernier BigCrush natif M3 v3, les archives natives de confirmation Metal séparée et de réparation vectorielle, ainsi qu’une répétition indépendante de la campagne TPU de débits bruts ne sont pas présentes. La session dropout du 24 octobre mesure une autre charge. Les résumés correspondants restent explicitement marqués comme tels. Ces manques n'empêchent pas de refaire les audits des campagnes dont les mesures brutes sont disponibles.
