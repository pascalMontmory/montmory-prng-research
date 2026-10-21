# Reproduire les vérifications

[Français](reproduce.md) · [English](en/reproduce.md) · [Documentation FR/EN](languages.md)

[Accueil](../README.fr.md) · [Méthodologie](methodology.md)

Toutes les commandes partent de la racine du dépôt, sauf indication contraire. Les vérifications locales ci-dessous ne nécessitent ni identifiants cloud ni matériel TPU.

## 1. Intégrité et référence exacte

Prérequis : Python 3.11 ou plus récent, compilateur C avec UBSan (`cc`/Clang/GCC).

```bash
python3 scripts/check_repository.py
python3 scripts/check_reference.py
```

La première commande contrôle les empreintes des 884 fichiers importés, les liens locaux de la documentation principale, les manifestes JSON des kits et les effectifs des résultats. La seconde compile le C inchangé avec UBSan, le compare aux 1 600 vecteurs, compare le modèle Python et recalcule les certificats exacts du mélangeur et des lanes.

Les sorties nouvelles vont dans `build/`, ignoré par Git. Aucune batterie statistique n'est relancée.

## 2. Recalculer l'audit TPU sur CPU

Prérequis supplémentaires : NumPy. L'audit ne nécessite pas JAX ni `libtpu`.

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install numpy==2.4.6
.venv-audit/bin/python scripts/replay_audit.py tpu
```

Le script travaille dans une copie temporaire, recompile l'oracle C depuis les sources, régénère les références Montmory et vérifie les 3 708 lignes, l'ordre des mesures, la sélection et les 44 synthèses. Il n'exécute aucune bibliothèque compilée reçue du TPU. Un répertoire daté sous `build/` conserve le journal et l'analyse recalculée.

Un audit de données réussi n'est pas une répétition native du benchmark. Le chemin temporaire et certaines métadonnées du rapport peuvent différer ; les contrôles et valeurs scientifiques doivent concorder.

## 3. Recalculer les analyses Metal et TACM

L'analyse Metal nécessite matplotlib en plus de NumPy. Elle porte sur **la première archive M3 reçue**, pas sur les confirmations disponibles seulement par terminal :

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install numpy==1.24.2 scipy==1.10.1 pandas==1.5.3 matplotlib==3.6.3
.venv-analysis/bin/python scripts/replay_audit.py metal
.venv-analysis/bin/python scripts/replay_audit.py tacm
```

Pour TACM, un compilateur C++17 (`g++`) est aussi requis. Le script reconstruit les comparateurs et leurs contrôles, recalcule les conclusions depuis les estimations archivées et rejoue indépendamment 180 estimations. Il ne relance pas le milliard de sorties de toute la campagne et n'ajuste pas les seuils.

Les versions TACM et JAX utilisent des environnements séparés : ne pas fusionner leurs anciens fichiers de dépendances dans un même environnement.

## 4. Vérifier le port JAX localement

Cette étape est facultative. Elle exécute réellement le port sur le **CPU disponible**, sans mesurer un TPU :

```bash
python3 -m venv .venv-jax
.venv-jax/bin/python -m pip install -r studies/tpu-opt-2026-10-21/montmory-tpu-opt/requirements.txt
JAX_PLATFORMS=cpu .venv-jax/bin/python studies/tpu-opt-2026-10-21/montmory-tpu-opt/verify_cpu.py --out build/jax-cpu-new
JAX_PLATFORMS=cpu .venv-jax/bin/python studies/tpu-opt-2026-10-21/montmory-tpu-opt/verify_candidates.py --backend cpu --out build/jax-candidates-new
```

Sur Mac, les outils de ligne de commande Xcode doivent être présents. Une exécution Pallas interprétée sur CPU vérifie des valeurs, pas une compilation ou une performance native TPU. Utiliser un nouveau chemin de sortie à chaque campagne conservée.

Cet environnement permet aussi de rejouer l'audit de la première campagne TPU :

```bash
.venv-jax/bin/python scripts/replay_audit.py tpu-initial
```

## 5. Relancer la campagne native TPU

Télécharger le [notebook autonome d'optimisation](../notebooks/Montmory_TPU_optimisation.ipynb), l'importer dans Colab, sélectionner un environnement **TPU**, puis exécuter les cellules dans l'ordre. Le notebook embarque le kit figé ; aucune authentification GitHub n'est nécessaire après le téléchargement du fichier.

Le journal doit annoncer un backend `tpu` et un appareil réel avant les mesures. `No jellyfish device found` signifie que l'initialisation TPU a échoué : ce n'est pas un résultat de performance. Le temps d'allocation Colab et les compilations ne sont pas inclus dans les temps de benchmark.

Le notebook utilise les versions figées, vérifie les sources, lance conformité/recherche/confirmation et fournit l'archive des résultats. Conserver la nouvelle archive avec son identifiant ; ne pas remplacer celle du 21 octobre. L'accès à un environnement TPU dépend du service et n'est pas assuré par ce dépôt.

Dans une VM TPU déjà configurée, il est aussi possible d'utiliser une copie du kit :

```bash
mkdir -p build
cp -R studies/tpu-opt-2026-10-21/montmory-tpu-opt build/tpu-new
.venv-jax/bin/python -m pip install -r build/tpu-new/requirements-tpu.txt
.venv-jax/bin/python build/tpu-new/run.py
```

`run.py` exige le TPU par défaut. L'option `--cpu-smoke` teste seulement l'orchestration sur CPU et ne doit jamais être étiquetée « résultat TPU ».

## 6. Relancer Metal et TestU01 sur un Mac Apple Silicon

Pour la confirmation Metal, copier le kit hors des archives figées :

```bash
mkdir -p build
cp -R studies/metal-opt-2026-10-21/montmory-metal-confirm-m3 build/metal-confirm-new
bash build/metal-confirm-new/run.sh
```

Pour refaire SmallCrush avec le lanceur TestU01 réparé :

```bash
mkdir -p build
cp -R studies/audit-2026-10-21/mac-metal-benchmark-v3 build/metal-testu01-new
cp -R studies/audit-2026-10-21/mac-metal-testu01-repair-v1 build/metal-testu01-new/testu01-repair-v1
bash build/metal-testu01-new/testu01-repair-v1/run.sh montmory_mulhi_pair small
```

Les arguments `crush`, `big` ou `all` lancent les autres batteries et peuvent immobiliser la machine longtemps. Chaque batterie repart de la seed 42, avec `T=2048,D=4096`. Le kit contient la source TestU01 1.2.3 et sa notice ; l'installation est locale au dossier de travail.

Ces commandes sont documentées à partir des kits conservés. La publication du dépôt n'a pas relancé de campagne native Metal ou TPU. Les autres scripts historiques peuvent contenir des chemins de l'ancien VPS ou dépendre de binaires volontairement exclus ; ils ne sont pas tous des points d'entrée portables de ce dépôt.

## 7. Conserver une nouvelle preuve

Conserver ensemble : commit, manifeste des sources, versions des dépendances et du compilateur, matériel, géométrie logique, seeds, indices, format, protocole, mesures individuelles, empreintes de sortie, code de retour et analyse. Identifier séparément une nouvelle campagne, une simple réanalyse et une transcription terminal. Le [workflow CI](../.github/workflows/validate.yml) exécute uniquement des contrôles CPU du dossier de recherche.

Le [rapport local historique](../results/repository-validation.json) consigne les contrôles exécutés lors de l'assemblage du dossier de recherche. Il ne constitue pas une preuve de réussite de la CI hébergée pour cette édition publique.

## 8. GitHub Actions public

La [page Actions](https://github.com/pascalMontmory/montmory-prng-research/actions) consigne les contrôles du dépôt public. Une exécution qui ne démarre pas n'est pas un résultat de test. La validation locale, les campagnes natives archivées et la CI hébergée sont des catégories de preuve distinctes.

Pour les éditions bilingues, lancer aussi `python3 scripts/check_translations.py`. Le [guide des langues](languages.md) décrit le périmètre traduit et les pièces conservées intactes.
