# Reproduire les contrôles

Depuis le dossier de cet audit, avec Python 3.8+ et un compilateur C/C++ :

```bash
python3 tools/verify_exact.py
cc -std=c11 -O2 -Wall -Wextra -Werror -fopenmp \
  -Isnapshot/montmory_gpu_test/montmory_gpu_test/src \
  tools/partition_check.c -o build/partition_check
build/partition_check
cc -std=c11 -O2 -Wall -Wextra -Werror -fsanitize=undefined \
  -fno-sanitize-recover=all \
  -Isnapshot/montmory_gpu_test/montmory_gpu_test/src \
  tools/conformance_dump.c -o build/conformance_dump
build/conformance_dump > build/golden-recheck.txt
cmp evidence/golden-vectors.txt build/golden-recheck.txt
```

Le fichier de référence contient 1 600 lignes de mots hexadécimaux et mantisses :
SHA-256 `ca5b2731b89fea75c78585350c3263f11fc2695a6126251288b287ac94acdecf`.

## TestU01 sur le VPS

L'archive officielle téléchargée depuis le site des auteurs est conservée dans
`deps/TestU01.zip`, avec son URL et son SHA-256 dans `evidence/references.json`.
La bibliothèque a été configurée avec `--disable-shared`, compilée par `make -j2`
et installée dans `deps/testu01-install`, sans installation système.

```bash
cc -std=c11 -O3 -Wall -Wextra -Werror \
  -Isnapshot/montmory_gpu_test/montmory_gpu_test/src \
  -Ideps/testu01-install/include tools/testu01_driver.c \
  -Ldeps/testu01-install/lib -ltestu01 -lprobdist -lmylib -lm \
  -o build/testu01_driver
build/testu01_driver metal01 42 2048 4096 small
```

L'adaptateur `metal01` reconstitue sur CPU l'ordre des sorties du kernel Metal,
sans modifier le mélange du C récupéré. Les modes `cpu01`, `low32` et `high32`
testent la lane zéro, respectivement en uniformes 53 bits, mots bas et mots hauts
de 32 bits. Les batteries acceptées sont `small`, `crush`, `big`. Seule `small`
a été relancée dans cet audit. Un code de sortie système nul ne suffit pas à
conclure au succès statistique : lire aussi le résumé final TestU01.

## Contrôle natif sur le Mac

Pour reproduire le premier comparatif des débits sur Apple Silicon, utiliser le kit
`mac-metal-benchmark-kit.tar.gz` : extraire l'archive, entrer dans le dossier
`montmory-metal-m3`, puis lancer `bash run.sh`. Il compile le C et le programme
de test en ARM64, compile Metal à l'exécution, contrôle les sorties puis mesure
Montmory, Threefry2x32-20 et Philox4x32-10. Les sorties CPU, GPU et statistiques
de débit sont conservées dans une archive de résultats. Voir
[mac-metal-benchmark/README.md](mac-metal-benchmark/README.md) pour le protocole.
La préparation a été vérifiée sur Linux. Le premier retour terminal utilisateur
sur Apple M3 (`20261021T124458Z-8820`) confirme la compilation Objective-C++/Metal,
la conformité et les 11 répétitions terminées. Les fichiers de mesures brutes
restent à recevoir sur le VPS ; voir la section M3 du rapport principal.

Pour tester les nouvelles implémentations exactes de Montmory, extraire
`mac-metal-benchmark-v2-kit.tar.gz`, entrer dans `montmory-metal-m3-v2`, puis
lancer `bash run.sh`. Cette archive distincte conserve les moteurs précédents
et ajoute trois candidats ainsi qu'un témoin d'écriture non aléatoire.
Le [protocole v2](mac-metal-benchmark-v2/README.md) et ses
[identités arithmétiques](mac-metal-benchmark-v2/EXACT.md) précisent la portée.
Le retour terminal utilisateur `20261021T131521Z-8925` confirme maintenant la
compilation et l'exécution natives sur Apple M3 : 4 153 344 mots comparés,
six vecteurs officiels CPU/GPU PASS et 11 répétitions terminées. Les données
brutes restent à recevoir. Les deux kits livrés sont figés ; leurs documents
internes reflètent l'état de préparation, les résultats figurent dans le
rapport principal et dans `evidence/metal-m3-v2-transcript-summary.json`.
Pour reproduire la préparation et les contrôles sur Linux :

```bash
cd mac-metal-benchmark-v2
python3 tools/prepare_metal.py
python3 tools/verify_emulation.py
bash -n run.sh
```

Le kit v3 poursuit la recherche : extraire `mac-metal-benchmark-v3-kit.tar.gz`,
entrer dans `montmory-metal-m3-v3`, puis lancer `bash run.sh`. Il conserve les
références précédentes et ajoute les variantes par lots et rotations spécialisées.
Le [protocole v3](mac-metal-benchmark-v3/README.md) précise les coûts de
préparation, les contrôles et les limites du diagnostic par réduction.
Le retour `20261021T134922Z-9053` annonce la compilation et les contrôles
natifs du benchmark réussis ; les résultats sont consignés dans le rapport
principal et dans `evidence/metal-m3-v3-transcript-summary.json`.
Les batteries utilisent un wrapper distinct dont la compilation et l'exécution
natives sont maintenant rapportées comme réussies avec le complément de
lancement. SmallCrush et Crush passent pour le candidat `montmory_mulhi_pair` ;
le résultat BigCrush reste en attente.

La première tentative du lanceur initial a échoué pendant la configuration
de TestU01. Le complément `mac-testu01-repair-v1.tar.gz` sélectionne le SDK
macOS explicitement et affiche les erreurs. Pour reproduire le lancement,
l'extraire dans le dossier v3 ; laisser une campagne déjà en cours se terminer :

```bash
# Toutes les batteries, en commençant par SmallCrush.
bash testu01-repair-v1/run.sh montmory_mulhi_pair all
# Ou seulement SmallCrush pour une première vérification.
# bash testu01-repair-v1/run.sh montmory_mulhi_pair small
# Profilage séparé, sans fusionner ces temps avec les mesures ordinaires.
# bash capture.sh montmory_mulhi_pair
```

Le lanceur TestU01 compile sa bibliothèque fournie sous `build/` et utilise
le flux du SmallCrush historique (2 048 × 4 096) pour toutes les batteries,
indépendant de la géométrie de benchmark. Les anciens journaux CUDA utilisent
4 096 × 8 192 pour Crush et 8 192 × 16 384 pour BigCrush : ce lancement ne
reproduit pas leur paramétrage et n'a pas à retrouver leurs p-values.
`all` comprend Crush et BigCrush, potentiellement longs. Le
lanceur exige le résumé de succès de chaque batterie et archive les journaux
même en cas d'échec. Le retour natif SmallCrush de `mulhi_pair` annonce
15 statistiques réussies et retrouve les quinze p-values imprimées historiques.
Les extraits sont dans `evidence/metal-m3-v3-native-smallcrush-user-extract.log` ;
Crush annonce ensuite 144 statistiques réussies, dans
`evidence/metal-m3-v3-native-crush-user-summary.log`. L'archive brute et
le résultat BigCrush restent attendus.
L'adaptation CPU de `spec_dual` a passé SmallCrush :

```bash
cd mac-metal-benchmark-v3
python3 tools/prepare_metal.py
python3 tools/verify_emulation.py
python3 tools/run_cpu_small.py ../deps/testu01-install
```

Pour le contrôle de conformité seul avec le compilateur Metal de Xcode :

Copier l'archive `mac-conformance-kit.tar.gz` depuis le VPS, l'extraire dans un
dossier de travail, puis lancer, depuis le dossier extrait :

```bash
bash tools/check_macos.sh
```

Le script nécessite les outils de développement Xcode avec le compilateur Metal.
Il compare d'abord exactement les vecteurs du CPU Mac avec ceux du VPS x86,
puis compile le kernel original et compare 203 760 mantisses produites par le
GPU à celles du C natif. Les résultats sont écrits sous `evidence/macos/`.
Ce contrôle reste **à exécuter sur le Mac** ; sa préparation sur Linux ne vaut
pas validation du compilateur Metal, du matériel ni du wrapper historique complet.

## CUDA

La sauvegarde OVH est figée dans `snapshot/cuda-ovh/`, originaux conservés.
Le contrôle suivant compile les corps des kernels et les fonctions des anciens
benchmarks en C++ sur CPU, avec une adaptation des qualificatifs CUDA et des
indices de threads. Il n'utilise pas de GPU :

```bash
python3 tools/verify_cuda.py
```

Il compare 251 904 mantisses CUDA/Metal/référence, vérifie les journaux, reproduit
le défaut de borne du kernel et l'écart du Philox des benchmarks au standard.
Il vérifie aussi la préservation des sorties par la correction du kernel et
l'absence d'écriture dans les zones sentinelles. Résultats dans `evidence/cuda-exact.json`.
Les headers officiels Random123 figés dans `deps/random123/` sont nécessaires
pour comparer le Philox des benchmarks à la référence.

La copie corrigée est dans `reviewed-source/cuda/`. Les fonctions du cœur C et
les opérations mathématiques du kernel sont conservées. La borne, la structure
des paramètres partagée et certains contrôles d'allocation sont corrigés.
**Le wrapper corrigé a été relu, mais nvcc et le GPU manquent sur ce VPS.**

Sur une machine équipée de CUDA, depuis le dossier de l'audit :

```bash
nvcc -O3 -std=c++17 tools/check_cuda.cu -o build/check_cuda
build/check_cuda
```

Ce programme préparé compare le kernel corrigé au C original pour quatre seeds,
cinq géométries logiques, trois bases dont une proche de 2^64, et quatre tailles
de bloc CUDA, en vérifiant aussi les sentinelles. Il affiche le nom du GPU.
Il reste à compiler/exécuter sur le matériel ; aucun résultat natif n'est présumé.
Avec TestU01 installé, le Makefile de `reviewed-source/cuda/` permet ensuite de
recompiler le wrapper et de relancer les batteries avec des chemins `TESTU_INC`
et `TESTU_LIB` adaptés. Les anciens benchmarks GPU ne doivent pas servir de
référence Philox en l'état.

## Comparaisons CPU reproductibles

Les fichiers officiels Random123 sont figés au commit
`9545ff6413f258be2f04c1d319d99aaef7521150` ; leurs empreintes Git et SHA-256 sont
dans `evidence/random123-upstream-manifest.json`. Leur licence est conservée.
Le code exact des mesures en volume est `evidence/benchmark-bulk-source.cpp` ;
celui des accès isolés est `evidence/benchmark-sparse-source.cpp`.
`tools/benchmark.cpp` fournit aussi les deux modes dans une seule version.

Pour reproduire la campagne en volume avec le source exact :

```bash
c++ -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Werror \
  -Ideps/random123/include -Isnapshot/montmory_gpu_test/montmory_gpu_test/src \
  evidence/benchmark-bulk-source.cpp -o build/benchmark-bulk-replay
OMP_PROC_BIND=true OMP_PLACES=cores \
  build/benchmark-bulk-replay 8388608 11 > build/benchmark-bulk-replay.csv
```

Pour les accès isolés, compiler `evidence/benchmark-sparse-source.cpp` avec les
mêmes options, puis ajouter `sparse` comme troisième argument du programme.
Répéter sans `-march=native` donne la comparaison avec les options génériques.
Une machine partagée produit du bruit ; préserver plusieurs répétitions et
examiner toutes les références, pas seulement la plus lente.

Chaque exécution vérifie 24 vecteurs officiels et 262 336 identités Montmory
avant les mesures. Le benchmark en volume consomme tous les mots des blocs
Random123. Le mode isolé sélectionne seulement le mot demandé par chaque indice.
Les seeds, temps, nombres de workers observés et checksums sont dans les CSV.
Les métadonnées, sources exacts et résultats du pilote antérieur restent séparés.

Pour vérifier et résumer les quatre campagnes conservées :

```bash
python3 tools/summarize_benchmarks.py
```

Le script vérifie aussi l'identité des checksums entre modes, nombres de workers
et compilations. Il ne transforme pas ces checksums en preuve exhaustive
d'équivalence des architectures, et les mesures CPU ne prédisent pas les débits GPU.
