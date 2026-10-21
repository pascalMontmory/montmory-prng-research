# Montmory — lanes logiques et threads Metal

Ce kit mesure deux questions différentes avec les kernels Metal v3 inchangés. Il s'exécute dans son propre dossier et ne modifie pas les fichiers de TestU01.

**Série A : même flux.** La géométrie reste celle du TestU01 actuel : T=2048 lanes logiques, D=4096 tirages par lane, base=0. Le volume est donc 8 388 608 mots, soit 64 Mio. Seuls le nombre de threads GPU et la taille des groupes changent. Chaque mot produit doit rester identique à la référence C pour la même seed.

- Threads GPU : 1024, 2048, 8192, 32768, 65536, 131072, 262144.
- Threads par groupe : 64, 128, 256, 512, dans la limite du GPU.
- Moteurs : `montmory_mulhi_pair`, `threefry2x32_20`, `philox4x32_10`.
- Formats : raw64 et mantissa53, tous deux stockés dans un mot de 64 bits.

Un thread GPU est une unité de travail programmée, pas un cœur matériel. Plusieurs threads peuvent traiter les positions d'une même lane logique. Quand le nombre de threads est inférieur à celui des lanes, le kernel v3 utilise son chemin d'indexation général.

**Série B : autre géométrie logique.** Le volume reste identique et les threads sont fixés à 65536. Les lanes logiques prennent les valeurs 256, 512, 1024, 2048, 4096, 8192, 16384. Les tirages par lane deviennent D=8388608/T : T et D changent donc ensemble. Les quatre tailles de groupe sont comparées. Seul Montmory est concerné par cette série ; les références ne dépendent pas de cette géométrie.

Cette deuxième série produit d'autres flux. Les résultats TestU01 obtenus pour T=2048,D=4096 ne suffisent pas à les valider. Le fichier `logical-changes.csv` compte les mots différents. La configuration T=2048 de la série A sert aussi de référence à la série B.

Attendre la fin du TestU01 en cours, puis :

```bash
bash run.sh
```

Neuf répétitions sont effectuées par défaut. Chaque répétition utilise la même seed pour toutes les configurations, puis leur ordre est mélangé. Le tableau final affiche les meilleures médianes par nombre de threads ou par géométrie ; il ne certifie pas un optimum universel. Les mesures individuelles et toutes les tailles de groupe restent dans les CSV et `summary.json`.

Le programme vérifie les 1600 vecteurs ARM64/x86 et les contrôles natifs du kit v3, dont les six vecteurs officiels CPU/GPU de Random123. Pour ce nouveau balayage, toutes les configurations sont comparées intégralement au code C original ou à Random123 avant d'accepter leur première mesure, puis chaque sortie chronométrée est comparée intégralement hors chronométrage. Les gardes mémoire sont contrôlées.

Les temps GPU comprennent génération et écritures. Les temps hôte comprennent soumission et attente ; les références CPU et comparaisons complètes sont exclues des deux. Leur coût peut néanmoins chauffer la machine : l'état thermique est enregistré. Les mesures sur une machine occupée peuvent être bruitées ; le script refuse de démarrer lorsqu'il détecte le processus TestU01 Metal connu.

Les résultats sont archivés automatiquement, y compris les diagnostics de compilation en cas d'erreur. Rien n'est transmis automatiquement. Une compilation et une exécution sur le M3 restent nécessaires : les contrôles Linux vérifient les calculs entiers et l'attribution des sorties, pas le compilateur ou le matériel Apple.

Le protocole complet est dans `protocol.json`. Les sources et les licences Random123 sont conservées ; `SHA256SUMS` permet le contrôle d'intégrité. La provenance v3 est conservée pour les kernels réutilisés ; le nouveau manifeste identifie ce kit.
