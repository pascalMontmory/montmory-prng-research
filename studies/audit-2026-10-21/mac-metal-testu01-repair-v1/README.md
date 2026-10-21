# Lanceur TestU01 pour le kit Metal v3 — correction 1

21 octobre 2026. Extraire cette archive dans `montmory-metal-m3-v3`, puis :

```bash
bash testu01-repair-v1/run.sh montmory_mulhi_pair all
```

Le lancement initial `20261021T140817Z-9132` s'arrête à la configuration de
TestU01, avant les batteries : `C compiler cannot create executables`.
La détection `arm-apple-darwin24.6.0` a réussi. Le message précis de l'éditeur
de liens dans le `config.log` du Mac n'a pas encore été reçu.

Le premier lanceur utilisait directement le chemin Clang fourni par
`xcrun --find`, contrairement au benchmark qui passait par `xcrun clang`.
Ce complément sélectionne explicitement le SDK macOS et l'exporte via
`SDKROOT` pour les outils Apple. Il retire de son seul processus les options
de compilation héritées, puis compile et exécute un petit programme ARM64
avant TestU01. L'absence de SDK est une hypothèse à confirmer avec le journal
initial ; la réussite native de cette correction reste à vérifier.

Le code Montmory, ses kernels, le wrapper statistique et la source officielle
TestU01 1.2.3 du kit restent inchangés. Les manifestes originaux et celui de
ce complément sont vérifiés avant compilation. Une extraction et une installation
locales neuves sont créées sous `build/testu01-repair1-<date>-<pid>/`, sans
réutiliser une installation partielle. Aucun téléchargement, aucune installation
système et aucun changement de l'environnement du terminal ne sont effectués.

Le flux statistique reste celui du wrapper original : seed 42, 2 048 lanes
logiques, 4 096 sorties par lane, sortie uniforme 53 bits, nouvelle instance
pour chaque batterie. `all` lance SmallCrush, Crush et BigCrush, dans cet ordre.
Le lanceur exige le résumé de succès de chaque batterie avant de continuer.

En cas d'échec, l'étape et la fin du journal de construction sont affichées.
Une erreur de configuration affiche aussi le détail du premier essai de liaison.
Les journaux, le code de sortie et `config.log` sont archivés sous
`montmory-metal-v3-testu01-repair1-<date>-<pid>.tar.gz`. Le `config.log`
de la tentative initiale est conservé dans l'archive s'il existe encore.

Validation locale : syntaxe Bash et cinq scénarios d'orchestration avec des
outils Apple simulés sur Linux (succès, échec du programme C préalable, échec
de configuration, échec du wrapper, résultat statistique sans résumé de succès).
Ces scénarios vérifient l'arrêt, les messages et l'archivage ; ils ne constituent
ni une compilation native Apple ni un passage réel des batteries.

Référence sur la sélection du SDK : [code officiel Clang](https://clang.llvm.org/extra/doxygen/CompileCommands_8cpp_source.html),
qui explique le rôle du lanceur Apple dans l'initialisation de `SDKROOT`.
