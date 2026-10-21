# Montmory Metal M3 — kit v3 d'optimisation et de diagnostic

Ce kit conserve les références des deux premiers essais et ajoute cinq
candidats. **Aucun gain M3 de ces nouvelles variantes n'est encore mesuré.**
Le kernel v2 `montmory_u32_tiled`, déjà mesuré autour de 63 GB/s sur le M3 de
l'utilisateur, reste présent. Les archives v1 et v2 sont conservées à l'identique.
Les constantes, les tours, le mélange et l'ordre logique des sorties ne changent
pas. Les identités sont détaillées dans [EXACT.md](EXACT.md).

## Lancement habituel sur le Mac

```bash
bash run.sh
```

Les outils Apple en ligne de commande suffisent au benchmark habituel, comme
pour les kits précédents. La compilation Metal se fait par l'API à l'exécution.
Ni Python ni Homebrew ne sont nécessaires sur le Mac. Brancher le Mac et fermer
les applications sollicitant le GPU facilite les comparaisons.

Par défaut : 4 194 304 mots de 64 bits, 11 seeds, 114 configurations par seed.
La compilation des variantes spécialisées peut prolonger le lancement ; sa
durée est enregistrée séparément. Deux échauffements sont effectués pour
chaque configuration et chaque seed avant les mesures mélangées. Ce protocole
diffère des kits précédents : comparer d'abord les moteurs d'une même campagne.

Le tableau final donne le meilleur réglage observé par moteur et format.
Il sélectionne parmi les médianes mesurées ; il ne prouve pas un optimum général.
Toutes les configurations, répétitions, seeds, temps GPU/muraux, tailles de
groupes, sommes et XOR sont conservés dans `measurements.csv`.

## Nouvelles variantes

| Nom | Changement |
|---|---|
| `montmory_pair` | Deux sorties adjacentes par thread et par lot, trois tours intercalés entre les états indépendants |
| `montmory_quad` | Quatre sorties par lot ; davantage de calculs indépendants, mais davantage de variables à conserver |
| `montmory_mulhi_pair` | Deux sorties par lot, produit haut explicite pour calculer les retenues de la fonction de tour |
| `montmory_spec_pair` | Deux sorties par lot, fonction de tour 64 bits originale, mélange 32 bits ; rotations spécialisées |
| `montmory_spec_dual` | Deux sorties par lot, expressions paire/impaire séparées de la fonction de tour ; rotations spécialisées |

La spécialisation porte seulement sur les trois rotations dérivées de la seed.
Les clés restent des paramètres ordinaires. Elle peut être réutilisée lorsque
le même triplet de rotations revient. `pipeline-preparation.json` enregistre
chaque coût de création ; `metadata.json` enregistre aussi la compilation de
la bibliothèque. **Ces coûts ne sont pas inclus dans le débit GPU affiché**.
Un éventuel gain de `spec_*` devra donc être évalué selon le nombre de sorties
produites entre les changements de seed, et en tenant compte du cache.

Cette technique utilise les
[constantes de fonction Metal](https://developer.apple.com/documentation/metal/mtlfunctionconstantvalues).
Elle ne change pas les rotations du générateur pour en choisir de plus rapides.

## Géométrie et mesure

Les mesures conservent 8 192 lanes logiques et D = N / 8192 sorties par lane.
Les threads physiques sont 8 192, 65 536 et 262 144. Les groupes de 256 sont
testés pour tous les moteurs ; les groupes de 64 et 128 sont aussi testés à
65 536 threads pour les nouveaux candidats, le meilleur v2, Philox et Threefry.
La taille effectivement utilisée, bornée par la capacité du pipeline, est
enregistrée. Changer N change D et donc généralement le flux Montmory ; dans
chaque campagne, les références partagent exactement cette géométrie logique.

Chaque sortie brute ou mantisse occupe huit octets. Le temps GPU inclut le
calcul et les écritures, exclut la préparation CPU, les compilations, les
vérifications et checksums. Le débit GB/s est décimal. Tous les mots des blocs
Philox sont consommés. Le témoin `write_baseline_NOT_PRNG` est non aléatoire et
ne représente pas une limite théorique de bande passante du matériel.

Ne pas comparer les valeurs en Gmots/s du diagnostic aux GB/s du tableau principal.
Dans le diagnostic, le programme calcule les mêmes sorties puis leur XOR par
thread : il écrit 65 536 mots au lieu de N. Les références et candidats utilisent
le même principe, et les réductions sont contrôlées. **Ce n'est pas un temps de
calcul pur** : l'accumulation, les registres et l'organisation des écritures
changent. On ne peut pas soustraire les deux temps pour obtenir un coût mémoire
exact, ni présenter ce débit comme celui d'une livraison du flux complet.

Pour examiner d'autres volumes après le premier essai :

```bash
bash run.sh 1048576 11
bash run.sh 16777216 11
```

## Vérifications

Avant les mesures natives, le programme vérifie 1 600 vecteurs ARM/x86, six
vecteurs officiels Random123 sur CPU/GPU, puis **4 835 328 mots complets et
4 004 592 réductions par thread**. Cela inclut plusieurs seeds et bases proches
du débordement, des répartitions non régulières, des lanes ne contenant qu'un
ou deux mots, des threads inactifs et des sentinelles mémoire. Une erreur
interrompt le programme.

Pendant les mesures, les sommes complètes doivent coïncider entre toutes les
variantes Montmory d'un même format ; le XOR global doit aussi coïncider entre
le flux complet et les diagnostics. Des positions individuelles sont contrôlées
dans chaque sortie complète. Ces checksums ne sont pas une preuve exhaustive.

Préparation sur VPS : **199 824 000 mots complets**, **73 967 040 réductions** et
**9 713 754 comparaisons arithmétiques** concordent avec UBSan. Les rotations
spécialisées sont représentées par des variables dans cette adaptation C++ :
elle ne teste pas le compilateur de spécialisation Apple. L'Objective-C++ et
le Metal natifs v3 restent à compiler et exécuter sur le Mac.

Un nouveau **SmallCrush réussit sur l'adaptation CPU** de `spec_dual`, avec
seed 42, 2 048 lanes logiques et 4 096 sorties par lane. Les quinze p-values
imprimées sont identiques à celles du replay historique. Ce n'est pas un
SmallCrush exécuté sur le GPU ; Crush et BigCrush n'ont pas été relancés sur v3.
Les preuves correspondantes sont dans `generated/`.

## Batteries TestU01 sur le véritable GPU

Après avoir choisi un moteur à partir des mesures, le lanceur séparé permet
de tester son binaire natif. Exemple avec le meilleur moteur précédemment mesuré :

```bash
bash run_testu01.sh montmory_u32_tiled small
bash run_testu01.sh montmory_u32_tiled all
```

Remplacer le nom par celui d'un candidat pour le tester. `all` lance SmallCrush,
Crush puis BigCrush, en repartant de la seed 42 à chaque batterie. Les deux
dernières batteries peuvent durer longtemps ; elles ne démarrent pas lors de
`run.sh`. Les paramètres statistiques historiques sont fixés à 2 048 lanes
logiques et 4 096 sorties par lane, indépendamment des threads physiques.
Cette géométrie diffère volontairement de celle des mesures de débit.

L'archive officielle TestU01 1.2.3, identique à celle figée dans l'audit, est
fournie dans `vendor/TestU01.zip`. Le script compile la bibliothèque localement
sous `build/`, en ARM64, sans installation système ni téléchargement. Les
contrôles natifs précèdent la batterie, et le premier bloc de 8 388 608 sorties
est comparé intégralement à la référence CPU. Le programme conserve les logs
et le script exige le résumé `All tests were passed` ; un code de retour nul
du programme ne suffit pas à annoncer un succès statistique.
Ce lanceur natif est préparé mais n'a pas encore été exécuté sur le Mac.

## Profil GPU facultatif

```bash
bash capture.sh montmory_spec_dual
```

Sur macOS 14 ou plus récent, le script active `MTL_CAPTURE_ENABLED` pour ce
processus et enregistre une commande dans `profile.gputrace`, avant les mesures.
Il s'arrête explicitement si la capture n'est pas prise en charge. Le fichier
peut être ouvert dans le débogueur Metal de Xcode pour examiner les coûts des
instructions et l'utilisation des ressources. Cette campagne est marquée
`profiling_run` ; ses temps ne doivent pas être mélangés avec les essais ordinaires.
La capture est facultative et n'est pas nécessaire au benchmark habituel.
Voir la [procédure Apple](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically).

## Résultats et provenance

Les archives `montmory-metal-v3-results-*.tar.gz` et
`montmory-metal-v3-testu01-*.tar.gz` contiennent les résultats et diagnostics,
y compris en cas d'échec. Aucun résultat n'est envoyé automatiquement.
`SHA256SUMS` et `generated/provenance.json` figent les sources. Le cœur C,
le kernel Metal historique et les headers officiels Random123 restent inchangés.
`run.sh` compile `src/benchmark_v3.mm` ; les kernels des versions précédentes
restent inclus pour comparaison dans la même campagne.

Sur Linux, avec Python 3 et un compilateur C++, la préparation se reproduit par :

```bash
python3 tools/prepare_metal.py
python3 tools/verify_emulation.py
python3 tools/run_cpu_small.py /chemin/vers/testu01-install
```
