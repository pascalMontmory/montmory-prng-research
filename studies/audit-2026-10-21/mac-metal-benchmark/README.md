# Comparatif GPU Metal pour le Mac Apple Silicon

Depuis ce dossier, dans le Terminal du Mac :

```bash
bash run.sh
```

Les outils de compilation Apple doivent être installés. S'ils manquent, lancer
`xcode-select --install`, terminer l'installation puis relancer. Le kit compile
le programme en ARM64 et compile le Metal au moment de l'exécution avec l'API
Apple. Il n'appelle pas le compilateur Metal de Xcode en ligne de commande.
Python et Homebrew ne sont pas nécessaires pour lancer le test.

Brancher le Mac et fermer les applications qui sollicitent fortement le GPU
permet de limiter les variations. Les états thermiques initial et final sont
enregistrés. Aucune configuration du Mac n'est modifiée.

## Contrôles avant les mesures

- Les 1 600 vecteurs du cœur C compilé en ARM64 sont comparés aux vecteurs x86
  conservés dans l'audit.
- Les six vecteurs officiels Threefry2x32-20 et Philox4x32-10 sont vérifiés sur
  le CPU puis sur le GPU.
- 909 312 mots de sortie sont comparés exactement à la référence CPU, avec
  quatre seeds, des bases proches du débordement de 64 bits, plusieurs nombres
  de threads et tailles de groupes. Des sentinelles détectent les écritures
  dépassant la zone demandée.
- Chaque mesure vérifie ensuite 160 positions de sortie et un checksum complet.
  Le checksum doit rester identique sous les deux répartitions physiques ; il
  doit aussi correspondre au kernel historique pour le flux Montmory 53 bits.

Une erreur arrête le programme et interdit de présenter les mesures comme valides.

## Mesures réalisées

Par défaut : 4 194 304 sorties de 64 bits, soit 32 Mio par lancement ; 11 seeds,
deux échauffements par configuration, ordre des moteurs mélangé entre répétitions,
et 8 192 puis 65 536 threads physiques.

| Moteur | Code et format |
|---|---|
| `montmory_indexed` | Mathématiques du cœur reçu, clés précalculées, écritures voisines affectées aux threads voisins |
| `threefry2x32_20` | Header officiel Random123, 20 tours, deux mots de 32 bits réunis en un mot de 64 bits |
| `philox4x32_10` | Header officiel Random123, 10 tours, les deux mots de 64 bits de chaque bloc sont consommés |
| `montmory_historical` | Kernel Metal original inchangé, pour comparer l'implémentation historique en mantisses 53 bits |

Les trois premiers moteurs produisent soit des mots bruts de 64 bits (`raw64`),
soit des entiers représentant les mantisses de 53 bits (`mantissa53`). Dans les
deux cas, huit octets sont écrits par sortie. Le kernel historique n'offre que
le second format. Aucun flottant 64 bits n'est calculé sur le GPU.

Ce test mesure **la génération en volume**. Les résultats du tableau CPU portant
sur des demandes à des indices aléatoires isolés correspondent à un autre usage.
Threefry4x64 n'est pas inclus : la configuration Metal officielle examinée
désactive cette variante. Voir le rapport d'antériorité de l'audit.

Pour Montmory, la géométrie **logique** reste fixée à 8 192 lanes et
`draws = nombre_de_sorties / 8192`, indépendamment des threads physiques.
L'affectation des sorties aux threads change, le flux historique est préservé.
L'initialisation des clés des trois nouveaux kernels est effectuée avant le
chronométrage ; le kernel historique effectue son initialisation dans le GPU.
Un écart entre ces deux versions ne s'attribue donc pas uniquement aux écritures.

Le temps GPU provient de `GPUStartTime`/`GPUEndTime` après la fin de la commande.
Il comprend le kernel et ses écritures en mémoire partagée. Compilation,
initialisation CPU, validation et checksum sont hors de ce temps. Le temps mural
incluant soumission et attente est également conservé dans le CSV.
Les unités GB/s sont décimales ; ns/mot est un temps amorti par le parallélisme,
pas la latence d'une requête individuelle.

## Résultats et variantes de lancement

Le tableau final donne les médianes GPU : une valeur ns/mot plus petite signifie
un débit plus élevé. Les CSV, résumés JSON, compilateur, version macOS, nom du
GPU et contrôles sont enregistrés sous `results/`. Une archive
`montmory-metal-results-....tar.gz` rassemble les résultats, y compris les
diagnostics en cas d'échec. Aucun envoi réseau n'est effectué par le script.

Pour un passage plus court ou plus volumineux :

```bash
bash run.sh 1048576 3
bash run.sh 16777216 11
```

Le nombre de mots doit être une puissance de deux entre 65 536 et 33 554 432 ;
le nombre de répétitions doit être entre 3 et 101. Le passage à trois répétitions
sert surtout à vérifier le fonctionnement. Comparer ensuite plusieurs tailles
avant de généraliser un avantage.

Ce kit effectue des contrôles d'identité et de débit. Les batteries TestU01
SmallCrush/Crush/BigCrush constituent une validation statistique distincte.

## Provenance et portée de la préparation

Random123 est figé au commit `9545ff6413f258be2f04c1d319d99aaef7521150`, avec ses
headers officiels, sa licence et ses vecteurs. Le cœur C et le kernel Metal
Montmory sont copiés de la sauvegarde reçue du Mac, sans modification.

`generated/all.metal` est produit par prétraitement des sources incluses, afin
que la compilation Metal à l'exécution ne dépende pas de chemins d'inclusion.
Les headers système Metal restent inclus par le compilateur Apple. Le script
`tools/prepare_metal.py` permet de reproduire cette préparation avec Python 3
et un préprocesseur C++ ; il n'est pas nécessaire à l'utilisateur du kit.

Avant livraison, 17 047 296 mots du calcul Metal généré ont été comparés à la
référence dans une adaptation C++ exécutée sur x86 avec UBSan. Les six vecteurs
officiels concordent et les sentinelles sont préservées. La syntaxe shell a été
vérifiée. **Le programme Objective-C++ et le Metal natif restent à compiler et
exécuter sur le Mac : le VPS ne dispose ni des frameworks Apple ni du GPU.**
Ces contrôles préalables ne sont donc pas une mesure ou une certification M3.
