# Étude des lanes Montmory — 21 octobre 2026

Le nombre de workers CPU peut varier sans changer un seul mot du flux Montmory fixé par seed, T=2048 lanes logiques et D=4096 tirages par lane. La campagne CPU réalisée sur le VPS vérifie cette propriété sur 830 472 192 mots. Le kit Metal séparé prépare le même contrôle pour le Mac M3 ; aucune mesure native Apple n'a été exécutée sur le VPS.

## Résultats CPU obtenus

Mesure de génération raw64 et écriture de 8 388 608 mots (64 Mio), avec neuf répétitions et neuf seeds, ordre mélangé, échauffement de chaque configuration. Référence scalaire C originale et comparaison intégrale hors chronométrage. CPU virtuel AMD EPYC-Milan, 16 vCPU, OpenMP avec affinité rapprochée. Les chiffres sont propres à cette exécution du VPS partagé.

| Workers CPU | Médiane ns/mot | Débit Go/s décimaux | Accélération / 1 worker |
|---:|---:|---:|---:|
| 1 | 3,956 | 2,02 | 1,00 |
| 2 | 2,065 | 3,87 | 1,92 |
| 4 | 1,084 | 7,38 | 3,65 |
| 8 | 2,388 | 3,35 | 1,66 |
| 16 | 2,588 | 3,09 | 1,53 |

Quatre workers donnent la meilleure médiane dans cette campagne. La cause du ralentissement à huit et seize workers n'est pas établie. Les cgroups consultés ne montrent ni quota CPU ni throttling ; cela n'exclut pas la contention du VPS ou d'autres limitations. Ce résultat ne détermine pas un optimum sur M3.

À quatre workers, modifier simultanément T et D à volume constant donne des médianes entre 1,050 et 1,125 ns/mot. Les dispersions se recouvrent : cette campagne ne démontre pas un gain utile dû au changement de géométrie. Les mots produits changent presque tous, contrairement à la variation des workers physiques. Le détail est dans `results/cpu_geometry_changes.csv`.

## Contrat de sortie

Pour un bloc de N=T·D mots, à base zéro, la position i utilise la lane logique `floor(i/D)` et le compteur `seed + (i+1)*(0x9E3779B97F4A7C15 + 2*lane)` modulo 2^64, puis la permutation et le mélange d'origine. Le nombre de workers n'apparaît pas dans cette définition : il répartit les calculs. La comparaison expérimentale vérifie les implémentations pour les configurations et seeds choisies ; elle n'est pas une preuve exhaustive de correction de tous les programmes possibles.

Les résultats SmallCrush et Crush transmis pour le Mac concernent T=2048,D=4096, seed42 et l'adaptateur TestU01 utilisé. Le précédent benchmark Metal v3 utilisait T=8192,D=512. La série A du nouveau kit garde la géométrie de TestU01. La série B modifie T et D : ses performances éventuelles ne valent pas validation statistique du nouveau flux. Une égalité du flux et de l'adaptateur conserve les données soumises aux tests, quel que soit le matériel qui les calcule.

## Kit M3

`mac/montmory-lanes-m3/README.md` décrit les deux séries. Le kit compare Montmory, Threefry2x32-20 et Philox4x32-10, sur les formats raw64 et mantissa53 stockés en 64 bits. Sept nombres de threads et quatre tailles de groupe sont explorés. Chaque sortie chronométrée est vérifiée intégralement. Les configurations ne sont retenues que si elles respectent les limites du GPU.

Les kernels Metal, le cœur C et les vecteurs d'origine sont repris sans modification du kit v3. La transcription C++ des calculs entiers passe 904 cas, soit 167 444 480 mots, avec UndefinedBehaviorSanitizer et gardes mémoire. La compilation Objective-C++ et l'exécution native GPU restent à faire sur le Mac. Le script archive les diagnostics en cas d'échec et n'envoie rien automatiquement. Attendre la fin de BigCrush avant les mesures.

## Reproduction

Le protocole est fixé dans `protocol.json`. Les résultats CPU bruts, leur résumé et les limites cgroup observées sont dans `results/`. `python3 analyze_cpu.py` valide la complétude et régénère le résumé, sans relancer les mesures.

Pour refaire les mesures CPU dans un nouveau dossier de résultats :

```bash
g++ -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra \
  -Imac/montmory-lanes-m3/src -Imac/montmory-lanes-m3/vendor/include \
  src/cpu_lanes.cpp -o build/cpu_lanes
mkdir -p results-replay
OMP_PROC_BIND=close OMP_PLACES=threads OMP_WAIT_POLICY=PASSIVE \
  build/cpu_lanes results-replay
```

Pour refaire la vérification portable des corps Metal, depuis le dossier du kit : `python3 tools/verify_lanes.py`. Aucun test TACM ou TestU01 supplémentaire n'est exécuté par ce benchmark de performances.
