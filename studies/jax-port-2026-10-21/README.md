# Etude Montmory JAX / Pallas — 21 octobre 2026

Premier portage du flux Montmory v3 en paires uint32, et prototype Pallas.
Le repertoire livre `montmory-jax-check` est fige apres production de l'archive.
Les evolutions doivent produire une nouvelle revision, sans changer ce kit.

Resultat VPS x86 : PASS sous JAX/jaxlib 0.10.2, numpy 2.4.6, Python 3.11.2.
129 controles de comparaison ; 12 plages invalides rejetees. Aucun calcul TPU
natif, aucune mesure de performance TPU, aucune nouvelle batterie statistique.
Le Mac ARM reste a tester par l'utilisateur.

Comptes des comparaisons (repetitions entre methodes et formats incluses) :

| Categorie | Comparaisons |
|---|---:|
| Reference C vs 1600 vecteurs x86 figes | 1600 |
| Vecteurs JAX raw64 eager/JIT, deux produits hauts | 6400 |
| Vecteurs JAX mantissa53 entiere | 6400 |
| Composants mulhi | 200098 |
| Seeds, indices et lanes aleatoires 64 bits | 65536 |
| Indices globaux aleatoires et limites | 393216 |
| Plages raw64 | 2359296 |
| Plages mantissa53 entiere | 2359296 |
| Morceaux executes en ordre inverse | 32770 |
| Pallas interprete, trois tailles de blocs | 147474 |

La reference C historique et les 1600 vecteurs sont copies sans modification.
Le noyau JAX garde T=2048,D=4096 et tous les calculs modulo 2^64. Les composants
uint32 utilisent soit une decomposition en produits 16 bits, soit `lax.mulhi`.
Les morceaux de sortie ne dependent pas de leur ordre d'execution. Les essais
Pallas changent les blocs physiques (8,16,32)x128 sans changer la geometrie
logique. Le mode x64 de JAX est desactive dans les essais.

Le premier essai (`results/vps-initial/report.json`) passait les controles JAX
mais echouait a construire le mapping Pallas : le callback devait accepter
l'argument de prechargement scalaire. La correction de cette signature a ete
suivie d'un passage complet reussi (`results/vps-confirmed/report.json`), puis
d'un passage reussi du lanceur livre avec verification de l'empreinte des sorties.

Le lanceur complet a utilise l'environnement dedie deja installe
(`MONTMORY_VENV=../.venv bash run.sh`). Son rapport et son journal sont sous
`montmory-jax-check/results/20261021T172909Z-2/`. Il a cree une archive de resultats.
L'installation neuve sur macOS n'a pas ete executee ici.

Empreinte deterministe des sorties comparees :
`83b3c1647b315e3e28fdcec6ef494398d41cdaef0a9ceb764f1fdcd62c626033`.

Livraison : `/home/pascal/prng-recovery/montmory-jax-check-kit.tar.gz`
(49332 octets, 12 fichiers ; controle du contenu archive effectue).
SHA256 : `6e38a22493c9923110e98a5ac9e2738c3e55829abe0f5109b9b80da0c9cee739`.
Metadonnees dans `results/delivery.json`.

Le commit JAX GitHub examine est
`d330fa6df1cfe4b9a86739ae3c597d37fc0a590b` ; il est distinct des paquets PyPI
0.10.2 utilises pour les essais. Aucun fork JAX publie ou modifie, aucun TPU
loue. Le module Montmory n'enregistre pas de nouvelles operations split/fold_in.
Les performances des unites matricielles d'un TPU ne prejugent pas de celles
de cet algorithme entier. Aucun avantage sur Threefry ou Philox n'est etabli.
