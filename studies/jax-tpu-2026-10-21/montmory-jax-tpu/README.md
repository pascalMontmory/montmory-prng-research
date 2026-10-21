# Montmory JAX / TPU : validation et comparaison reproductible

Prototype de recherche. Le flux Montmory existant reste inchange, T=2048,D=4096.
Le code utilise deux uint32 par mot de 64 bits, sans calcul flottant pour le PRNG.
La preparation des cles est reutilisee entre blocs. Il ne definit aucune
nouvelle operation `split` ou `fold_in` et ne modifie aucun paquet JAX.
La grille Pallas declare ses blocs independants (`dimension_semantics=parallel`)
pour permettre au compilateur de les repartir sur les coeurs du device TPU.
L'exploitation effective de ces coeurs et le gain restent a mesurer sur TPU.

## Sur Mac ou VPS (CPU)

```bash
bash run_cpu.sh
```

Python 3.11 ou plus recent et un compilateur C sont necessaires. Sur Mac, le
compilateur et le SDK sont trouves par `xcrun`. Les dependances sont installees
dans `.venv` ; ni Metal ni un TPU ne sont utilises par cette commande.
`MONTMORY_PYTHON` et `MONTMORY_VENV` permettent de choisir Python et un
environnement dedie existant. `bash run_cpu.sh --quick` reduit seulement les
mesures de performance, pas les controles de conformite.

Le protocole execute les vecteurs historiques, les deux formes de multiplication,
les decoupages, Pallas en mode interprete, l'export TPU et les partitions sur
1, 2, 4 devices CPU virtuels. Ces devices ne sont pas des TPU simules et ne
correspondent pas necessairement a autant de coeurs physiques.

## Sur un vrai TPU deja disponible

```bash
python -m pip install -r requirements-tpu.txt
python run_tpu.py
```

Le notebook autonome livre avec le kit realise ces etapes dans Colab. Il faut
choisir un environnement TPU avant de lancer ses cellules. L'acces depend du
compte et des quotas. Aucun script ne reserve ou ne loue de ressources cloud.
Le lanceur exige le backend TPU et echoue si celui-ci manque : il ne remplace
jamais silencieusement un essai TPU par un essai CPU.

Le lanceur controle les vecteurs historiques et les limites de compteurs sur
le backend actif, execute les noyaux Pallas natifs, verifie le decoupage JAX
sur 1, 2, 4 devices si disponibles, puis mesure un seul device. Les mesures
ne constituent pas un benchmark de cluster ou de communication inter-TPU.
Les logs et rapports sont archives, y compris apres une erreur de test.
Le kit ne transmet rien ; le notebook propose de telecharger l'archive finale.

## Export cible TPU : limite precise

```bash
python check_target.py --out results/export
```

L'export StableHLO/Mosaic pour TPU peut etre effectue sur CPU. C'est une etape
de traduction, pas une compilation complete en instructions d'un modele TPU,
et pas une execution. Sous JAX 0.10.2, nos chemins JAX portable/mulhi et nos
noyaux Pallas portables 8x128,16x128,32x128 passent cette etape. Le test optionnel
Pallas `mulhi` est refuse par le lowering Pallas TPU : primitive non implementee.
Le noyau Pallas livre utilise donc uniquement des produits partiels de 16 bits
dans des uint32. L'echec de cette sonde reste visible dans le rapport ; il n'est
pas transforme en resultat PASS. Les exports obligatoires doivent tous reussir.

## Mesures

Le protocole complet utilise 65536 et 1048576 mots de 64 bits stockes, cinq seeds
fixees, sept repetitions par seed et par moteur. Les configurations Pallas sont
fixees a 8x128,16x128,32x128 ; aucune selection sur les p-values ou les timings.
Les cles de tous les moteurs, y compris celles de Montmory, sont preparees et
attendues avant les mesures. Tous les pipelines sont compiles avant chronometrage.
Trois executions d'echauffement par seed/moteur sont verifiees hors chronometre.
L'ordre des moteurs est melange a chaque repetition selon un ordre reproductible.

Chaque temps est mesure cote appelant et inclut lancement, allocation des sorties
et `jax.block_until_ready`. Compilation, preparation des cles, transfert de la
sortie vers l'hote et verification sont exclus. Toutes les sorties chronometrees
sont verifiees. Les CSV conservent toutes les mesures, sans rejet d'outliers.
Le debit est le nombre d'octets de sortie divise par le temps ; il ne represente
pas une mesure directe des transactions HBM ou DRAM.

Tous les moteurs renvoient deux tableaux uint32 de meme longueur : huit octets
stockes par mot compare. Les concurrents utilisent l'API `jax.random.bits` de
JAX 0.10.2, avec forme (2,N). La seed de 64 bits est adaptee par `key(low32)`,
puis `fold_in(high32)`. Ce sont des flux JAX precis, pas necessairement les
memes compteurs, cles ou assemblages que les anciens benchmarks Metal Random123.

Les sorties Montmory sont comparees au C historique et a des empreintes figees.
Les empreintes des concurrents proviennent de l'API JAX CPU, pas d'une seconde
implementation independante de Random123. Sur TPU, elles servent a controler
la reproductibilite CPU/TPU pour les cas du protocole. Les versions exactes de
JAX et jaxlib sont exigees. Le controle d'ecriture est explicitement NON PRNG.

`--include-rbg` ajoute l'API XLA RBG. Ce moteur est separe : seules des sorties
repetees sur le meme backend sont comparees, pas une empreinte CPU/TPU commune.
Son comportement materiel precis depend du backend. Aucune qualite statistique
additionnelle ou reproductibilite intergenerations TPU n'est revendiquee.

Le tableau affiche la mediane des medianes par seed. Le ratio est la mediane
des ratios apparies par seed relativement a Montmory JAX portable ; il peut
donc differer du quotient de deux debits affiches. Les temps CPU ne permettent
pas de predire le classement TPU. Le protocole ne mesure pas de simulation aval
ni de generation fusionnee avec un consommateur.

## Etat de validation a la livraison

VPS x86 : controles CPU/Pallas interprete, export TPU, partitions CPU virtuelles
et comparaison des sept moteurs CPU executes. Le noyau Pallas natif et les
performances TPU restent a verifier sur du materiel reel. Mac ARM non execute
ici. Aucun nouveau TestU01/TACM lance. L'identite sur les cas controles ne prouve
pas tous les cas possibles et ne valide pas les futures operations split/fold_in.

Sources :
- https://docs.jax.dev/en/latest/pallas/tpu/prng.html
- https://docs.jax.dev/en/latest/pallas/tpu/details.html
- https://docs.jax.dev/en/latest/benchmarking.html
- https://docs.jax.dev/en/latest/export/export.html
