# Montmory TPU : optimisation conservant le flux v3

Cette campagne conserve la reference C, les trois tours Feistel, les constantes,
le melange final, la seed, les indices globaux et les lanes logiques T=2048,D=4096.
Elle cherche une implementation plus rapide du meme flux. Aucune garantie de
gain ou d'optimalite globale n'est annoncee avant les mesures natives.

## Equivalence de la nouvelle multiplication

Pour x uint32, la fonction de tour utilise x si x est pair, 3x sinon. Posons
M=2^32 et p=x mod 2. La nouvelle implementation calcule :

    low = x * (1 + 2*p) modulo M
    high = ((x >> 31) & p) + [low < x]

Si x est pair, low=x et high=0. Si x est impair, on ecrit 3x=x+2x.
La partie haute de 2x est x>>31. L'addition de x a la partie basse de 2x
produit une retenue exactement lorsque low<x. La paire low/high est donc
egale au produit mathematique 64 bits pour tout x uint32. Les autres operations
du tour restent identiques. Les comparaisons finies de code complet completent
cette identite arithmetique ; elles ne sont pas une preuve universelle du logiciel.

## Candidats et protocole

- 16 variantes Montmory : JAX portable/mulhi, avec/sans retenue specialisee,
  et Pallas original/retenue sur 8,16,32,64,128,256 lignes de 128 mots.
- Quatre variantes Pallas Threefry2x32 a 20 tours (16,32,64,128 lignes),
  compatibles avec les sorties uint32 partitionables de l'API JAX 0.10.2.
- API JAX Threefry2/4 et Philox2/4, RBG, temoin d'ecriture non PRNG.
- Volumes 512 Kio, 8 Mio, 64 Mio, 128 Mio. Les deux derniers couvrent
  toutes les 2048 lanes logiques, sans modifier leur definition.
- Recherche sur trois seeds et trois repetitions. Deux finalistes Montmory
  et la meilleure implementation Threefry2 sont choisis par volume.
- Selection sauvegardee avant toute confirmation. Confirmation sur sept
  autres seeds, trois blocs de trois repetitions. L'ordre des moteurs est
  melange a chaque repetition. Tous les temps, y compris extremes, sont gardes.
- La configuration primaire de recherche n'est adoptee que si son gain median
  apparie est au moins 3 % et qu'elle gagne pour les sept seeds face au Pallas
  original 32x128. Sinon ce dernier reste la recommandation. Le secondaire est
  rapporte mais ne remplace pas opportunement le primaire apres confirmation.
- Deux appels de chauffe controles par moteur/seed/bloc. Compilation et
  preparation des cles exclues du chronometre pour tous les moteurs. Lancement,
  allocation et attente de completion inclus. Transferts et controles exclus.
- Chaque sortie chronometree est comparee a une empreinte CPU figee avant la
  campagne. Montmory utilise la reference C ; les concurrents l'API JAX CPU.
  RBG est separe : repetabilite sur le meme backend seulement.
- Une divergence de sortie arrete l'essai. Un candidat Pallas optionnel que
  le compilateur refuse est exclu avec diagnostic ; la reference est obligatoire.

La selection peut differer selon le volume. Elle ne s'applique pas automatiquement
a une autre generation de TPU, une autre version JAX ou un autre format de sortie.
Le protocole ne mesure pas l'interieur du noyau seul ni une application complete.
Il ne change pas le moteur statistique et ne teste pas de nouvelles operations
split/fold_in Montmory. Les mantisses controlees sont des entiers raw64>>11 ;
aucune identite de calculs flottants CPU/GPU/TPU n'est deduite.

## Utilisation

Importer le notebook livre dans Colab, choisir TPU, executer les quatre cellules
dans l'ordre. Le lancement utilise un sous-processus neuf et affiche les logs
en direct. Aucun TPU n'est provisionne ou loue par ces scripts, aucun resultat
n'est transmis automatiquement. Une archive est produite en cas d'echec.

Le mode `python run.py --cpu-smoke` est seulement un test local du harnais :
il ne mesure pas de performances TPU et exclut les candidats Pallas des mesures.
`verify_candidates.py --backend cpu` controle les nouveaux noyaux en interpretation.
`export_candidates.py DOSSIER` exporte l'IR cible TPU sans execution native.

## Sources et portee

Regles JAX de chronometrage : https://docs.jax.dev/en/latest/benchmarking.html
Pallas TPU : https://docs.jax.dev/en/latest/pallas/tpu/details.html
Threefry : Salmon et al. (2011), Random123 ; implementation de reference JAX
Apache-2.0, https://github.com/jax-ml/jax/blob/jax-v0.10.2/jax/_src/random/threefry2x32.py .
Les empreintes de l'API JAX CPU ne constituent pas des KAT Random123 independants.
La conformite du portage GPU/ARM existant est conservee comme reference historique ;
les nouveaux noyaux JAX/Pallas ne sont pas annonces testes nativement sur GPU/ARM.
