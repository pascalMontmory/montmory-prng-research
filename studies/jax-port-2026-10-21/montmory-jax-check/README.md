# Montmory : premier portage JAX et prototype Pallas

Ce kit controle le flux Montmory existant sur le CPU de la machine locale.
Il execute aussi le noyau Pallas en mode interprete sur CPU. Il ne mesure
aucune performance TPU et ne certifie pas la compilation native TPU.

## Utilisation

Prerequis : Python 3.11 ou plus recent et compilateur C. Sur Mac, Xcode ou ses
Command Line Tools doivent etre disponibles. Le kit utilise `xcrun` et le SDK
macOS explicite pour la reference C. Il n'utilise pas le GPU Metal.

```bash
bash run.sh
```

Au premier lancement, JAX et ses dependances sont telecharges dans `.venv`.
Les versions sont figees dans `requirements.txt` (JAX et jaxlib 0.10.2).
L'installation ne change pas les paquets Python du systeme ou de Conda.
Pour choisir Python, utiliser `MONTMORY_PYTHON=/chemin/vers/python3 bash run.sh`.
`MONTMORY_VENV` permet de choisir un autre environnement dedie (defaut `.venv`).
Une archive de diagnostics est creee a la fin, y compris en cas d'echec apres
le controle des sources. Aucun fichier n'est transmis automatiquement.

## Contrat du flux

Tous les mots sont calcules en arithmetique entiere non signee, modulo 2^64,
representee par deux uint32 (partie basse, partie haute). JAX x64 est desactive
dans le protocole de validation. Deux formes du produit haut sont controlees :
decomposition exacte en produits de 16 bits, et primitive `jax.lax.mulhi`.

Pour l'indice global k : lane = floor(k/4096) modulo 2048 ;
gamma = 0x9E3779B97F4A7C15 + 2*lane ; compteur = seed + (k+1)*gamma modulo 2^64.
Les cles, les trois tours Feistel et le melange final restent ceux du moteur C
historique. La routine historiquement nommee `pcg_dxsm` est notre melange
personnalise, et ne constitue pas une implementation du PRNG PCG64DXSM.

Les compteurs internes peuvent deborder volontairement. Une plage dont les
indices globaux debordent est rejetee. La geometrie logique T=2048,D=4096 est
fixe. La taille des blocs Pallas et l'ordre d'execution des morceaux n'y changent
rien. Les blocs partiels Pallas sont completes puis tronques a la taille demandee.

`mantissa_words` renvoie exactement l'entier raw64 >> 11, sous forme de deux
uint32. Ce n'est pas une conversion float32 ou bfloat16. Aucune identite des
calculs flottants d'une simulation aval n'est revendiquee.

## Controles

- reference C native reconstruite depuis `mmc_core.c` et `.h`, copies inchangees ;
- 1600 vecteurs historiques x86, avec JAX eager et JIT ;
- produits hauts, seeds, lanes et indices sur toute la largeur de 64 bits ;
- passages des limites de lanes, blocs, 2^32, 2^63 et fin de l'espace d'indices ;
- sorties raw64 et mantisses entieres de 53 bits ;
- morceaux executes en ordre inverse et parametres invalides ;
- noyau Pallas interprete, blocs 8x128, 16x128 et 32x128, sorties partielles ;
- empreinte SHA256 deterministe de toutes les sorties controlees, comparee
  au rapport x86 livre dans `reference/vps-report.json`.

Les comptes du rapport sont des comparaisons de mots avec repetitions entre
methodes et representations, pas un nombre de sorties toutes distinctes.
Ces controles sont finis : ils ne prouvent pas tous les cas possibles.

## Travail restant

Execution et compilation natives sur TPU ; profilage et comparaisons de vitesse
avec Threefry, Philox et les PRNG materiels pertinents ; comportement sur plusieurs
devices. Le mode natif existe dans le prototype (`interpret=False`) mais reste
non valide et refuse de s'executer si le backend par defaut n'est pas un TPU.

Ce module ne s'enregistre pas comme une implementation complete de `jax.random`.
Les operations `split` et `fold_in` restent a concevoir et valider. Aucune nouvelle
batterie TestU01 ou TACM n'a ete lancee ; le BigCrush historique ne constitue pas
une validation de ces operations futures. Aucun gain TPU n'est annonce.

Sources techniques :
- https://docs.jax.dev/en/latest/pallas/tpu/details.html
- https://docs.jax.dev/en/latest/pallas/tpu/prng.html
- https://docs.jax.dev/en/latest/jax.extend.random.html
- https://github.com/jax-ml/jax/tree/d330fa6df1cfe4b9a86739ae3c597d37fc0a590b/jax/_src/random

Le commit GitHub examine et la version PyPI executee sont deux references
distinctes ; les essais de ce kit utilisent les paquets PyPI figes ci-dessus.
