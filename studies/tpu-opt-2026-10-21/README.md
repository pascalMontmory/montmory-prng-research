# Campagne Montmory TPU — 21 octobre 2026

Cette etude optimise l'implementation du flux Montmory v3 avec une contrainte
de reproductibilite bit a bit CPU/GPU/TPU. Les anciens kits et leurs resultats
restent inchanges. Aucun gain des nouvelles variantes sur TPU n'est encore mesure.

## Livraison

- Notebook autonome : `../Montmory_TPU_optimisation.ipynb`.
- Kit : `../montmory-tpu-opt-kit.tar.gz`.
- SHA256 du kit : `a615a9c11c6c5378935090a552b7729efff17ce871aa719fdb1f600a08e1a372`.
- SHA256 du notebook : `e13790cd7d622c0d09aa03efd6e28e9bd7f640cf668e285793255858cc10bede`.
- Metadonnees : `results/delivery.json`.

Le notebook embarque le kit, controle son empreinte puis les 22 fichiers du
manifeste. Il affiche les logs en direct et propose le telechargement d'une archive.
Les fichiers livres sont figes ; toute correction ulterieure doit etre versionnee.

## Contrat preserve et modifications

La reference C et le module JAX historique sont conserves. Les constantes,
trois tours Feistel, melange final, seed, indice global et definition des lanes
logiques T=2048,D=4096 restent identiques. Une identite arithmetique remplace
le produit haut generique du facteur 1 ou 3 par un calcul de retenue, documente
dans `montmory-tpu-opt/README.md`. Les autres variantes changent la geometrie
d'execution Pallas. Quatre implementations Pallas Threefry2x32 a 20 tours
completent les comparateurs JAX pour une comparaison plus equitable.

La campagne couvre 26 configurations et quatre volumes, jusqu'a 128 Mio.
Les deux grands volumes couvrent toutes les lanes logiques. La recherche sur
trois seeds precede une selection figee puis une confirmation sur sept autres
seeds, trois blocs de trois repetitions. Toutes les sorties chronometrees sont
controlees. L'adoption exige un gain median apparie d'au moins 3 % et sept seeds
sur sept plus rapides que la reference Pallas 32x128. Sinon la reference reste
recommandee. Il s'agit d'une confirmation dans une session, pas d'une preuve
d'optimalite globale ou de reproductibilite des performances entre sessions.

Le module `tuned.py` applique une selection seulement pour le TPU, les versions
JAX/jaxlib et les volumes mesures compatibles. Ailleurs il conserve le chemin
portable. La compatibilite bit a bit concerne les mots et les mantisses entieres ;
elle ne promet pas l'identite de tous les calculs flottants des applications.

## Verification locale realisee

- `results/candidate-cpu/report.json` : 20 configurations, 1 684 047 mots
  compares et 1 048 592 valeurs de composant ; Pallas interprete sur CPU.
- `results/tpu-export/report.json` : 20 exports IR TPU reussis ; ceci ne
  constitue ni une compilation native complete ni une execution sur TPU.
- `results/original-protocol/report.json` : protocole historique reussi,
  empreinte `83b3c1647b315e3e28fdcec6ef494398d41cdaef0a9ceb764f1fdcd62c626033`.
- `reference/optimization-hashes.json` dans le kit : 240 empreintes CPU
  preetablies. Les 36 recoupements avec les anciennes references sont identiques.
- `results/api-smoke/report.json` : 16 386 mots de l'API verifies et trois
  plages invalides refusees.
- `results/runner-smoke/` : lancement complet en mode CPU explicite,
  recherche, selection figee et trois blocs de confirmation reussis.
- `results/delivery-checks.json` : recalcul independant de 222 echantillons
  et neuf lignes de synthese du test CPU, controle de la selection et de la
  regle d'adoption, syntaxe des quatre cellules et extraction du notebook.
- `results/no-tpu-guard/` : echec attendu sans TPU, archive de diagnostic
  produite et aucun repli silencieux vers CPU.

Les nouveaux noyaux restent a executer nativement sur le TPU de l'utilisateur.
Ils ne sont pas annonces mesures sur Mac/GPU ni soumis a de nouveaux tests
TestU01 ou TACM. Les comparaisons finies et l'identite arithmetique du composant
ne constituent pas une preuve universelle de tout le logiciel.

## Resultats natifs recus apres livraison

La campagne `20261021T185102Z-tpu-opt` est revenue avec succes sur un TPU v5 lite.
L'audit independant des 3 708 mesures et 44 lignes de synthese est passe ; les
40 empreintes Montmory ont ete regenerees depuis le C local fige. Les variantes
`pallas_carry_128` a 64 Mio et `pallas_carry_64` a 128 Mio satisfont la regle
d'adoption, avec +15,42 % et +23,25 % de gain apparie face a Pallas original 32.
La reference reste retenue a 512 Kio et 8 Mio. Threefry2 Pallas est environ 4 %
plus rapide que le Montmory retenu sur les deux grands volumes.

Voir [l'analyse complete](results/audit-20261021T185102Z-tpu-opt/README.md),
[l'audit machine](results/audit-20261021T185102Z-tpu-opt/audit.json) et
[le graphique](results/audit-20261021T185102Z-tpu-opt/confirmation-tpu.png).
Les fichiers du kit livre sont restes inchanges. La mention ci-dessus sur
l'execution native en attente decrit l'etat au moment de sa livraison.
