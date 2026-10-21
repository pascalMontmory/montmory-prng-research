# Montmory : continuation JAX / TPU, 21 octobre 2026

Le kit `montmory-jax-tpu` est fige a la livraison. Il prolonge le premier
portage sans modifier le precedent kit. Aucun TPU loue par l'agent, aucun
depot externe modifie ou publie. Le Mac de l'utilisateur n'est pas accessible.
Les controles locaux ci-dessous precedent le premier retour Colab rapporte
par l'utilisateur, consigne a la fin de ce document.

## Changements et controles

La preparation des cles a ete extraite du calcul des blocs et peut etre
reutilisee. L'empreinte des sorties du premier protocole est conservee :
`83b3c1647b315e3e28fdcec6ef494398d41cdaef0a9ceb764f1fdcd62c626033`.
Le flux reste celui de Montmory v3 T=2048,D=4096, avec indice global et
arithmetique exacte modulo 2^64 representee en deux uint32.

129 controles du premier protocole PASS, rapport `results/prepared-cpu/report.json`.
Les partitions JAX sur 1,2,4 devices CPU virtuels donnent 786432 comparaisons
de mots PASS (`results/cpu-sharding/report.json`). Cela n'est pas une mesure
de coeurs physiques ni une simulation de performances TPU.

Le nouveau harnais destine aux essais natifs a ete execute sur CPU, avec
Pallas interprete : 1972720 comparaisons de mots PASS, y compris les mantisses
entieres et les frontieres de compteurs. La grille Pallas indique que ses
blocs sont independants afin d'autoriser leur repartition entre coeurs TPU.
La version finale de cette grille est controlee dans `results/parallel-grid-cpu`.

L'export StableHLO/Mosaic cible TPU passe pour JAX portable/mulhi et pour
Pallas portable 8x128,16x128,32x128. La sonde Pallas `mulhi` echoue car cette
primitive n'est pas implementee dans le lowering Pallas TPU de JAX 0.10.2.
Les noyaux Pallas livres utilisent les produits partiels 16 bits dans des
uint32. Rapports finaux dans `results/tpu-export-parallel`. L'export n'est
ni une compilation complete pour une generation physique de TPU, ni une
execution native. Les fichiers IR sont conserves comme diagnostics.

## Mesures x86 JAX CPU

JAX/jaxlib 0.10.2, deux tailles, cinq seeds, sept repetitions : 490 mesures,
272957440 mots chronometres compares. Les cles sont preparees avant les mesures,
tous les pipelines sont precompiles, les appels attendent `block_until_ready`.
Les controles et transferts de sortie sont hors chronometre. Les mesures portent
sur un seul device JAX CPU ; cela ne signifie pas un seul coeur physique.

| Moteur | 65536 mots, GB/s | 1048576 mots, GB/s |
|---|---:|---:|
| Montmory JAX portable | 1.651 | 2.761 |
| Montmory JAX mulhi | 1.717 | 3.011 |
| Threefry2x32 JAX | 0.422 | 0.512 |
| Threefry4x32 JAX | 1.725 | 3.132 |
| Philox4x32 JAX | 1.910 | 3.940 |
| Philox2x32 JAX | 1.972 | 6.304 |
| Temoin d'ecriture NON PRNG | 6.889 | 10.297 |

Ces debits sont les medianes des medianes par seed pour ces implementations
et ce VPS. Ils ne prejugent pas du classement sur TPU, ni du classement entre
algorithmes avec d'autres implementations. Les flux des concurrents sont ceux
de l'API JAX, qui ne reproduit pas necessairement les adaptations Random123
des anciens benchmarks Metal. Aucun avantage statistique n'est demontre ici.
Les CSV, temps, ratios apparies et compteurs ont ete recomputes/controles
separement dans `results/verification-summary.json`.

## Livraison et limites

- Le lanceur CPU complet a passe un essai de livraison avec mesures courtes
  et l'option RBG (`results/delivery-smoke`). Cet essai ne remplace pas les
  mesures completes ci-dessus ; l'option RBG verifie seulement la repetition
  sur le meme backend.
- Le lanceur TPU a ete teste sur le VPS sans libtpu : echec attendu, archive
  de diagnostic creee, aucun repli CPU (`results/no-tpu-guard`). Ce n'est pas
  un essai de compilation/execution TPU du moteur.
- Le notebook contient une copie complete du kit, son empreinte verifiee,
  l'installation des versions figees et les commandes pour un TPU deja alloue.
  Les cellules Python ont ete analysees syntaxiquement et l'extraction de
  l'archive embarquee a ete executee dans un dossier temporaire ; les 20
  empreintes des fichiers ont ete controlees (`results/notebook-check.json`).
  Le notebook n'a pas ete execute dans Colab ici.
- A la livraison, les essais natifs TPU et ARM macOS, les performances TPU, les nouvelles
  operations split/fold_in et les tests statistiques correspondants restent
  hors de la validation realisee.

Archive : `/home/pascal/prng-recovery/montmory-jax-tpu-kit.tar.gz`, 60184 octets,
21 fichiers, SHA256 `92baf16a55f3822298d3b42b21677a0b5828343640e7281a7d826a0db624f172`.

Notebook : `/home/pascal/prng-recovery/Montmory_TPU_validation.ipynb`, 85904 octets,
SHA256 `0a1daca474fecc0800bca4a92e92d5f5535d542a245075d728332c12f801fad4`.
Metadonnees dans `results/delivery.json`.

## Premier retour natif TPU rapporte par l'utilisateur

Le 21 octobre 2026, l'utilisateur a colle les sorties de son execution Colab.
Le premier echec etait survenu avant les comparaisons, lors de l'initialisation
TPU ; l'utilisateur a ensuite confirme que TPU n'etait pas selectionne et a
change d'accelerateur. Le retour suivant indique : empreintes des 20 fichiers
PASS, vecteurs historiques PASS, frontieres PASS pour les quatre seeds,
partition sur un device TPU PASS, puis deux volumes et cinq seeds par volume
avec toutes les sorties chronometrees controlees. Les variantes Pallas sont
executees nativement dans ce protocole, sans interpretation CPU.

| Moteur | 65536 mots, GB/s rapportes | 1048576 mots, GB/s rapportes |
|---|---:|---:|
| Montmory JAX portable | 1.808 | 14.450 |
| Montmory JAX mulhi | 1.823 | 14.199 |
| Threefry2x32 JAX | 1.850 | 13.157 |
| Threefry4x32 JAX | 1.648 | 10.560 |
| Philox4x32 JAX | 1.535 | 7.740 |
| Philox2x32 JAX | 1.752 | 10.538 |
| Temoin d'ecriture NON PRNG | 1.933 | 16.961 |
| Montmory Pallas 8x128 | 1.895 | 12.462 |
| Montmory Pallas 16x128 | 1.886 | 14.278 |
| Montmory Pallas 32x128 | 1.920 | 15.254 |
| RBG, controle sur le meme backend seulement | 1.855 | 17.608 |

Sur le grand volume, le quotient des debits resumes donne +15.94 % pour
Montmory Pallas 32x128 face a Threefry2x32, +44.45 % face a Threefry4x32,
+44.75 % face a Philox2x32 et +97.08 % face a Philox4x32. Ces quotients ne
sont pas des ratios apparies recomputes : les ratios affiches par le programme
sont des medianes de ratios par seed, qui peuvent differer du quotient des
debits resumes. Pallas 32x128 est le meilleur des trois blocs mesures dans
cet essai ; il n'est pas confirme optimal sur toutes tailles ou generations.
RBG reste plus rapide sur le grand volume et sa conformite inter-backends
n'est pas revendiquee.

Au moment de ce premier retour textuel, les fichiers n'avaient pas encore ete
recus. Ils ont depuis ete transmis et audites comme detaille ci-dessous.
Le chronometrage inclut dispatch, allocation et attente de completion ; la
compilation, la preparation des cles et les transferts/controles de sorties
en sont exclus. Le seul essai de partition rapporte porte sur un device,
pas sur plusieurs devices TPU. L'avertissement hugepages n'a pas empeche
les controles affiches ; aucun effet chiffre sur les mesures n'est etabli.

## Audit des fichiers Colab recus

Dossier original conserve a `20261021T180635Z-colab-tpu/`. Le fichier
`run-status.json` est PASS et liste les quatre etapes terminees. JAX/jaxlib
0.10.2, device_kind exact `TPU v5 lite`, un seul device visible. Les deux
tailles sont 65536 mots (512 Kio) et 1048576 mots (8 Mio), a 8 octets par mot
stocke sous forme de deux tableaux uint32 ; cela ne mesure pas une bande
passante physique HBM. Cinq seeds, sept repetitions par seed et par moteur,
11 moteurs, soit 770 mesures. Trois appels de chauffe controles precedaient
les mesures de chaque moteur/seed. L'ordre des moteurs est melange a chaque
repetition. Compilation, preparation des cles et transferts de controle sont
hors chronometre ; lancement, allocation et attente sont inclus.

`audit_colab.py` a controle les 20 empreintes du kit local, les statuts,
la grille complete de mesures sans doublon, l'ordre pseudoaleatoire prescrit,
les empreintes CSV et les 22 lignes du resume. Les 10 empreintes Montmory
ont ete regenerees avec la reference C locale recompilee. Aucun binaire .so
recu n'a ete charge. Les fichiers originaux sont empreintes dans le rapport.

Resultats de l'audit :
- 1972720 comparaisons raw64/mantisses entieres rapportees dans la conformite,
  avec 120 cas complets, les vecteurs historiques et Pallas natif.
- 262144 comparaisons de partition rapportees, sur un device seulement.
- 700 mesures / 389939200 mots dont les empreintes correspondent aux references
  CPU figees ; 70 mesures / 38993920 mots RBG controles pour repetabilite locale.
- 428933120 mots chronometres controles au total, repetitions incluses. Ces
  repetitions ne sont pas autant d'echantillons statistiques independants.
- Aucune nouvelle batterie TestU01/TACM ni validation split/fold_in Montmory.

### Ratios apparies recalcules

Pour chaque seed s et chaque moteur, t est la mediane des sept temps.
Le rapport compare mediane_s(t_reference(s)/t_Montmory(s)). Les gains ci-dessous
portent sur le debit, pas sur une reduction du temps de meme pourcentage.
Ils different legitimement du quotient des deux debits resumes : ces deux
debits peuvent retenir des seeds differentes pour leurs medianes.

| Reference | Gain Pallas 32x128, 512 Kio | Gain Pallas 32x128, 8 Mio | Seeds gagnantes, 8 Mio | Gain min..max par seed, 8 Mio |
|---|---:|---:|---:|---:|
| Threefry2x32 | +3.69 % | +19.28 % | 5/5 | +14.29..+25.43 % |
| Threefry4x32 | +16.83 % | +51.51 % | 5/5 | +42.03..+57.85 % |
| Philox2x32 | +8.82 % | +47.52 % | 5/5 | +42.69..+55.93 % |
| Philox4x32 | +25.50 % | +102.90 % | 5/5 | +89.39..+122.39 % |
| Montmory JAX portable | +6.21 % | +7.00 % | 5/5 | +5.57..+11.39 % |
| Montmory JAX mulhi | +4.80 % | +9.47 % | 5/5 | +3.69..+17.17 % |
| Montmory Pallas 8x128 | +1.12 % | +29.56 % | 5/5 | +16.92..+37.11 % |
| Montmory Pallas 16x128 | +0.55 % | +9.10 % | 5/5 | +3.18..+14.92 % |
| RBG (meme backend) | +3.73 % | -7.57 % | 0/5 | -17.13..-2.96 % |
| Temoin NON PRNG | -1.15 % | -8.35 % | 0/5 | -14.56..-0.84 % |

Montmory JAX portable depasse deja Threefry2x32 de +12.99 % en ratio apparie
sur 8 Mio, avec cinq seeds gagnantes. Le gain ne depend donc pas seulement
du choix de Pallas. Toutefois les comparateurs sont les implementations de
l'API JAX 0.10.2, pas des noyaux Pallas specifiquement optimises ici. Le nombre
de rounds, les espaces de cles et les garanties des moteurs ne sont pas
interchangeables. Ces essais ne classent pas leur qualite statistique.

### Variabilite, volumes et lanes

Pour Pallas 32x128 sur 8 Mio, la mediane des medianes vaut 549.92 microsecondes
et 15.254 GB/s ; les medianes par seed vont de 490.95 a 569.78 microsecondes.
La mediane, sur les cinq seeds, de l'ecart absolu median normalise vaut 5.31 %.
Pour Threefry2x32, elle vaut 3.69 %. Aucun temps extreme n'a ete elimine.
Le gain de 19.28 % est observe pour chacune des cinq seeds, mais ces cinq
seeds et sept repetitions proviennent d'une seule session. Les plages min/max
ne sont pas des intervalles de confiance. Les seeds sont executees successivement ;
leur dispersion ne permet pas d'attribuer la variation uniquement a la seed.
La selection du meilleur bloc a utilise ces memes mesures : une confirmation
avec configurations figees, nouvelles seeds et nouvelles sessions reste a faire.

Sur 512 Kio, Pallas 32x128 prend 273.05 microsecondes et le temoin 271.29 ;
sur 8 Mio, Pallas prend 549.92 microsecondes. Un volume multiplie par 16 ne
multiplie donc le temps que par environ 2.01. Cela suggere un cout fixe important
des appels pour les petits volumes ; aucun profilage n'isole sa cause exacte.
Le debit sur 8 Mio n'est pas un plafond de debit TPU demontre.

Les blocs Pallas 8x128,16x128,32x128 produisent respectivement 1024,2048,4096
mots par invocation de programme. Ce sont des decoupages de travail, pas des
nombres de coeurs physiques. Ils conservent T=2048,D=4096 et le flux logique.
A la base 4294963199, les plages chronometrees visitent 17 lanes logiques
distinctes pour 65536 mots et 257 pour 1048576 mots, pas la totalite des 2048.
Les petits blocs de sortie sont des prefixes des grands. Les benchmarks
Montmory couvrent 5242880 couples (seed, indice) distincts, reutilises par les
variantes et repetitions. Les controles de conformite explorent d'autres
frontieres separement. Aucun test de performance multi-device n'a ete execute.

La conformite constate l'identite des entiers sur les cas controles. Elle
n'etablit pas l'egalite de toutes les sorties possibles, l'identite de calculs
flottants de distributions transformees, ni une nouvelle qualite statistique.
RBG n'a ici que des controles sur le meme backend. La documentation JAX
ne lui attribue pas la meme identite CPU/GPU/TPU que Threefry et Philox :
https://docs.jax.dev/en/latest/jax.random.html#advanced-rng-configuration .
Le prechauffage et l'attente explicite suivent les recommandations de
https://docs.jax.dev/en/latest/benchmarking.html .

Les prochaines confirmations utiles sont une session independante avec les
configurations figees, des volumes couvrant le cycle de lanes (au moins
8388608 mots), un profilage des noyaux et des comparateurs specialises de
maniere comparable. Une integration applicative avec consommation directe
des sorties est necessaire pour mesurer un gain sur une simulation complete.

Audit detaille : `results/audit-20261021T180635Z-colab-tpu/audit.json`.
Graphique exportable PNG/PDF : `results/audit-20261021T180635Z-colab-tpu/comparaison-tpu.*`.
L'audit porte sur les donnees recues ; il ne constitue pas une nouvelle
execution independante sur un TPU.
