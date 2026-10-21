# Benchmark applicatif JAX — protocole v1

[Accueil](../README.fr.md) · [Programme](../applications/jax_rng/benchmark.py) · [Résultats bruts des kernels](results.md)

**Objectif : mesurer le temps d'une opération utilisant le RNG, avec le consommateur dans le même `jit`.** Ce protocole est distinct de la campagne de débits bruts et n'en modifie aucun kernel. Il fournit un premier cas applicatif reproductible ; il ne représente pas un entraînement IA complet ni un moteur VaR.

## Trois opérations

| Opération | Travail chronométré | Sortie conservée |
|---|---|---|
| `uniform` | Génération des mots, extraction de 23 bits par uint32, conversion en uniformes float32 strictement dans `(0,1)`. | Tableau `(2, count64)` float32. |
| `normal` | Même génération et conversion, puis Box–Muller avec les deux sorties sinus/cosinus. | Tableau `(2, count64)` float32. |
| `gaussian_expectation` | Même génération et normales, puis estimation de `E[exp(-Z²/2)]` par moyenne. | Un scalaire float32. |

L'espérance idéale pour une normale standard continue est `1/sqrt(2)`. Les estimations sont conservées à titre diagnostique ; ce repère n'est ni une référence bit à bit, ni un seuil de passage statistique.

Le maillage uniforme est `((word >> 9) + 0.5) * 2^-23`, sans zéro ni un. Il utilise 23 bits par demi-mot, indépendamment du format historique `mantissa53`. Il produit une approximation discrète des normales, dont les queues sont limitées par ce maillage. Ce choix explicite ne constitue pas une validation pour des quantiles financiers extrêmes.

## Comparaisons prévues avant les mesures

Les trois opérations utilisent exactement les mêmes transformations float32 pour Montmory `jax_original`, Threefry2x32 JAX à vingt tours en mode partitionable, Philox4x32 JAX et, sur TPU, les deux variantes Pallas ci-dessous. La géométrie Montmory reste `T=2048,D=4096`. Les configurations viennent de la campagne de **débits bruts antérieure** ; elles ne sont pas sélectionnées sur les nouveaux résultats applicatifs.

| `count64` | Nombres float32 produits | Montmory Pallas | Threefry Pallas |
|---|---:|---|---|
| 65 536 | 131 072 | `pallas_original_32` | `threefry_pallas_16` |
| 1 048 576 | 2 097 152 | `pallas_original_32` | `threefry_pallas_32` |
| 8 388 608 | 16 777 216 | `pallas_carry_128` | `threefry_pallas_128` |
| 16 777 216 | 33 554 432 | `pallas_carry_64` | `threefry_pallas_128` |

Le programme ajoute **`threefry_native_normal`** : `jax.random.normal` directement, puis le même consommateur pour l'espérance. Cette comparaison répond à l'usage applicatif courant. Elle utilise une transformation différente et ne doit pas être décrite comme une comparaison à sorties identiques avec Box–Muller. Les différences de discrétisation, de queues et de dépendances doivent être examinées avant une substitution en production.

Le chemin Pallas conserve son découpage antérieur et peut matérialiser des tableaux intermédiaires. Un `jit` englobant ne garantit pas la fusion des opérations avec le corps Pallas. Cet effet d'intégration fait partie du résultat applicatif à mesurer. Aucun Philox Pallas spécialisé n'est inclus.

## Mesures et contrôle

- **JAX/jaxlib 0.10.2 requis**, un device ; aucune émulation Pallas chronométrée sur CPU et aucun repli silencieux TPU→CPU.
- Base Montmory `2^32−4097`. La base n'est pas appliquée aux compteurs Threefry ; leurs sorties suivent la convention JAX `(2,count64)`. Les moteurs distincts ne sont pas censés produire les mêmes nombres.
- Sept seeds nouvelles, définies dans le programme : `101`, `1009`, `65537`, `0xFEDCBA9876543210`, `0x800000000000002D`, `0x0123456789ABCDEF`, `0xFFFFFFFFFFFFFFF1`.
- Compilation explicite de toutes les opérations d'un volume avant sa mesure, deux appels de chauffe par seed/opération, trois blocs de trois répétitions par seed.
- Ordre mélangé de façon déterministe dans chaque répétition. Les mêmes seeds et blocs apparient les comparaisons ; ils ne constituent pas sept sessions matérielles indépendantes.
- Temps mural de l'appel jusqu'à `block_until_ready`, allocation de sortie incluse. Compilation, préparation des clés, transfert du résultat vers l'hôte et contrôles exclus. Les arguments restent dynamiques pour éviter de pré-calculer un résultat constant.
- Tableaux entiers complets vérifiés avant les mesures : Montmory contre la référence C ; Threefry Pallas contre l'API JAX. Les API tierces servent de références de leur famille, sans oracle indépendant supplémentaire.
- Toutes les sorties chronométrées contrôlées, formes et types compris. Uniformes : égalité exacte avec le calcul NumPy. Box–Muller et espérance : référence NumPy float64, tolérances annoncées `atol=rtol=3e-5`. Normales JAX natives : référence JAX hors chronométrage ; leur réduction est recalculée en float64.
- Rapport JSON avec environnement, commit et état de travail, empreintes des sources, compteurs de vérification et estimations ; CSV contenant chaque mesure, y compris ordre, seed, bloc et erreur maximale.

La reproductibilité interarchitecture promise par le projet concerne le **flux entier**. Les tolérances flottantes de ce programme ne sont pas une promesse de normales ou de réductions bit à bit identiques sur CPU/GPU/TPU. La conformité de ce benchmark ne remplace ni TestU01 ni TACM.

## Lecture des résultats

Le temps principal est la médiane des médianes par seed. Le débit applicatif est donné en **millions de nombres traités par seconde** ; le Go/s de sortie n'est calculé que pour les deux opérations qui matérialisent un tableau. Aucun débit mémoire fictif n'est attribué à la sortie scalaire.

Un ratio compare les temps médians **appariés par seed**, puis en prend la médiane : `temps référence / temps candidat`. Plus de 1 signifie un candidat plus rapide. Le rapport donne aussi le nombre de seeds favorables. Les comparaisons sont descriptives, sans test de supériorité automatique et sans sélection après observation.

L'opération principale pour un futur argument applicatif est **`gaussian_expectation` à 16 777 216 mots**, avec comparaison Montmory Pallas contre Threefry Pallas et contre `jax.random.normal` natif. Les tailles et autres opérations restent des résultats secondaires à présenter, régressions comprises. Une confirmation dans une nouvelle session matérielle devra précéder une promesse commerciale. Le gain de ce cas ne se transpose pas automatiquement au modèle du client.

## Exécution

Depuis la racine du dépôt, dans l'environnement JAX épinglé décrit dans [reproduce.md](reproduce.md), sans changer les archives :

```bash
# Contrôle court sur VPS ou Mac CPU : 2 seeds, 2 répétitions, 1 volume.
.venv-jax/bin/python applications/jax_rng/benchmark.py \
  --backend cpu --smoke --out build/application-cpu-smoke

# Campagne sur une VM disposant effectivement d'un TPU.
.venv-jax/bin/python applications/jax_rng/benchmark.py \
  --backend tpu --out build/application-tpu-v1
```

Le dossier de sortie doit être nouveau. Avec Colab, utiliser l'interpréteur où JAX est installé, par exemple `python`, et une copie autorisée du dépôt dans la VM. Un compilateur C est nécessaire pour la référence. Le programme n'installe aucune dépendance, ne crée aucune ressource cloud et ne transmet rien. Les campagnes complètes comportent 4 284 mesures TPU, ou 2 772 mesures CPU ; le mode court ne qualifie aucune performance TPU.

La [validation CPU locale conservée](../applications/jax_rng/validation/README.md) documente l'exécution de contrôle et ses limites. **Aucun résultat applicatif natif TPU n'est encore fourni.** Le benchmark CPU confirme que l'essai est exécutable, pas que Pallas accélère l'application sur TPU.

Référence de méthode : [mesurer du code JAX](https://docs.jax.dev/en/latest/benchmarking.html), notamment compilation et synchronisation des calculs asynchrones.
