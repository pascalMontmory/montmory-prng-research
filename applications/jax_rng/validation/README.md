# Validation du nouveau benchmark

[Français](README.md) · [English](README.en.md) · [Documentation FR/EN](../../../docs/languages.md)

Le [rapport CPU](cpu-smoke/report.json) et les [44 mesures](cpu-smoke/measurements.csv) proviennent d'une exécution locale réussie sur le VPS : un seul volume, deux seeds et deux répétitions. Le programme a comparé 393 216 mots entiers aux références de leur famille et vérifié les 44 sorties chronométrées. Pour Montmory, la référence entière est le code C ; les API tierces constituent les références de leurs familles.

Ce contrôle court vérifie l'exécution, la conformité entière et les sorties applicatives. Il ne mesure pas les kernels Pallas sur TPU, ne relance pas de batterie statistique et n'établit aucun gain commercial. Les fichiers de référence restent dans les archives inchangées.

Le [contrôle complémentaire](check-report.json) a vérifié les bornes des uniformes, le rejet de trois sorties volontairement corrompues, 18 cas applicatifs avec Pallas interprété sur CPU et le calcul des ratios appariés. Cette interprétation n'est pas une exécution native TPU. Il peut être rejoué depuis la racine avec l'interpréteur JAX épinglé :

```bash
.venv-jax/bin/python applications/jax_rng/check.py
```

Les empreintes enregistrées dans le rapport identifient les sources effectivement exécutées. Un commit de base accompagné de `git_dirty=true` signifie que le nouveau benchmark était encore en cours d'ajout ; les empreintes précisent alors son état exact.
