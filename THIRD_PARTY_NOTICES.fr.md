# Droits et notices des tiers

[Français](THIRD_PARTY_NOTICES.fr.md) · [English](THIRD_PARTY_NOTICES.md) · [Documentation FR/EN](docs/languages.md)

Ce dépôt conserve des travaux de recherche attribués à Pascal Montmory et les composants tiers identifiés ci-dessous. Il n'introduit **aucune** licence MIT, Apache ou autre licence générale de redistribution pour le code, les données ou la documentation Montmory d'origine. La visibilité et l'accès au dépôt ne constituent pas une nouvelle concession de licence.

Les octets des sources figées et les notices des tiers restent inchangés. La réserve de droits sur les contributions originales figure dans [LICENSE](LICENSE.fr.md).

Les éléments tiers conservent leurs conditions originales, indépendamment du statut du dépôt qui les entoure :

- **Random123**, copyright D. E. Shaw Research. La notice originale figure dans chaque copie intégrée et dans [third_party/Random123-LICENSE](third_party/Random123-LICENSE). Le commit amont figé et les empreintes des fichiers figurent dans [le manifeste de provenance](studies/audit-2026-10-21/evidence/random123-upstream-manifest.json).
- **TestU01 1.2.3**, copyright Pierre L'Ecuyer. Les ZIP des sources originales inclus dans les kits de test figés conservent leurs notices. Le [fichier COPYING original](third_party/TestU01-COPYING) est également fourni séparément. Ses conditions distinguent les usages privés, de recherche, universitaires et non commerciaux des usages commerciaux ; ce dépôt ne remplace pas ces conditions.
- **xoshiro256\*\***, David Blackman et Sebastiano Vigna. La source conserve sa déclaration de mise dans le domaine public, sa notice d'autorisation et son exclusion de garantie dans [xoshiro256starstar.c](studies/tacm-comparison-2026-10-21/deps/xoshiro256starstar.c).
- **JAX, NumPy, SciPy, pandas, matplotlib et openpyxl** sont des dépendances externes, et non des installations embarquées. Consulter leurs distributions amont pour leurs conditions de licence respectives. JAX est un framework externe utilisé par le code TPU archivé ; il ne fait pas l'objet d'une revendication de propriété au titre des dépendances de Montmory.

Les noms de fichiers historiques contenant « Certified », « official » ou « DXSM » sont conservés pour la provenance. Ils n'établissent ni une certification externe, ni un statut officiel des vecteurs Montmory, ni une identité avec l'algorithme standard PCG64DXSM. « Official vectors » dans les rapports de comparaison désigne les vecteurs amont des comparateurs cités.

Les notices originales liées ci-dessus restent les textes de référence pour leurs composants ; cette traduction n'en modifie pas les conditions.
