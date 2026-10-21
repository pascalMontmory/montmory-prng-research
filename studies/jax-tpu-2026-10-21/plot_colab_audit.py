"""Export recorded per-seed comparisons; whiskers are not confidence intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("audit", type=Path)
args = parser.parse_args()
audit = json.loads(args.audit.read_text())
references = [
    ("threefry2x32", "Threefry2x32"),
    ("threefry4x32", "Threefry4x32"),
    ("philox2x32", "Philox2x32"),
    ("philox4x32", "Philox4x32"),
    ("montmory_jax_portable", "Montmory JAX portable"),
    ("rbg_same_device_only", "RBG (même backend)"),
    ("write_baseline_NOT_PRNG", "Témoin d’écriture"),
]
fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
for ax, n, title in zip(axes, (65536, 1048576), ("65 536 mots · 512 Kio", "1 048 576 mots · 8 Mio")):
    ax.axvline(1, color="#555555", linewidth=1, linestyle="--")
    for i, (engine, _) in enumerate(references):
        item = next(p for p in audit["paired_comparisons"] if p["words"] == n
                    and p["candidate"] == "montmory_pallas_32x128" and p["reference"] == engine)
        values = list(item["per_seed_speedups"].values())
        ax.plot([min(values), max(values)], [i, i], color="#b5b5b5", linewidth=2, zorder=1)
        for j, (value, color) in enumerate(zip(values, colors)):
            ax.scatter(value, i+(j-2)*0.055, s=25, color=color, alpha=.85, zorder=2)
        median = item["median_paired_speedup"]
        ax.scatter(median, i, s=65, marker="D", color="#161616", zorder=3)
        ax.text(1.02, i, f"{item['gain_percent']:+.1f} % · {item['faster_seeds']}/5",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set_title(title, fontsize=12, pad=15)
    ax.set_yticks(range(len(references)), [label for _, label in references])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Temps de la référence / temps Montmory", labelpad=12)
axes[0].invert_yaxis()
axes[0].set_xlim(.94, 1.36)
axes[1].set_xlim(.78, 2.3)
fig.suptitle("Montmory Pallas 32×128 — comparaisons seed par seed", fontsize=16, y=.98)
fig.text(.5, .91, "TPU v5 lite · JAX 0.10.2 · 7 mesures par seed · une seule session", ha="center", fontsize=11)
fig.text(.02, .065, "Points colorés : 5 seeds. Losange noir : médiane des 5 ratios. Trait gris : minimum–maximum, pas un intervalle de confiance.", fontsize=9)
fig.text(.02, .035, "À droite de 1 : Montmory plus rapide. Étiquette : gain de débit apparié et nombre de seeds gagnantes sur 5.", fontsize=9)
fig.text(.02, .005, "Chronométrage côté appelant avec attente de fin ; compilation, préparation des clés et transferts de contrôle exclus.", fontsize=9)
fig.subplots_adjust(left=.18, right=.86, bottom=.19, top=.82, wspace=.60)
for extension in ("png", "pdf"):
    destination = args.audit.parent / f"comparaison-tpu.{extension}"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    print(destination)
