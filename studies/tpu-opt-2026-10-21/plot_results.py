"""Export the independently audited confirmation measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/audit-20261021T185102Z-tpu-opt"
report = json.loads((OUT / "audit.json").read_text())
data = {(r["words"], r["engine"]): r for r in report["summary"]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
labels = ["Montmory retenu", "Montmory précédent", "Threefry2 JAX", "Threefry2 Pallas",
          "Philox4 JAX", "RBG (même backend)", "Écriture (non PRNG)"]
colors = ["#087f5b", "#a4cdbf", "#6a99d0", "#23538a", "#8c71ad", "#cb873a", "#a7acb3"]
for ax, count, title in zip(axes.flat, (65536, 1048576, 8388608, 16777216),
                          ("512 Kio", "8 Mio", "64 Mio", "128 Mio")):
    decision = report["decisions"][str(count)]
    three = next(name for name in decision["retained_comparisons"] if name.startswith("threefry_pallas"))
    names = [decision["retained"], "pallas_original_32", "threefry2x32", three,
             "philox4x32", "rbg_same_backend", "write_NOT_PRNG"]
    values = [data[count, name]["GB_per_second"] for name in names]
    ax.barh(range(len(names)), values, color=colors, height=.66)
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values)*1.18)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.18)
    ax.set_xlabel("Débit de sortie (Go/s décimaux)")
    ax.set_title(title + " · " + decision["retained"], fontsize=11, loc="left", pad=12)
    for i, value in enumerate(values):
        ax.text(value+max(values)*.018, i, f"{value:.2f}", va="center", fontsize=9)
fig.suptitle("Montmory : gains confirmés sur les grands blocs\nThreefry Pallas reste légèrement plus rapide à 64 et 128 Mio",
             x=.03, ha="left", fontsize=15, fontweight="bold", y=.98)
fig.text(.03, .026, "TPU v5 lite · JAX 0.10.2 · 7 seeds de confirmation · 9 mesures par seed et configuration\n"
         "Débit calculé à partir de la médiane des temps médians par seed. Temps appelant, allocation et attente inclus ; compilation et transferts exclus.\n"
         "Une session, un périphérique. Les gains appariés du rapport ne sont pas les quotients des débits affichés.", fontsize=9, color="#454b54")
fig.tight_layout(rect=(.015, .095, .995, .90), h_pad=3, w_pad=3)
fig.savefig(OUT / "confirmation-tpu.png", dpi=180)
fig.savefig(OUT / "confirmation-tpu.pdf")
plt.close(fig)
print(OUT / "confirmation-tpu.png")
