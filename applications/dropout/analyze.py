#!/usr/bin/env python3
"""Audit the uploaded TPU run without rerunning training or changing its sources."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import argparse
import io
import tarfile
import json
import math
import statistics as stats

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=ROOT / "build/dropout-tpu-audit")
parser.add_argument("--plots", action="store_true", help="Also regenerate figures; requires Matplotlib")
args = parser.parse_args()
BASE = ROOT / "applications/dropout/validation/tpu-20261024"
RAW = BASE
OUT = args.out.resolve()
if OUT == BASE or BASE in OUT.parents:
    parser.error("write the new audit outside the retained evidence directory")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, records):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = json.loads((RAW / "report.json").read_text())
manifest = json.loads((RAW / "manifest.json").read_text())
steps = read_csv(RAW / "steps.csv")
curves = read_csv(RAW / "learning_curves.csv")
seeds = report["protocol"]["seeds"]
nsteps = report["protocol"]["steps"]
variants = ["threefry_jax", "montmory_jax", "threefry_pallas", "montmory_pallas"]
checks = {}


def check(name, condition):
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


check("successful_native_tpu_run", report["status"] == "pass" and report["requested_backend"] == "tpu"
      and report["native_tpu_measurements"] and report["verification"]["mode"] == "native_tpu")
check("manifest_contains_report_sources", all(manifest.get(p) == h for p, h in report["source_sha256"].items()))
# The received results and the original source kit are retained byte for byte.
for line in (BASE / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split("  ", 1)
    path = (BASE / name).resolve()
    check("safe_evidence_path:" + name, path.is_relative_to(BASE))
    check("retained_evidence_hash:" + name, digest(path) == expected)
with tarfile.open(BASE / "dropout-tpu.tar.gz", "r:gz") as tar:
    received = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    expected_members = {"dropout-tpu/" + n: n for n in ("report.json", "steps.csv", "learning_curves.csv")}
    expected_members.update({"run.log": "run.log", "manifest.json": "manifest.json"})
    check("complete_received_archive", set(received) == set(expected_members))
    for archived, retained in expected_members.items():
        check("unmodified_received_bytes:" + retained, received[archived] == (BASE / retained).read_bytes())
with tarfile.open(BASE / "source-kit.tar.gz", "r:gz") as tar:
    sources = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
check("complete_source_kit", set(sources) == set(manifest) | {"manifest.json"})
check("source_kit_manifest", json.loads(sources["manifest.json"]) == manifest)
source_checks = {name: hashlib.sha256(sources[name]).hexdigest() == h for name, h in manifest.items()}
check("all_manifest_hashes_verified", all(source_checks.values()))
for name, expected in report["source_sha256"].items():
    check("current_execution_source:" + name, digest(ROOT / name) == expected)

records = {}
order_groups = {}
for row in steps:
    key = (int(row["seed"]), row["variant"], int(row["step"]))
    check("unique_step:" + str(key), key not in records)
    check("valid_values:" + str(key), int(row["elapsed_ns"]) > 0 and math.isfinite(float(row["loss"]))
          and row["timing_scope"] == "native_training_step")
    records[key] = row
    order_groups.setdefault((key[0], key[2]), []).append(int(row["order"]))
expected_keys = {(s, v, n) for s in seeds for v in variants for n in range(nsteps)}
check("complete_training_grid", set(records) == expected_keys)
check("order_is_permutation_at_every_step", all(sorted(order) == list(range(4)) for order in order_groups.values()))
summaries = {(r["seed"], r["variant"]): r for r in report["training"]}
check("complete_report_training_grid", len(summaries) == len(report["training"]) == len(seeds) * len(variants)
      and set(summaries) == {(s, v) for s in seeds for v in variants})

timings = []
for seed in seeds:
    check("same_initial_validation:" + str(seed), len({(summaries[seed, v]["initial_validation_loss"],
          summaries[seed, v]["initial_validation_accuracy"]) for v in variants}) == 1)
    for variant in variants:
        times = [int(records[seed, variant, n]["elapsed_ns"]) / 1000 for n in range(nsteps)]
        median = stats.median(times)
        check("reported_median:" + str((seed, variant)), math.isclose(median / 1000,
              summaries[seed, variant]["median_step_ms"], rel_tol=1e-12, abs_tol=1e-12))
        timings.append(dict(seed=seed, variant=variant, count=len(times), median_us=median,
                            mean_us=stats.mean(times), p10_us=float(np.percentile(times, 10)),
                            p90_us=float(np.percentile(times, 90)), min_us=min(times), max_us=max(times)))
time_by = {(r["seed"], r["variant"]): r for r in timings}

trajectory_checks = []
for seed in seeds:
    for engine in ("threefry", "montmory"):
        diffs = [abs(float(records[seed, engine + "_jax", n]["loss"]) -
                     float(records[seed, engine + "_pallas", n]["loss"])) for n in range(nsteps)]
        check("equal_recorded_training_losses:" + str((seed, engine)), max(diffs) == 0)
        trajectory_checks.append(dict(seed=seed, engine=engine, comparisons=len(diffs), max_loss_difference=max(diffs)))
curve_by = {(int(r["seed"]), r["variant"], int(r["steps"])): r for r in curves}
check("complete_validation_grid", len(curves) == len(curve_by) == len(seeds) * 4 * 3
      and set(curve_by) == {(s, v, n) for s in seeds for v in variants for n in (50, 100, 150)})
for seed in seeds:
    for engine in ("threefry", "montmory"):
        for n in (50, 100, 150):
            a, b = (curve_by[seed, engine + "_" + v, n] for v in ("jax", "pallas"))
            check("equal_recorded_validation:" + str((seed, engine, n)), all(float(a[k]) == float(b[k])
                  for k in ("validation_loss", "validation_accuracy")))
    for variant in variants:
        a, b = curve_by[seed, variant, nsteps], summaries[seed, variant]
        check("reported_final_validation:" + str((seed, variant)), all(float(a[k]) == b["final_" + k]
              for k in ("validation_loss", "validation_accuracy")))
check("zero_weight_differences_reported", len(report["trained_equivalence"]) == 6
      and all(r["max_weight_difference"] == 0 for r in report["trained_equivalence"]))

# Reconstruct the dataset and batch order independently, without importing JAX.
rng = np.random.Generator(np.random.PCG64(20261021))
teacher = rng.standard_normal((32, 4)).astype(np.float32)
x = rng.standard_normal((2560, 32)).astype(np.float32)
y = (x @ teacher).argmax(axis=1).astype(np.int32)
batch_rng = np.random.Generator(np.random.PCG64(123456))
batch_indices = batch_rng.integers(0, 2048, (nsteps, report["protocol"]["batch"]), dtype=np.int32)
data_hash = hashlib.sha256()
for array in (x[:2048], y[:2048], x[2048:], y[2048:], batch_indices):
    data_hash.update(array.tobytes())
check("reconstructed_dataset_and_batch_order_hash", data_hash.hexdigest() == report["dataset_and_batch_order_sha256"])

comparisons = []
for seed in seeds:
    for candidate, baseline in [("montmory_pallas", "montmory_jax"), ("threefry_pallas", "threefry_jax"),
                                ("montmory_pallas", "threefry_jax"), ("montmory_pallas", "threefry_pallas")]:
        a, b = time_by[seed, candidate], time_by[seed, baseline]
        differences = [(int(records[seed, baseline, n]["elapsed_ns"]) -
                        int(records[seed, candidate, n]["elapsed_ns"])) / 1000 for n in range(nsteps)]
        comparisons.append(dict(seed=seed, candidate=candidate, baseline=baseline,
                                baseline_median_over_candidate_median=b["median_us"] / a["median_us"],
                                median_time_reduction_percent=100 * (1 - a["median_us"] / b["median_us"]),
                                mean_time_reduction_percent=100 * (1 - a["mean_us"] / b["mean_us"]),
                                median_paired_saving_us=stats.median(differences),
                                steps_candidate_faster=sum(d > 0 for d in differences)))
for item in report["performance_comparisons"]:
    ratios = [r["baseline_median_over_candidate_median"] for r in comparisons
              if r["candidate"] == item["engine"] + "_pallas" and r["baseline"] == item["engine"] + "_jax"]
    check("reported_ratios:" + item["engine"], all(math.isclose(a, b, rel_tol=1e-12)
          for a, b in zip(ratios, item["reference_step_time_over_pallas"], strict=True))
          and math.isclose(stats.median(ratios), item["median_ratio"], rel_tol=1e-12))

diagnostics = []
for seed in seeds:
    for variant in variants:
        for first, end in [(0, 50), (50, 100), (100, 150)]:
            vals = [int(records[seed, variant, n]["elapsed_ns"]) / 1000 for n in range(first, end)]
            diagnostics.append(dict(seed=seed, variant=variant, first_step=first, last_step=end - 1,
                                    median_us=stats.median(vals), mean_us=stats.mean(vals)))
order_counts = {f"{seed}:{variant}": dict(Counter(int(records[seed, variant, n]["order"])
                for n in range(nsteps))) for seed in seeds for variant in variants}
cpu_report = json.loads(sources["applications/dropout/validation/cpu/report.json"])
cpu_rows = list(csv.DictReader(io.StringIO(sources["applications/dropout/validation/cpu/steps.csv"].decode())))
check("cpu_same_execution_sources", cpu_report["source_sha256"] == report["source_sha256"])
check("cpu_same_dataset_and_batch_order", cpu_report["dataset_and_batch_order_sha256"] == report["dataset_and_batch_order_sha256"])
check("cpu_same_workload", all(cpu_report["protocol"][k] == report["protocol"][k]
      for k in ("steps", "seeds", "rows", "width", "batch")))
check("cpu_functional_timing_scope", len(cpu_rows) == 900 and all(r["timing_scope"] == "cpu_functional_run" for r in cpu_rows))
cpu_keys = [(int(r["seed"]), r["variant"], int(r["step"])) for r in cpu_rows]
check("cpu_complete_training_grid", len(set(cpu_keys)) == len(cpu_keys) and set(cpu_keys) ==
      {(s, v, n) for s in seeds for v in ("threefry_jax", "montmory_jax") for n in range(nsteps)})
cpu_comparisons = []
for seed in seeds:
    cpu_ms = stats.median(int(r["elapsed_ns"]) / 1e6 for r in cpu_rows
                         if int(r["seed"]) == seed and r["variant"] == "montmory_jax")
    for variant in ("montmory_jax", "montmory_pallas"):
        tpu_ms = time_by[seed, variant]["median_us"] / 1000
        cpu_comparisons.append(dict(seed=seed, cpu_variant="montmory_jax", tpu_variant=variant,
                                    cpu_median_step_ms=cpu_ms, tpu_median_step_ms=tpu_ms,
                                    cpu_over_tpu_ratio=cpu_ms / tpu_ms,
                                    time_reduction_percent=100 * (1 - tpu_ms / cpu_ms),
                                    scope="indicative_only_separate_cpu_functional_session"))
write_csv("cpu_comparisons.csv", cpu_comparisons)
write_csv("timing_summary.csv", timings)
write_csv("comparisons.csv", comparisons)
write_csv("timing_by_block.csv", diagnostics)
audit = dict(status="pass", archive_sha256=digest(BASE / "dropout-tpu.tar.gz"),
             original_started_utc=report["started_utc"], manifest_files_verified=len(source_checks),
             execution_sources_verified=len(report["source_sha256"]),
             checks_passed=sum(checks.values()), training_rows=len(steps), validation_rows=len(curves),
             source_checks=source_checks, dataset_and_batch_order_sha256=data_hash.hexdigest(),
             training_loss_pairs=trajectory_checks, recorded_validation_pairs=18,
             final_weight_verification="Zero difference recorded by the runner; weight tensors were not included in the archive.",
             timing_summary=timings, comparisons=comparisons, order_counts=order_counts,
             cpu_comparisons=cpu_comparisons, gpu_training_measured=False,
             source_kit_sha256=digest(BASE / "source-kit.tar.gz"),
             limitations=["One TPU session; three training seeds are not independent benchmark sessions.",
                          "Wall time includes dispatch and synchronization; no device-only profile.",
                          "Compilation, dataset preparation, batch extraction and validation are outside timed steps.",
                          "Synthetic task, 512 validation examples, 150 updates; no statistical equivalence test.",
                          "CPU ratios are indicative: a separate functional session, not a controlled hardware benchmark.",
                          "No memory, energy, large-model or distributed-training measurements."])
(OUT / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")

if args.plots:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    colors = {"threefry_jax": "#59728c", "threefry_pallas": "#a4bdd2",
              "montmory_jax": "#006f65", "montmory_pallas": "#62b9a1"}
    labels = {v: v.split("_")[0].capitalize() + " · " + ("JAX" if v.endswith("_jax") else "Pallas") for v in variants}
    for i, variant in enumerate(variants):
        axes[0, 0].bar(np.arange(3) + (i - 1.5) * .19,
                       [time_by[s, variant]["median_us"] for s in seeds], width=.18,
                       label=labels[variant], color=colors[variant])
    axes[0, 0].set(xticks=range(3), xticklabels=seeds, xlabel="Graine", ylabel="Temps médian par étape (µs)",
                   ylim=(0, 370), title="Étape complète d’entraînement")
    axes[0, 0].legend(fontsize=8, ncol=2, loc="upper right")
    for i, engine in enumerate(("threefry", "montmory")):
        reductions = [r["median_time_reduction_percent"] for r in comparisons
                      if r["candidate"] == engine + "_pallas" and r["baseline"] == engine + "_jax"]
        bars = axes[0, 1].bar(np.arange(3) + (i - .5) * .3, reductions, width=.28,
                              color=colors[engine + "_jax"], label=engine.capitalize())
        axes[0, 1].bar_label(bars, labels=[f"{v:+.2f} %" for v in reductions], padding=4, fontsize=9)
    axes[0, 1].axhline(0, color="#555", lw=.8)
    axes[0, 1].set(xticks=range(3), xticklabels=seeds, xlabel="Graine", ylabel="Réduction du temps (%)",
                   ylim=(-1.5, 7.5), title="Pallas comparé au JAX du même générateur")
    axes[0, 1].legend(loc="upper left", fontsize=8, ncol=2)
    for col, (metric, scale, ylabel) in enumerate([("validation_loss", 1, "Perte de validation"),
                                                ("validation_accuracy", 100, "Précision de validation (%)")]):
        ax = axes[1, col]
        for engine in ("threefry", "montmory"):
            values = np.array([[summaries[s, engine + "_jax"]["initial_" + metric]] +
                               [float(curve_by[s, engine + "_jax", n][metric]) for n in (50, 100, 150)]
                               for s in seeds]) * scale
            ax.plot([0, 50, 100, 150], values.mean(axis=0), "o-", color=colors[engine + "_jax"], label=engine.capitalize())
            ax.fill_between([0, 50, 100, 150], values.min(axis=0), values.max(axis=0),
                             color=colors[engine + "_jax"], alpha=.12)
        ax.set(xlabel="Mises à jour", ylabel=ylabel, xticks=[0, 50, 100, 150], title=ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Dropout sur TPU : conformité et entraînement du petit MLP", fontsize=16, fontweight="bold", y=.99)
    fig.text(.5, .944, "TPU v5 lite · 21 252 paramètres · lot 64 · largeur 128 · dropout 25 %", ha="center", fontsize=10)
    fig.text(.5, .015, "Une session, trois graines. Courbes : moyenne et étendue min–max, sans intervalle de confiance.\n"
             "JAX et Pallas ont les mêmes pertes et précisions enregistrées pour chaque générateur.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .92), h_pad=2.3, w_pad=3)
    fig.savefig(OUT / "dropout-tpu-analysis.png", dpi=180)
    fig.savefig(OUT / "dropout-tpu-analysis.pdf")
    plt.close(fig)

print(json.dumps({"status": "pass", "manifest_files_verified": len(manifest),
                  "training_rows": len(steps), "loss_pairs_identical": 900,
                  "cpu_comparison_scope": "indicative only", "output": str(OUT)}, indent=2))
