"""Audit an imported TPU run without rerunning TPU workloads or loading its binaries."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    require(not out.is_relative_to(run), "Keep audit separate from received evidence")
    out.mkdir(parents=True, exist_ok=True)
    kit = Path(__file__).resolve().parent / "montmory-jax-tpu"
    manifest = read_json(kit / "manifest.json")
    for name, expected in manifest["sha256"].items():
        require(fingerprint(kit / name) == expected, f"Local kit mismatch: {name}")
    evidence = {str(p.relative_to(run)): fingerprint(p) for p in sorted(run.rglob("*")) if p.is_file()}
    status = read_json(run / "run-status.json")
    require(status == {"status": "pass", "backend_requested": "tpu", "completed":
            ["check_manifest.py", "verify_native.py", "verify_sharding.py", "benchmark.py"]}, "Runner incomplete")
    report = read_json(run / "benchmark/report.json")
    conf = read_json(run / "conformance/report.json")
    sharding = read_json(run / "sharding/report.json")
    for part in (report, conf, sharding):
        require(part["status"] == "pass" and part["backend"] == "tpu", "Backend/status mismatch")
    require(conf["native_pallas_tpu"] and not conf["interpreted_pallas"], "Pallas not native")
    require(conf["statistical_tests_run"] is False, "Unexpected statistical claim")
    conf_expected = set(itertools.product((0, 42, 2**64-1, 2**63+1),
                        (0, 4095, 8388607, 2**32-7, 2**63-7, 2**64-1-8192),
                        ("jax_portable", "jax_mulhi", "pallas_8", "pallas_16", "pallas_32")))
    conf_actual = [(c["seed"], c["base"], c["engine"]) for c in conf["cases"]]
    require(len(conf_actual) == len(conf_expected) and set(conf_actual) == conf_expected, "Conformance cases incomplete")
    require(all(c["words"] == 8193 for c in conf["cases"]), "Conformance size mismatch")
    require(conf["word_comparisons"] == 6400 + 2*8193*120 == 1972720, "Conformance count mismatch")
    shard_expected = set(itertools.product((1,), (42, 2**64-1), (4093, 8388607, 2**32-7, 2**64-1-32767)))
    shard_actual = [(c["devices"], c["seed"], c["base"]) for c in sharding["cases"]]
    require(len(shard_actual) == 8 and set(shard_actual) == shard_expected, "Sharding cases mismatch")
    require(all(c["words"] == 32768 for c in sharding["cases"]), "Sharding size mismatch")
    require(sharding["word_comparisons"] == 262144 and not sharding["split_fold_in_tested"], "Sharding count/scope mismatch")
    require(len(report["devices"]) == len(sharding["devices"]) == 1, "Unexpected device count")
    require(report["jax"] == report["jaxlib"] == conf["jax"] == "0.10.2", "Version mismatch")
    require(report["base"] == 4294963199 and report["logical_lanes"] == 2048 and report["draws_per_lane"] == 4096, "Stream geometry mismatch")
    sizes = (65536, 1048576)
    seeds = (0, 1, 42, 2**64-1, 2**63+1)
    engines = ("montmory_jax_portable", "montmory_jax_mulhi", "threefry2x32", "threefry4x32",
               "philox4x32", "philox2x32", "write_baseline_NOT_PRNG", "montmory_pallas_8x128",
               "montmory_pallas_16x128", "montmory_pallas_32x128", "rbg_same_device_only")
    require(report["sizes"] == list(sizes) and report["seeds"] == list(seeds) and report["repetitions"] == 7 and not report["quick"], "Protocol mismatch")
    rows = list(csv.DictReader((run / "benchmark/measurements.csv").open(newline="")))
    groups, hashes = defaultdict(list), defaultdict(set)
    keys, check_counts, checked_words = [], Counter(), Counter()
    refs = read_json(kit / "reference/expected-output-hashes.json")["sha256"]
    for row in rows:
        n, seed, rep, elapsed = (int(row[k]) for k in ("words", "seed", "repetition", "nanoseconds"))
        engine = row["engine"]
        require(row["backend"] == "tpu" and row["device_kind"] == report["device_kind"] and elapsed > 0, "Invalid measurement")
        keys.append((n, engine, seed, rep))
        groups[n, engine, seed].append(elapsed)
        hashes[n, engine, seed].add(row["output_sha256"])
        check_counts[row["check"]] += 1
        checked_words[row["check"]] += n
        if engine == "rbg_same_device_only":
            require(row["check"] == "same_device_repeatability", "RBG scope mismatch")
        else:
            family = "montmory" if engine.startswith("montmory_") else engine
            require(row["check"] == "fixed_cpu_fingerprint", "Fingerprint scope mismatch")
            require(row["output_sha256"] == refs[f"{family}/{n}/{seed}"], "Output fingerprint mismatch")
    expected_keys = set(itertools.product(sizes, engines, seeds, range(7)))
    require(len(keys) == len(expected_keys) and set(keys) == expected_keys, "Missing/duplicate measurement")
    require(all(len(v) == 1 for v in hashes.values()), "Non-repeatable output")
    require(report["samples"] == len(rows) == 770, "Sample count mismatch")
    require(report["checked_timed_words"] == sum(checked_words.values()) == 428933120, "Word count mismatch")
    # Reproduce the prespecified randomized order without relying on the summaries.
    order_expected = []
    for n in sizes:
        for i, seed in enumerate(seeds):
            rng = random.Random(20261021+n+i)
            for rep in range(7):
                order = list(engines)
                rng.shuffle(order)
                order_expected.extend((n, e, seed, rep) for e in order)
    require(keys == order_expected, "Acquisition order differs from frozen protocol")
    medians = {key: st.median(values) for key, values in groups.items()}
    recomputed = []
    for n, engine in itertools.product(sizes, engines):
        times = [medians[n, engine, seed] for seed in seeds]
        base_times = [medians[n, "montmory_jax_portable", seed] for seed in seeds]
        recomputed.append({"engine": engine, "words": n, "median_ns_per_word": st.median(times)/n,
                           "stored_GB_per_second": 8*n/st.median(times),
                           "paired_ratio_vs_montmory_jax_portable": st.median(a/b for a, b in zip(base_times, times)),
                           "faster_seeds": sum(b < a for a, b in zip(base_times, times)), "seed_count": 5})
    require(len(report["summary"]) == len(recomputed), "Summary size mismatch")
    for actual, expected in zip(report["summary"], recomputed):
        require(actual.keys() == expected.keys(), "Summary fields mismatch")
        for key, value in expected.items():
            equal = math.isclose(actual[key], value, rel_tol=1e-12) if isinstance(value, float) else actual[key] == value
            require(equal, f"Summary mismatch: {expected['engine']}/{expected['words']}/{key}")
    # Rebuild only the trusted, hash-checked local C reference; never load uploaded .so files.
    sys.path.insert(0, str(kit))
    from support import Oracle
    import numpy as np
    oracle = Oracle(out / "local-reference")
    fresh_hashes = {}
    for n, seed in itertools.product(sizes, seeds):
        value = hashlib.sha256(np.asarray(oracle.range(seed, report["base"], n), dtype="<u8").tobytes()).hexdigest()
        key = f"montmory/{n}/{seed}"
        require(value == refs[key], f"Fresh C reference mismatch: {key}")
        fresh_hashes[key] = value
    pairs = []
    for n in sizes:
        for numerator in ("montmory_pallas_32x128", "montmory_jax_portable"):
            for denominator in engines:
                ratios = [medians[n, denominator, seed]/medians[n, numerator, seed] for seed in seeds]
                pairs.append({"words": n, "candidate": numerator, "reference": denominator,
                              "median_paired_speedup": st.median(ratios),
                              "gain_percent": 100*(st.median(ratios)-1),
                              "faster_seeds": sum(r > 1 for r in ratios),
                              "per_seed_speedups": dict(zip(map(str, seeds), ratios)),
                              "min_seed_speedup": min(ratios), "max_seed_speedup": max(ratios)})
    seed_stats = [{"words": n, "engine": e, "seed": seed, "median_ns": medians[n, e, seed],
                   "min_ns": min(v), "max_ns": max(v),
                   "relative_mad": st.median(abs(x-medians[n, e, seed]) for x in v)/medians[n, e, seed]}
                  for (n, e, seed), v in groups.items()]
    coverage = []
    for n in sizes:
        first_lane_index = report["base"] // 4096
        last_lane_index = (report["base"] + n - 1) // 4096
        lanes = sorted({i % 2048 for i in range(first_lane_index, last_lane_index + 1)})
        coverage.append({"words": n, "logical_lanes_touched": len(lanes), "lane_ids": lanes})
    audit = {"status": "pass", "run_directory": str(run), "device_kind": report["device_kind"],
             "received_file_sha256": evidence, "local_manifest_files_checked": len(manifest["sha256"]),
             "samples": len(rows), "checked_timed_words": sum(checked_words.values()),
             "checks_by_scope": dict(check_counts), "words_by_scope": dict(checked_words),
             "conformance_comparisons_reported": conf["word_comparisons"],
             "sharding_comparisons_reported": sharding["word_comparisons"],
             "summary_rows_recomputed": len(recomputed), "randomized_order_verified": True,
             "fresh_local_C_hashes": fresh_hashes, "summary": recomputed,
             "paired_comparisons": pairs, "per_seed_timings": seed_stats,
             "benchmark_lane_coverage": coverage,
             "distinct_montmory_seed_index_pairs_in_benchmark": len(seeds)*max(sizes),
             "limits": ["Independent audit of recorded data, not an independent TPU rerun.",
                        "Five seeds in one session; repeated samples are not independent TPU sessions.",
                        "Pallas 32x128 selected from these measurements; fresh confirmation remains required.",
                        "Only one TPU device; no multi-device performance validation.",
                        "No new statistical battery or custom split/fold_in validation.",
                        "No device-only profiling; host dispatch/allocation/wait included."]}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("status", "device_kind", "samples", "checked_timed_words", "checks_by_scope", "words_by_scope", "summary_rows_recomputed")}, indent=2))
    for item in pairs:
        if item["words"] == 1048576 and item["candidate"] == "montmory_pallas_32x128":
            print(f"Pallas 32x128 / {item['reference']}: {item['median_paired_speedup']:.6f}, "
                  f"{item['gain_percent']:+.3f}%, {item['faster_seeds']}/5, "
                  f"range {item['min_seed_speedup']:.4f}..{item['max_seed_speedup']:.4f}")


if __name__ == "__main__":
    main()
