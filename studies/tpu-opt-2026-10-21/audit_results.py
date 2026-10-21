"""Audit received native results without executing any received binary or script."""
import argparse
from collections import Counter, defaultdict
import csv
import ctypes
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
KIT = ROOT / "montmory-tpu-opt"
SIZES = (65536, 1048576, 8388608, 16777216)
SEARCH = (0, 42, 2**64 - 1)
CONFIRM = (7, 19, 20261021, 0x13579BDF2468ACE0, 0xDEADBEEF01234567,
           2**64 - 4, 2**63 + 12345)
BASE = 2**32 - 4097
BASELINE = "pallas_original_32"
VENDORS = ("threefry2x32", "threefry4x32", "philox2x32", "philox4x32")
MONT = ("jax_original", "jax_mulhi", "jax_carry", "jax_carry_mulhi") + tuple(
    f"pallas_{method}_{r}" for r in (8, 16, 32, 64, 128, 256)
    for method in ("original", "carry"))
THREE = tuple(f"threefry_pallas_{r}" for r in (16, 32, 64, 128))
NAMES = (*MONT, *THREE, *VENDORS, "write_NOT_PRNG", "rbg_same_backend")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(name):
    if name in MONT:
        return "montmory"
    if name in THREE:
        return "threefry2x32"
    return name


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    out = ROOT / "results" / ("audit-" + source.name)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python3", str(KIT / "check_manifest.py")], cwd=KIT, check=True)
    status = read(source / "run-status.json")
    report = read(source / "campaign/report.json")
    frozen = read(source / "campaign/frozen-selection.json")
    old = read(source / "historical-conformance/report.json")
    native = read(source / "campaign/candidate-conformance/report.json")
    refs = read(KIT / "reference/optimization-hashes.json")
    assert status["status"] == report["status"] == old["status"] == native["status"] == "pass"
    assert status["completed"] == ["check_manifest.py", "verify_native.py", "campaign.py"]
    assert not status["cpu_smoke"] and not report["cpu_smoke"]
    assert report["backend_requested"] == report["backend"] == native["backend"] == old["backend"] == "tpu"
    assert report["jax"] == report["jaxlib"] == refs["jax"] == refs["jaxlib"] == "0.10.2"
    assert report["sizes"] == refs["sizes"] == list(SIZES)
    assert report["search_seeds"] == refs["search_seeds"] == list(SEARCH)
    assert report["confirm_seeds"] == refs["confirm_seeds"] == list(CONFIRM)
    assert not set(SEARCH) & set(CONFIRM)
    assert report["base"] == refs["base"] == BASE
    assert (report["logical_lanes"], report["draws_per_lane"]) == (2048, 4096)
    assert report["baseline"] == BASELINE
    assert report["confirmation_blocks"] == report["confirmation_repetitions_per_block"] == 3
    assert len(report["devices"]) == 1 and report["devices"] == old["devices"]
    assert old["word_comparisons"] == 1972720 and not old["statistical_tests_run"]
    assert old["native_pallas_tpu"] and not old["interpreted_pallas"]
    assert len(old["cases"]) == 4 * 6 * 5
    assert not native["interpreted_pallas"] and not native["compile_rejections"]
    assert native["words_compared"] == 1684047 and native["component_values"] == 1048592
    assert native["candidates"] == list((*MONT, *THREE))
    assert len(report["compilation"]) == 104
    assert {(v["words"], v["engine"]) for v in report["compilation"]} == set(itertools.product(SIZES, NAMES))
    assert all(v["status"] == "pass" and v["seconds"] >= 0 for v in report["compilation"])
    assert sha(source / "campaign/frozen-selection.json") == report["selection_sha256"]

    with (source / "campaign/measurements.csv").open() as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in ("words", "seed", "block", "repetition", "nanoseconds"):
            row[field] = int(row[field])
        assert row["nanoseconds"] > 0
        assert row["engine"] in NAMES
    assert len(rows) == report["samples"] == 3708
    assert sum(r["words"] for r in rows) == report["checked_words"]
    ids = [(r["phase"], r["words"], r["block"], r["seed"], r["repetition"], r["engine"]) for r in rows]
    assert len(set(ids)) == len(ids)
    expected_ids = []
    for count in SIZES:
        expected_ids.extend(itertools.product(("search",), (count,), (0,), SEARCH, range(3), NAMES))
        expected_ids.extend(itertools.product(("confirmation",), (count,), range(3), CONFIRM, range(3), frozen[str(count)]["finalists"]))
    assert set(expected_ids) == set(ids)
    phase_order = [r["phase"] for r in rows]
    assert phase_order == ["search"] * 936 + ["confirmation"] * 2772

    # Check logged order against the predetermined randomized schedule.
    expected_order = []
    for phase, seeds, blocks in (("search", SEARCH, 1), ("confirmation", CONFIRM, 3)):
        for count in SIZES:
            names = list(NAMES if phase == "search" else frozen[str(count)]["finalists"])
            for block in range(blocks):
                for si, seed in enumerate(seeds):
                    rng = random.Random(20261021 + count + block * 1009 + si + (0 if phase == "search" else 1000003))
                    for repetition in range(3):
                        order = names.copy()
                        rng.shuffle(order)
                        expected_order.extend((phase, count, block, seed, repetition, name) for name in order)
    assert ids == expected_order
    totals = defaultdict(lambda: {"samples": 0, "words": 0})
    rbg_hashes = defaultdict(set)
    used_references = set()
    for row in rows:
        name = row["engine"]
        group = "rbg_same_backend" if name == "rbg_same_backend" else "fixed_cpu_reference"
        totals[group]["samples"] += 1
        totals[group]["words"] += row["words"]
        if name in MONT:
            totals["montmory"]["samples"] += 1
            totals["montmory"]["words"] += row["words"]
        if name == "rbg_same_backend":
            assert row["check"] == "same_backend_repeatability"
            rbg_hashes[row["words"], row["seed"]].add(row["output_sha256"])
        else:
            assert row["check"] == "fixed_cpu_reference"
            key = f"{family(name)}/{row['words']}/{row['seed']}"
            assert row["output_sha256"] == refs["sha256"][key], key
            used_references.add(key)
    assert len(used_references) == 240
    assert len(rbg_hashes) == 40 and all(len(v) == 1 for v in rbg_hashes.values())

    # Fresh compile from the frozen local C source. Uploaded .so files are never loaded.
    target = out / "oracle-rebuilt.so"
    subprocess.run(["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                    str(KIT / "reference/oracle.c"), "-o", str(target)], check=True)
    oracle = ctypes.CDLL(str(target))
    array = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    oracle.reference_words.argtypes = [ctypes.c_uint64, array, array, ctypes.c_size_t]
    oracle.reference_words.restype = None
    index = np.arange(max(SIZES), dtype=np.uint64) + np.uint64(BASE)
    output = np.empty(max(SIZES), dtype=np.uint64)
    regenerated = {}
    for seed in (*SEARCH, *CONFIRM):
        oracle.reference_words(seed, index, output, len(index))
        for count in SIZES:
            key = f"montmory/{count}/{seed}"
            digest = hashlib.sha256(output[:count].astype("<u8", copy=False).tobytes()).hexdigest()
            assert digest == refs["sha256"][key], key
            regenerated[key] = digest
        print(f"CPU reference rebuilt and checked: seed {seed}", flush=True)

    values = defaultdict(list)
    blocks = defaultdict(list)
    for row in rows:
        values[row["phase"], row["words"], row["engine"], row["seed"]].append(row["nanoseconds"])
        blocks[row["phase"], row["words"], row["engine"], row["seed"], row["block"]].append(row["nanoseconds"])
    def med(phase, count, name, seed):
        return st.median(values[phase, count, name, seed])
    def compare(count, name, reference):
        ratios = [med("confirmation", count, reference, s) / med("confirmation", count, name, s) for s in CONFIRM]
        block_ratios = [st.median(blocks["confirmation", count, reference, s, b]) /
                        st.median(blocks["confirmation", count, name, s, b]) for s in CONFIRM for b in range(3)]
        return dict(speedup=st.median(ratios), gain_percent=100*(st.median(ratios)-1),
                    elapsed_time_reduction_percent=100*(1-1/st.median(ratios)),
                    faster_seeds=sum(v > 1 for v in ratios), total_seeds=7,
                    min_ratio=min(ratios), max_ratio=max(ratios), ratios=ratios,
                    faster_seed_blocks=sum(v > 1 for v in block_ratios), total_seed_blocks=21,
                    min_block_ratio=min(block_ratios), max_block_ratio=max(block_ratios), block_ratios=block_ratios)
    summary_by_key = {(v["words"], v["engine"]): v for v in report["summary"]}
    assert len(summary_by_key) == len(report["summary"]) == 44
    summaries = []
    decisions = {}
    search_tables = {}
    for count in SIZES:
        def score(name):
            return st.median(med("search", count, name, s) for s in SEARCH)
        ranked_mont = sorted(MONT, key=score)
        ranked_three = sorted((*THREE, "threefry2x32"), key=score)
        frozen_row = frozen[str(count)]
        assert frozen_row["primary"] == ranked_mont[0] and frozen_row["secondary"] == ranked_mont[1]
        assert frozen_row["threefry"] == ranked_three[0]
        finalists = list(dict.fromkeys([*ranked_mont[:2], BASELINE, "jax_original", *VENDORS, ranked_three[0], "write_NOT_PRNG", "rbg_same_backend"]))
        assert frozen_row["finalists"] == finalists
        assert all(report["selection"][str(count)][k] == v for k, v in frozen_row.items())
        primary = frozen_row["primary"]
        primary_vs_old = compare(count, primary, BASELINE)
        adopt = primary if primary != BASELINE and primary_vs_old["speedup"] >= 1.03 and primary_vs_old["faster_seeds"] == 7 else BASELINE
        assert report["selection"][str(count)]["recommended_after_confirmation"] == adopt
        for name in finalists:
            times = [med("confirmation", count, name, s) for s in CONFIRM]
            gbps = 8 * count / st.median(times)
            paired = compare(count, primary, name)
            original = summary_by_key[count, name]
            close(original["GB_per_second"], gbps)
            close(original["primary_speedup_vs_engine"], paired["speedup"])
            assert original["primary"] == primary and original["seed_count"] == 7
            assert original["primary_faster_seeds"] == paired["faster_seeds"]
            for observed, expected in zip(original["ratios"], paired["ratios"], strict=True):
                close(observed, expected)
            mads = []
            for seed in CONFIRM:
                raw = values["confirmation", count, name, seed]
                middle = st.median(raw)
                mads.append(st.median(abs(v-middle) for v in raw)/middle)
            summaries.append(dict(words=count, engine=name, GB_per_second=gbps,
                median_microseconds=st.median(times)/1000,
                seed_min_microseconds=min(times)/1000, seed_max_microseconds=max(times)/1000,
                median_relative_MAD=st.median(mads), retained=name == adopt,
                seed_medians_ns=times,
                block_medians_ns=[st.median(st.median(blocks["confirmation", count, name, s, b]) for s in CONFIRM) for b in range(3)]))
        refs_to_compare = [BASELINE, "jax_original", "threefry2x32", frozen_row["threefry"],
                           "threefry4x32", "philox2x32", "philox4x32", "write_NOT_PRNG", "rbg_same_backend"]
        decisions[str(count)] = dict(primary=primary, secondary=frozen_row["secondary"],
            retained=adopt, primary_vs_previous=primary_vs_old,
            retained_comparisons={name:compare(count, adopt, name) for name in refs_to_compare})
        search_tables[str(count)] = [dict(engine=name, median_ns=score(name), GB_per_second=8*count/score(name))
                                    for name in sorted(NAMES, key=score)]

    input_paths = [p for p in source.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    audit = dict(status="pass", source=str(source), source_sha256={str(p.relative_to(source)):sha(p) for p in sorted(input_paths)},
        backend=report["backend"], device_kind=report["device_kind"], devices=report["devices"],
        jax=report["jax"], jaxlib=report["jaxlib"], native_historical_words=old["word_comparisons"],
        native_candidate_words=native["words_compared"], native_component_values=native["component_values"],
        compilation_entries_checked=104, samples_checked=len(rows), summaries_recomputed=len(summaries),
        phases=dict(Counter(r["phase"] for r in rows)), timed_output_checks=dict(totals),
        fixed_cpu_reference_keys_checked=len(used_references), montmory_hashes_rebuilt=len(regenerated),
        montmory_distinct_timed_seed_index_pairs=len((*SEARCH, *CONFIRM))*max(SIZES),
        selection_sha256=report["selection_sha256"], random_order_verified=True, adoption_rules_verified=True,
        rbg_repeated_hashes_consistent_across_blocks=True,
        summary=summaries, decisions=decisions, search=search_tables, rebuilt_montmory_hashes=regenerated,
        limits=["One TPU device and one session; held-out seeds share the same session.",
                "Output hashes verified against frozen CPU references; uploaded arrays are not retained.",
                "No new TestU01/TACM run, no simultaneous new native ARM/GPU run.",
                "Host dispatch/allocation/wait included; compilation, key setup, output transfers/checks excluded.",
                "Montmory large volumes cover all logical lanes, not an explicit physical-core affinity sweep.",
                "Warmup checks are reported by the runner but not present as individual CSV rows."])
    (out / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    fields = [k for k in summaries[0] if k not in ("seed_medians_ns", "block_medians_ns")]
    with (out / "confirmation-summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader();writer.writerows(summaries)
    print(json.dumps({k:v for k,v in audit.items() if k in ("status", "samples_checked", "summaries_recomputed", "timed_output_checks", "montmory_hashes_rebuilt", "montmory_distinct_timed_seed_index_pairs")}, indent=2))
    print("Audit:", out / "audit.json")


if __name__ == "__main__":
    main()
