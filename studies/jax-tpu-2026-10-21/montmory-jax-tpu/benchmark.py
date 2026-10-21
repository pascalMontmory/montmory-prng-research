"""Fixed-protocol JAX/TPU benchmark with host synchronization and output checks."""
import argparse
import csv
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

# Backend must be selected before importing JAX. Never silently fall back from TPU.
parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--prepare-references", action="store_true")
parser.add_argument("--quick", action="store_true", help="3 seeds, 3 repetitions, smallest size")
parser.add_argument("--include-rbg", action="store_true", help="Add XLA RBG, with same-device checks only")
args = parser.parse_args()
os.environ["JAX_PLATFORMS"] = args.backend
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

import montmory_jax as m
from support import ROOT, Oracle, unpack, write_json

jax.config.update("jax_enable_x64", False)
SEEDS = (0, 1, 42, (1 << 64)-1, 0x8000000000000001)
SIZES = (65536, 1048576)
BASE = (1 << 32)-4097
REFERENCE_PATH = ROOT / "reference/expected-output-hashes.json"
VENDORS = ("threefry2x32", "threefry4x32", "philox4x32", "philox2x32")


def digest(words):
    return hashlib.sha256(np.asarray(words, dtype="<u8").tobytes()).hexdigest()


def identifier(family, count, seed):
    return f"{family}/{count}/{seed}"


def vendor_bits(key, count):
    bits = jax.random.bits(key, (2, count), dtype=jnp.uint32)
    return bits[0], bits[1]


def baseline(p, count):
    x = jnp.arange(count, dtype=jnp.uint32) + p[0]
    return x ^ p[1], x + p[1]


def inputs(family, seed):
    if family == "montmory":
        out = m.pack_parameters(m.constant(seed), m.constant(BASE))
    elif family == "write_baseline_NOT_PRNG":
        out = jnp.asarray([seed & m.MASK32, seed >> 32], dtype=jnp.uint32)
    else:
        # Defined 64-bit seed adapter even with jax_enable_x64=False.
        key = jax.random.key(np.uint32(seed & m.MASK32), impl=family)
        out = jax.random.fold_in(key, np.uint32(seed >> 32))
    return jax.block_until_ready(out)


def functions(count, backend):
    result = [
        ("montmory_jax_portable", "montmory", functools.partial(m.prepared_range, count=count, method="portable")),
        ("montmory_jax_mulhi", "montmory", functools.partial(m.prepared_range, count=count, method="mulhi")),
    ]
    for name in VENDORS:
        result.append((name, name, functools.partial(vendor_bits, count=count)))
    result.append(("write_baseline_NOT_PRNG", "write_baseline_NOT_PRNG", functools.partial(baseline, count=count)))
    if backend == "tpu":
        for rows in (8, 16, 32):
            result.append((f"montmory_pallas_{rows}x128", "montmory", functools.partial(m._pallas_range,
                           count=count, block_rows=rows, method="portable", interpret=False)))
    if args.include_rbg:
        result.append(("rbg_same_device_only", "rbg", functools.partial(vendor_bits, count=count)))
    return result


def prepare_references(outdir):
    if args.backend != "cpu":
        raise ValueError("Reference preparation must run on CPU")
    oracle = Oracle(outdir)
    records = {}
    for count in SIZES:
        for seed in SEEDS:
            records[identifier("montmory", count, seed)] = digest(oracle.range(seed, BASE, count))
            a, b = np.uint32(seed & m.MASK32), np.uint32(seed >> 32)
            x = np.arange(count, dtype=np.uint32) + a
            raw = (x ^ b).astype(np.uint64) | ((x + b).astype(np.uint64) << np.uint64(32))
            records[identifier("write_baseline_NOT_PRNG", count, seed)] = digest(raw)
        fn = jax.jit(functools.partial(vendor_bits, count=count))
        for family in VENDORS:
            for seed in SEEDS:
                records[identifier(family, count, seed)] = digest(unpack(fn(inputs(family, seed))))
    write_json(REFERENCE_PATH, {"jax": jax.__version__, "jaxlib": jax.lib.__version__, "source_backend": "cpu",
                               "base": BASE, "sizes": SIZES, "seeds": SEEDS, "sha256": records,
                               "vendor_scope": "JAX 0.10.2 API outputs, not independent Random123 KATs"})
    print(f"Prepared {len(records)} fixed CPU output fingerprints.", flush=True)


def main():
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "requested_backend": args.backend, "jax": jax.__version__,
              "jaxlib": jax.lib.__version__, "platform": platform.platform(),
              "scope": "single-device generation of two uint32 arrays, 8 stored bytes per word",
              "timing": "host wall clock including dispatch, allocation and block_until_ready; excluding setup, compilation, transfers and checking",
              "seed_setup": "all seed/key derivation is outside timed calls, including Montmory splitmix",
              "logical_lanes": 2048, "draws_per_lane": 4096, "base": BASE,
              "quick": args.quick, "samples": 0, "checked_timed_words": 0, "summary": []}
    rows = []
    try:
        report["backend"] = jax.default_backend()
        report["devices"] = [str(d) for d in jax.devices()]
        report["device_kind"] = jax.devices()[0].device_kind
        if jax.default_backend() != args.backend:
            raise RuntimeError("Requested backend is unavailable")
        if args.prepare_references:
            prepare_references(args.out)
            report["status"] = "references_prepared"
            return
        references = json.loads(REFERENCE_PATH.read_text())
        if references["jax"] != jax.__version__ or references["jaxlib"] != jax.lib.__version__:
            raise RuntimeError("JAX versions must match the fixed reference protocol")
        oracle = Oracle(args.out)
        seeds = SEEDS[:3] if args.quick else SEEDS
        sizes = SIZES[:1] if args.quick else SIZES
        repetitions = 3 if args.quick else 7
        report.update(seeds=seeds, sizes=sizes, repetitions=repetitions)
        fields = ("backend", "device_kind", "words", "engine", "seed", "repetition", "nanoseconds", "output_sha256", "check")
        with (args.out / "measurements.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            # Single device only; multi-device correctness has its own protocol.
            with jax.default_device(jax.devices()[0]):
                for count in sizes:
                    candidates = functions(count, args.backend)
                    compiled = {}
                    for name, family, fn in candidates:
                        compiled[name] = jax.jit(fn).lower(inputs(family, seeds[0])).compile()
                    print(f"{args.backend}: {count} words, {len(candidates)} pipelines compiled before timing.", flush=True)
                    for seed_index, seed in enumerate(seeds):
                        # Check the frozen Montmory fingerprint against freshly rebuilt C too.
                        assert digest(oracle.range(seed, BASE, count)) == references["sha256"][identifier("montmory", count, seed)]
                        arguments = {family: inputs(family, seed) for _, family, _ in candidates}
                        expected = {}
                        for name, family, _ in candidates:
                            expected[name] = (references["sha256"][identifier(family, count, seed)] if family != "rbg" else None)
                            for warmup in range(3):
                                actual = compiled[name](arguments[family])
                                jax.block_until_ready(actual)
                                h = digest(unpack(actual))
                                if expected[name] is None:
                                    expected[name] = h
                                if h != expected[name]:
                                    raise AssertionError(f"Warmup output mismatch: {name}, seed={seed}")
                        order_rng = random.Random(20261021 + count + seed_index)
                        for repetition in range(repetitions):
                            order = list(candidates)
                            order_rng.shuffle(order)
                            for name, family, _ in order:
                                start = time.perf_counter_ns()
                                actual = compiled[name](arguments[family])
                                jax.block_until_ready(actual)
                                elapsed = time.perf_counter_ns()-start
                                h = digest(unpack(actual))  # Transfer/check outside timing.
                                if h != expected[name]:
                                    raise AssertionError(f"Timed output mismatch: {name}, seed={seed}")
                                row = dict(backend=args.backend, device_kind=report["device_kind"], words=count,
                                           engine=name, seed=seed, repetition=repetition, nanoseconds=elapsed,
                                           output_sha256=h, check="same_device_repeatability" if family == "rbg" else "fixed_cpu_fingerprint")
                                writer.writerow(row)
                                rows.append(row)
                                report["samples"] += 1
                                report["checked_timed_words"] += count
                            file.flush()
                        print(f"  seed {seed_index+1}/{len(seeds)}: all timed outputs checked.", flush=True)
                    ref = {seed: statistics.median(r["nanoseconds"] for r in rows if r["words"] == count and r["engine"] == "montmory_jax_portable" and r["seed"] == seed) for seed in seeds}
                    for name, _, _ in candidates:
                        times = {seed: statistics.median(r["nanoseconds"] for r in rows if r["words"] == count and r["engine"] == name and r["seed"] == seed) for seed in seeds}
                        ns = statistics.median(times.values())
                        item = {"engine": name, "words": count, "median_ns_per_word": ns/count,
                                "stored_GB_per_second": 8*count/ns,
                                "paired_ratio_vs_montmory_jax_portable": statistics.median(ref[s]/times[s] for s in seeds),
                                "faster_seeds": sum(times[s] < ref[s] for s in seeds), "seed_count": len(seeds)}
                        report["summary"].append(item)
                        print(f"{name:30s} {item['stored_GB_per_second']:8.3f} GB/s  ratio={item['paired_ratio_vs_montmory_jax_portable']:.3f}", flush=True)
        report["status"] = "pass"
    except BaseException as error:
        report.update(status="fail", error=repr(error))
        raise
    finally:
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
