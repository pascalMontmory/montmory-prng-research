#!/usr/bin/env python3
"""Application v1: shared uniforms/Box-Muller, then a Gaussian expectation.

Uses frozen engines without editing them. CPU mode does not time TPU Pallas.
All inputs are dynamic device arguments. Compilation and checks are not timed.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "studies/tpu-opt-2026-10-21/montmory-tpu-opt"
SEEDS = (101, 1009, 65537, 0xFEDCBA9876543210, 0x800000000000002D,
         0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFF1)
SELECTIONS = {
    65536: ("pallas_original_32", "threefry_pallas_16"),
    1048576: ("pallas_original_32", "threefry_pallas_32"),
    8388608: ("pallas_carry_128", "threefry_pallas_128"),
    16777216: ("pallas_carry_64", "threefry_pallas_128"),
}
WORKLOADS = ("uniform", "normal", "gaussian_expectation")
BASE = (1 << 32) - 4097
ATOL = RTOL = 3e-5


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def uniforms(words):
    # Exact midpoint grid: 23 useful bits, values strictly between zero and one.
    return jnp.stack(tuple(((w >> jnp.uint32(9)).astype(jnp.float32)
                            + jnp.float32(0.5)) * jnp.float32(2**-23)
                           for w in words))


def normals(words):
    u = uniforms(words)
    radius = jnp.sqrt(-jnp.float32(2) * jnp.log(u[0]))
    angle = jnp.float32(2 * np.pi) * u[1]
    return jnp.stack((radius * jnp.cos(angle), radius * jnp.sin(angle)))


def application(spec, count, workload, *, native_normal=False):
    raw = k.function(spec, count)

    def run(parameters):
        if native_normal:
            z = jax.random.normal(parameters, (2, count), dtype=jnp.float32)
        elif workload == "uniform":
            return uniforms(raw(parameters))
        else:
            z = normals(raw(parameters))
        if workload == "normal":
            return z
        # A complete estimator, not just stored RNG output. Ideal N(0,1):
        # E[exp(-Z**2/2)] = 1/sqrt(2).
        return jnp.mean(jnp.exp(-jnp.float32(0.5) * z * z))
    return run


def numpy_reference(words, workload):
    u = np.stack(tuple(((w >> np.uint32(9)).astype(np.float64) + 0.5) * 2**-23
                       for w in words))
    if workload == "uniform":
        return u.astype(np.float32)
    radius = np.sqrt(-2 * np.log(u[0]))
    angle = float(np.float32(2 * np.pi)) * u[1]
    z = np.stack((radius * np.cos(angle), radius * np.sin(angle)))
    if workload == "normal":
        return z
    return np.mean(np.exp(-0.5 * z * z), dtype=np.float64)


def check(actual, expected, workload):
    actual = np.asarray(jax.device_get(actual))
    if actual.shape != expected.shape or actual.dtype != np.float32:
        raise AssertionError(f"{workload}: shape or dtype mismatch")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{workload}: non-finite output")
    if workload == "uniform":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    return float(np.max(np.abs(actual.astype(np.float64) - expected)))


def summarize(rows):
    result = []
    groups = sorted({(r["count64"], r["workload"], r["engine"]) for r in rows})
    for count, workload, engine in groups:
        selected = [r for r in rows if (r["count64"], r["workload"], r["engine"]) ==
                    (count, workload, engine)]
        per_seed = {}
        for seed in sorted({r["seed"] for r in selected}):
            per_seed[seed] = statistics.median(r["seconds"] for r in selected if r["seed"] == seed)
        med = statistics.median(per_seed.values())
        result.append(dict(count64=count, workload=workload, engine=engine,
                           median_seed_ms=med * 1e3, per_seed_seconds=per_seed,
                           million_variates_per_second=2 * count / med / 1e6,
                           stored_GB_per_second=(8 * count / med / 1e9
                                                if workload != "gaussian_expectation" else None)))
    # Within each seed, compare medians; never divide two printed rounded rates.
    lookup = {(s["count64"], s["workload"], s["engine"]): s for s in result}
    for item in result:
        comparisons = {}
        for baseline in ("jax_original", "threefry2x32", "threefry_native_normal",
                         SELECTIONS[item["count64"]][1]):
            other = lookup.get((item["count64"], item["workload"], baseline))
            if other is None:
                continue
            ratios = [other["per_seed_seconds"][seed] / duration
                      for seed, duration in item["per_seed_seconds"].items()]
            comparisons[baseline] = dict(median_ratio=statistics.median(ratios),
                                         faster_seeds=sum(r > 1 for r in ratios),
                                         total_seeds=len(ratios))
        item["paired_speed_ratios"] = comparisons
    return result


def run(args, report):
    if jax.__version__ != "0.10.2" or jax.lib.__version__ != "0.10.2":
        raise RuntimeError("Protocol v1 requires JAX/jaxlib 0.10.2; record a new protocol for another version.")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_threefry_partitionable", True)
    device = jax.devices()[0]
    if jax.default_backend() != args.backend:
        raise RuntimeError("Requested backend unavailable; no automatic fallback.")
    report.update(backend=jax.default_backend(), device_kind=device.device_kind,
                  devices=[str(d) for d in jax.devices()], jax=jax.__version__,
                  jaxlib=jax.lib.__version__, numpy=np.__version__,
                  single_device=True, smoke=args.smoke,
                  scope="CPU functional smoke only" if args.smoke else "native application timing")
    sizes = (65536,) if args.smoke else tuple(SELECTIONS)
    seeds = SEEDS[:2] if args.smoke else SEEDS
    blocks, repetitions = (1, 2) if args.smoke else (3, 3)
    report["protocol"] = dict(version=1, sizes=list(sizes), seeds=[str(s) for s in seeds],
                              blocks=blocks, repetitions_per_block=repetitions,
                              warmups=2, base=str(BASE), T=2048, D=4096,
                              dtype="float32", normal_atol=ATOL, normal_rtol=RTOL,
                              uniform_grid_bits=23, selections=SELECTIONS,
                              integer_equivalence=True,
                              floating_cross_architecture_bit_identity=False,
                              timing="host call through device completion; allocation included; "
                                     "compile, key preparation, transfer to host, checks excluded")
    if args.smoke and args.backend != "cpu":
        raise ValueError("--smoke is explicitly CPU-only; use the full protocol on TPU.")
    oracle = Oracle(args.out / "oracle")
    rows = []
    fields = ("count64", "variates", "workload", "engine", "seed", "block", "repeat",
              "order", "seconds", "max_absolute_error", "estimate")
    with (args.out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for count in sizes:
            names = ["jax_original", "threefry2x32", "philox4x32"]
            if args.backend == "tpu":
                names += list(SELECTIONS[count])
            jobs = [(name, workload, k.candidates()[name], False)
                    for name in names for workload in WORKLOADS]
            jobs += [("threefry_native_normal", w, k.candidates()["threefry2x32"], True)
                     for w in ("normal", "gaussian_expectation")]
            compiled = {}
            raw_compiled = {}
            for name in names:
                spec = k.candidates()[name]
                p = k.arguments(spec, seeds[0], BASE)
                raw_compiled[name] = jax.jit(k.function(spec, count)).lower(p).compile()
            for name, workload, spec, native in jobs:
                p = k.arguments(spec, seeds[0], BASE)
                fn = jax.jit(application(spec, count, workload, native_normal=native))
                compiled[name, workload] = fn.lower(p).compile()
            print(f"{count} words: {len(jobs)} application pipelines compiled.", flush=True)
            for seed in seeds:
                # One common reference per family; all whole integer arrays checked.
                expected_raw = {}
                parameters = {}
                references = {}
                for name in names:
                    spec = k.candidates()[name]
                    parameters[name] = k.arguments(spec, seed, BASE)
                    family = spec["family"]
                    if family not in expected_raw:
                        if family == "montmory":
                            raw64 = oracle.range(seed, BASE, count)
                            expected_raw[family] = (raw64.astype(np.uint32),
                                                  (raw64 >> np.uint64(32)).astype(np.uint32))
                            del raw64
                        else:
                            vendor_spec = k.candidates()[family]
                            p = k.arguments(vendor_spec, seed, BASE)
                            expected_raw[family] = tuple(np.asarray(x) for x in
                                jax.device_get(raw_compiled[family](p)))
                    actual_raw = jax.device_get(raw_compiled[name](parameters[name]))
                    for actual, expected in zip(actual_raw, expected_raw[family]):
                        np.testing.assert_array_equal(actual, expected)
                    report["integer_words_checked"] += count
                native_key = parameters["threefry2x32"]
                native_z = np.asarray(jax.device_get(jax.random.normal(
                    native_key, (2, count), dtype=jnp.float32)))
                for name, workload, spec, native in jobs:
                    if native:
                        parameters[name] = native_key
                        expected = native_z if workload == "normal" else np.mean(
                            np.exp(-0.5 * native_z.astype(np.float64)**2), dtype=np.float64)
                        references[name, workload] = expected
                    else:
                        key = (spec["family"], workload)
                        if key not in references:
                            references[key] = numpy_reference(expected_raw[spec["family"]], workload)
                        references[name, workload] = references[key]
                    fn = compiled[name, workload]
                    for _ in range(2):
                        jax.block_until_ready(fn(parameters[name]))
                for block in range(blocks):
                    for repetition in range(repetitions):
                        order = list(jobs)
                        random.Random(seed ^ count ^ (block << 20) ^ repetition).shuffle(order)
                        for position, (name, workload, spec, native) in enumerate(order):
                            fn, p = compiled[name, workload], parameters[name]
                            started = time.perf_counter_ns()
                            output = fn(p)
                            jax.block_until_ready(output)
                            seconds = (time.perf_counter_ns() - started) / 1e9
                            error = check(output, references[name, workload], workload)
                            row = dict(count64=count, variates=2*count, workload=workload,
                                       engine=name, seed=str(seed), block=block+1,
                                       repeat=repetition+1, order=position, seconds=seconds,
                                       max_absolute_error=error,
                                       estimate=float(output) if workload == "gaussian_expectation" else "")
                            writer.writerow(row)
                            stream.flush()
                            rows.append(row)
                            report["timed_outputs_checked"] += 1
                            del output
                    print(f"{count} words, seed {seed}, block {block+1}/{blocks}: outputs PASS.", flush=True)
                del references, expected_raw, native_z
            # Release compiled executables for this shape before the next volume.
            del compiled, raw_compiled
    report["summary"] = summarize(rows)
    report["status"] = "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", requested_backend=args.backend,
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(),
                  statistical_battery_run=False, integer_words_checked=0,
                  timed_outputs_checked=0, summary=[])
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        sources = [Path(__file__), KIT/"kernels.py", KIT/"montmory_jax.py",
                   KIT/"support.py", KIT/"reference/oracle.c",
                   KIT/"reference/mmc_core.c", KIT/"reference/mmc_core.h", KIT/"requirements.txt",
                   ROOT/"docs/benchmark-applicatif.md"]
        report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources}
        os.environ["JAX_PLATFORMS"] = args.backend
        sys.path.insert(0, str(KIT))
        global jax, jnp, np, k, Oracle
        import jax
        import jax.numpy as jnp
        import numpy as np
        import kernels as k
        from support import Oracle
        with jax.default_device(jax.devices()[0]):
            run(args, report)
    except BaseException as exc:
        report.update(status="fail", error=repr(exc))
        raise
    finally:
        write_json(args.out / "report.json", report)
        print(f"Report: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
