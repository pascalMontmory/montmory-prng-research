#!/usr/bin/env python3
"""Sobol32 correctness and same-device benchmark. No silent backend fallback."""
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
PINNED = {"jax": "0.10.2", "jaxlib": "0.10.2", "numpy": "2.4.6", "scipy": "1.17.1"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def archive(out):
    target = out.with_name(out.name + ".tar.gz")
    with tarfile.open(target, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    print("Archive:", target, flush=True)


def campaign(out, backend, quick):
    import jax
    import numpy as np
    import kernels as k
    import reference as ref

    names = list(k.VARIANTS if backend == "tpu" else k.VARIANTS[:4])
    sizes = [512, 8192] if quick else [512, 8192, 131072]
    search_seeds = [11] if quick else [11, 29, 47]
    confirm_seeds = [101, 211] if quick else [101, 211, 307, 401, 503]
    repetitions = {"search": 3, "confirmation": 3 if quick else 7}
    report = {"status": "running", "backend": backend, "quick": quick,
              "scope": "native TPU throughput" if backend == "tpu" else "CPU functional campaign only",
              "dimensions": ref.DIMENSIONS, "bits": 32,
              "timing": "call to ready; allocation and output writes included; host reads excluded",
              "excluded": ["SciPy parameter initialization", "local table preparation", "device uploads",
                           "compilation", "warm-up", "host oracle and timed-output checks"],
              "parameter_reuse": "one parameter set reused for all repeats; cold setup is not a throughput claim",
              "comparison": "selected Pallas versus separately selected best JAX (Gray or block)",
              "samples": 0, "timed_words_compared": 0, "selection": {}, "confirmation": []}
    write_json(out / "protocol.json", dict(versions=PINNED, variants=names, points=sizes,
               search_seeds=search_seeds, confirmation_seeds=confirm_seeds, repetitions=repetitions,
               start_rule="(2**31) + (case_index+1)*512", parameters="SciPy LMS+shift, exported and hashed",
               selection="lowest median per-seed median time, separately within JAX and Pallas",
               success_rule="all timed outputs match exactly; performance reported without a universal-win claim"))

    parameters = {}
    preparation = []
    for phase, seeds in (("search", search_seeds), ("confirmation", confirm_seeds)):
        for seed in seeds:
            begin = time.perf_counter_ns()
            p = ref.parameters(seed)
            param_ns = time.perf_counter_ns() - begin
            begin = time.perf_counter_ns()
            tables = {t: ref.local_table(p, t) for t in (128, 256, 512)}
            table_ns = time.perf_counter_ns() - begin
            p.save(out / f"parameters-{seed}.npz")
            begin = time.perf_counter_ns()
            resident = jax.device_put((p.directions, p.shift, tables))
            jax.block_until_ready(resident)
            upload_ns = time.perf_counter_ns() - begin
            parameters[phase, seed] = (p, resident)
            preparation.append(dict(phase=phase, seed=seed, parameter_sha256=p.digest(),
                                    parameter_ns=param_ns, local_tables_ns=table_ns, upload_ns=upload_ns,
                                    local_table_bytes={t: a.nbytes for t, a in tables.items()}))
    write_json(out / "preparation.json", preparation)
    compiled = {}
    compilation = []
    _, (v0, s0, tables0) = parameters["search", search_seeds[0]]
    start0 = jax.device_put(np.array([1 << 31], dtype=np.uint32))
    for n in sizes:
        for name in names:
            print(f"Compilation: {n} points x 128, {name}", flush=True)
            args = (v0, s0, tables0[k.tile_for(name)], start0)
            begin = time.perf_counter_ns()
            executable = k.make_kernel(name, n, aligned=True).lower(*args).compile()
            compile_ns = time.perf_counter_ns() - begin
            for _ in range(2):
                executable(*args).block_until_ready()
            compiled[n, name] = executable
            compilation.append(dict(points=n, variant=name, compile_ns=compile_ns))
            write_json(out / "compilation.json", compilation)
    write_json(out / "compilation.json", compilation)
    order_rng = np.random.default_rng(20261021)
    observations = {}
    fields = ["phase", "points", "dimensions", "seed", "start", "repeat", "order", "variant",
              "elapsed_ns", "stored_GB_per_second", "output_sha256", "verified"]
    with (out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for phase, seeds in (("search", search_seeds), ("confirmation", confirm_seeds)):
            for n in sizes:
                selected = report["selection"].get(str(n))
                active = names if phase == "search" else list(dict.fromkeys(
                    ["jax_gray", selected["jax"]] + ([selected["pallas"]] if "pallas" in selected else [])))
                for case, seed in enumerate(seeds):
                    p, (v, shift, tables) = parameters[phase, seed]
                    start = (1 << 31) + (case + 1) * 512
                    ref.check_range(start, n)
                    expected = ref.words(p, start, n)
                    digest = hashlib.sha256(expected.astype("<u4").tobytes()).hexdigest()
                    index = jax.device_put(np.array([start], dtype=np.uint32))
                    for name in active:
                        assert "block" not in name or start % k.tile_for(name) == 0
                        args = (v, shift, tables[k.tile_for(name)], index)
                        warm = compiled[n, name](*args)
                        warm.block_until_ready()
                        np.testing.assert_array_equal(np.asarray(warm), expected)
                        del warm
                    for repeat in range(repetitions[phase]):
                        for position, name in enumerate(order_rng.permutation(active)):
                            args = (v, shift, tables[k.tile_for(name)], index)
                            begin = time.perf_counter_ns()
                            result = compiled[n, name](*args)
                            result.block_until_ready()
                            elapsed = time.perf_counter_ns() - begin
                            actual = np.asarray(result)
                            np.testing.assert_array_equal(actual, expected, err_msg=f"timed {name} seed={seed}")
                            writer.writerow(dict(phase=phase, points=n, dimensions=128, seed=seed,
                                start=start, repeat=repeat, order=position, variant=name, elapsed_ns=elapsed,
                                stored_GB_per_second=(n * 128 * 4 / elapsed), output_sha256=digest, verified=True))
                            stream.flush()
                            observations.setdefault((phase, n, str(name), seed), []).append(elapsed)
                            report["samples"] += 1
                            report["timed_words_compared"] += actual.size
                            del result, actual
                    print(f"{phase}: {n} points, seed={seed}, all timed outputs PASS", flush=True)
                if phase == "search":
                    median = lambda name: statistics.median(statistics.median(observations[phase, n, name, seed]) for seed in seeds)
                    choice = {"jax": min((name for name in names if name.startswith("jax")), key=median)}
                    if backend == "tpu":
                        choice["pallas"] = min((name for name in names if name.startswith("pallas")), key=median)
                    report["selection"][str(n)] = choice
                    write_json(out / "selection.json", report["selection"])
                else:
                    medians = {name: [statistics.median(observations[phase, n, name, seed]) for seed in seeds]
                               for name in active}
                    baseline = selected["jax"]
                    for name in active:
                        ratios = [a / b for a, b in zip(medians[baseline], medians[name], strict=True)]
                        row = dict(points=n, variant=name, selected_jax_baseline=baseline,
                                   stored_GB_per_second=n * 128 * 4 / statistics.median(medians[name]),
                                   paired_ratio_vs_selected_jax=statistics.median(ratios),
                                   faster_seeds=sum(r > 1 for r in ratios), seeds=len(seeds),
                                   per_seed_ratio=ratios)
                        report["confirmation"].append(row)
                        print(f"  {name:23} {row['stored_GB_per_second']:.3f} GB/s; "
                              f"paired ratio vs {baseline}={row['paired_ratio_vs_selected_jax']:.3f}", flush=True)
                write_json(out / "campaign.json", report)
    report["status"] = "pass"
    write_json(out / "campaign.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="smaller pilot, not full confirmation")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "backend_requested": args.backend,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": sys.version, "host_platform": platform.platform(),
              "statistical_tests_run": False, "native_tpu_tested": False,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(HERE.glob("*.py"))}}
    try:
        os.environ["JAX_PLATFORMS"] = args.backend
        versions = {name: importlib.metadata.version(name) for name in PINNED}
        report["versions"] = versions
        if versions != PINNED:
            raise RuntimeError(f"version mismatch: required={PINNED}, installed={versions}")
        try:
            report["libtpu"] = importlib.metadata.version("libtpu")
        except importlib.metadata.PackageNotFoundError:
            report["libtpu"] = None
        import jax
        if jax.default_backend() != args.backend:
            raise RuntimeError("requested backend unavailable; fallback forbidden")
        report.update(devices=[str(d) for d in jax.devices()], device_kind=jax.devices()[0].device_kind,
                      jax_enable_x64=bool(jax.config.jax_enable_x64))
        print(f"Backend: {args.backend}, {report['device_kind']}", flush=True)
        import verify
        verification = verify.verify(args.backend)
        write_json(out / "verification.json", verification)
        report["verification"] = {k: v for k, v in verification.items() if k != "cases"}
        report["native_tpu_tested"] = args.backend == "tpu"
        if not args.verify_only:
            report["campaign"] = campaign(out, args.backend, args.quick)
        report["status"] = "pass"
        print("PASS: exact checked outputs. No application-speed or QMC-quality claim.", flush=True)
    except BaseException as error:
        report.update(status="fail", error=repr(error))
        (out / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(out / "report.json", report)
        archive(out)


if __name__ == "__main__":
    main()
