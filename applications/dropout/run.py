#!/usr/bin/env python3
"""Validate dropout and train a small MLP. CPU is a functional check only."""
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import time
import traceback


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 42])
    parser.add_argument("--rows", type=int, choices=(8, 16, 32, 64, 128), default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if not 2 <= args.steps < (1 << 32) or not 1 <= args.width <= 4096 or not 1 <= args.batch <= 2048:
        parser.error("require 2 <= steps < 2**32, width in [1,4096], batch in [1,2048]")
    if not args.seeds or any(not 0 <= s < (1 << 64) for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct unsigned 64-bit integers")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_PLATFORMS"] = args.backend
    report = {"status": "running", "requested_backend": args.backend,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "protocol": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "statistical_batteries_run": False, "native_tpu_measurements": False,
              "cross_architecture_training_bitwise_claim": False}
    start = time.perf_counter()
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        if jax.__version__ != "0.10.2":
            raise RuntimeError(f"protocol requires JAX 0.10.2; found {jax.__version__}")
        jax.config.update("jax_enable_x64", False)
        jax.config.update("jax_threefry_partitionable", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        if jax.default_backend() != args.backend:
            raise RuntimeError("requested backend unavailable; no fallback allowed")
        import dropout as d
        import model
        import verify
        root_dir = Path(__file__).resolve().parents[2]
        source_paths = [Path(__file__).parent / name for name in ("dropout.py", "model.py", "run.py", "verify.py")] + [
            d.KIT / name for name in ("kernels.py", "montmory_jax.py", "support.py", "reference/oracle.c")]
        source_paths += list((d.KIT / "reference").glob("*.h"))
        source_paths += list((d.KIT / "reference").glob("*.c"))
        report.update(jax=jax.__version__, jaxlib=jax.lib.__version__, numpy=np.__version__,
                      platform=platform.platform(), devices=[str(v) for v in jax.devices()],
                      device_kind=jax.devices()[0].device_kind,
                      matmul_precision="highest", logical_geometry={"T": 2048, "D": 4096},
                      source_sha256={str(p.relative_to(root_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(set(source_paths))})
        try:
            report["libtpu"] = importlib.metadata.version("libtpu")
        except importlib.metadata.PackageNotFoundError:
            report["libtpu"] = None
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root_dir, capture_output=True, text=True)
        report["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root_dir, capture_output=True, text=True)
        report["git_status"] = status.stdout.splitlines() if status.returncode == 0 else None
        print(f"Backend: {args.backend}; {report['device_kind']}. Checking masks and gradients...", flush=True)
        report["verification"] = verify.run(out, interpret=args.backend == "cpu", selected_rows=args.rows)
        save(out / "report.json", report)

        (host_x, host_y), (host_vx, host_vy) = model.dataset()
        data_x, data_y, val_x, val_y = map(jnp.asarray, (host_x, host_y, host_vx, host_vy))
        jax.block_until_ready((data_x, data_y, val_x, val_y))
        batch_rng = np.random.Generator(np.random.PCG64(123456))
        # Fixed batch order shared by every engine, implementation and seed.
        batch_indices = jnp.asarray(batch_rng.integers(0, len(host_x), (args.steps, args.batch), dtype=np.int32))
        batches_x, batches_y = data_x[batch_indices], data_y[batch_indices]
        jax.block_until_ready((batches_x, batches_y))
        data_digest = hashlib.sha256()
        for array in (host_x, host_y, host_vx, host_vy, np.asarray(batch_indices)):
            data_digest.update(array.tobytes())
        report["dataset_and_batch_order_sha256"] = data_digest.hexdigest()
        report["model"] = {"inputs": model.INPUTS, "hidden": [args.width, args.width],
                           "classes": model.CLASSES, "parameters": sum(x.size for x in jax.tree.leaves(model.initialize(args.width, 0))),
                           "dropout_rate": 0.25, "learning_rate": model.LEARNING_RATE,
                           "training_examples": len(host_x), "validation_examples": len(host_vx),
                           "synthetic_task": "class of maximum of four fixed linear teacher scores"}
        variants = [(e, "jax") for e in ("threefry", "montmory")]
        if args.backend == "tpu":
            variants += [(e, "pallas") for e in ("threefry", "montmory")]
        compiled = {}
        report["compilation_seconds"] = {}
        for engine, implementation in variants:
            name = engine + "_" + implementation
            print(f"Compiling complete training step: {name}", flush=True)
            fn = model.training_step(engine, implementation, args.rows)
            weights = model.initialize(args.width, args.seeds[0])
            params = d.root_parameters(engine, args.seeds[0])
            compilation_start = time.perf_counter()
            compiled[name] = fn.lower(weights, batches_x[0], batches_y[0], params, jnp.uint32(0)).compile()
            report["compilation_seconds"][name] = time.perf_counter() - compilation_start
            for number in range(3):
                jax.block_until_ready(compiled[name](weights, batches_x[0], batches_y[0], params, jnp.uint32(number)))

        raw_rows, curves, summaries = [], [], []
        report["training"] = summaries
        for seed in args.seeds:
            initial = model.initialize(args.width, seed)
            initial_loss, initial_accuracy = map(float, model.evaluate(initial, val_x, val_y))
            states = {e + "_" + i: initial for e, i in variants}
            parameters = {e: d.root_parameters(e, seed) for e in ("threefry", "montmory")}
            jax.block_until_ready((initial, parameters))
            histories = {name: [] for name in states}
            times = {name: [] for name in states}
            order_rng = np.random.Generator(np.random.PCG64(seed))
            for number in range(args.steps):
                x, y = batches_x[number], batches_y[number]
                step_number = jnp.uint32(number)
                jax.block_until_ready((x, y, step_number))
                # Randomized interleaving at every step limits order/drift bias.
                for position, variant_index in enumerate(order_rng.permutation(len(variants))):
                    engine, implementation = variants[int(variant_index)]
                    name = engine + "_" + implementation
                    begun = time.perf_counter_ns()
                    states[name], loss = compiled[name](states[name], x, y, parameters[engine], step_number)
                    jax.block_until_ready((states[name], loss))
                    elapsed = time.perf_counter_ns() - begun
                    loss_value = float(loss)
                    if not np.isfinite(loss_value):
                        raise AssertionError(f"nonfinite loss: {seed=} {name=} {number=}")
                    histories[name].append(loss_value)
                    times[name].append(elapsed / 1e6)
                    raw_rows.append({"seed": seed, "variant": name, "step": number, "order": position,
                                     "elapsed_ns": elapsed, "loss": loss_value,
                                     "timing_scope": "native_training_step" if args.backend == "tpu" else "cpu_functional_run"})
                if (number + 1) % 50 == 0 or number == args.steps - 1:
                    for name in states:
                        val_loss, accuracy = map(float, model.evaluate(states[name], val_x, val_y))
                        curves.append({"seed": seed, "variant": name, "steps": number + 1,
                                       "validation_loss": val_loss, "validation_accuracy": accuracy})
                    print(f"Seed {seed}: {number + 1}/{args.steps} updates complete", flush=True)
            for name in states:
                for leaf in jax.tree.leaves(states[name]):
                    if not np.isfinite(np.asarray(leaf)).all():
                        raise AssertionError(f"nonfinite trained weights: {seed=} {name=}")
                loss, accuracy = map(float, model.evaluate(states[name], val_x, val_y))
                entry = {"seed": seed, "variant": name, "initial_validation_loss": initial_loss,
                         "final_validation_loss": loss, "initial_validation_accuracy": initial_accuracy,
                         "final_validation_accuracy": accuracy}
                if args.backend == "tpu":
                    entry["median_step_ms"] = float(np.median(times[name]))
                summaries.append(entry)
                print(f"  {name}: validation loss {initial_loss:.4f} -> {loss:.4f}; accuracy {initial_accuracy:.1%} -> {accuracy:.1%}", flush=True)
            if args.backend == "tpu":
                report.setdefault("trained_equivalence", [])
                for engine in ("threefry", "montmory"):
                    left, right = engine + "_jax", engine + "_pallas"
                    # Accumulated floating-point differences are checked with a tolerance.
                    np.testing.assert_allclose(histories[left], histories[right], rtol=2e-4, atol=2e-5)
                    worst = 0.0
                    for a, b in zip(jax.tree.leaves(states[left]), jax.tree.leaves(states[right]), strict=True):
                        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5)
                        worst = max(worst, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
                    report["trained_equivalence"].append({"seed": seed, "engine": engine, "max_weight_difference": worst})
            for filename, records in (("steps.csv", raw_rows), ("learning_curves.csv", curves)):
                with (out / filename).open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
            save(out / "report.json", report)

        if args.backend == "tpu":
            report["native_tpu_measurements"] = True
            comparisons = []
            for engine in ("threefry", "montmory"):
                ratios = []
                for seed in args.seeds:
                    entries = {r["variant"]: r for r in summaries if r["seed"] == seed}
                    ratios.append(entries[engine + "_jax"]["median_step_ms"] / entries[engine + "_pallas"]["median_step_ms"])
                comparisons.append({"engine": engine, "reference_step_time_over_pallas": ratios,
                                    "median_ratio": float(np.median(ratios)),
                                    "seeds_pallas_faster": sum(r > 1 for r in ratios)})
            report["performance_comparisons"] = comparisons
        report["status"] = "pass"
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = repr(error)
        (out / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        save(out / "report.json", report)
        archive = out.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out, arcname=out.name, filter=lambda item: None if item.name.endswith((".so", ".dylib")) else item)
        print(f"Status: {report['status']}. Results: {out}. Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
