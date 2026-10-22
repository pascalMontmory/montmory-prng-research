"""Shared experiment reporting. Developed for ENINCA CONSULTING; see LICENSE."""
from contextlib import contextmanager
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

ROOT = Path(__file__).resolve().parents[2]
PINS = {"jax": "0.10.2", "jaxlib": "0.10.2", "numpy": "2.4.6", "scipy": "1.17.1"}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def csv_save(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def hashes():
    paths = []
    for name in ("stochastic", "diffusion", "asian_option", "sobol"):
        paths += list((ROOT / "applications" / name).glob("*.py"))
    kit = ROOT / "studies/tpu-opt-2026-10-21/montmory-tpu-opt"
    paths += [kit / n for n in ("kernels.py", "montmory_jax.py", "support.py")]
    paths += list((kit / "reference").glob("*.c")) + list((kit / "reference").glob("*.h"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


@contextmanager
def experiment(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_PLATFORMS"] = args.backend
    started = time.perf_counter()
    report = dict(status="running", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  protocol={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  native_tpu_measurements=False, statistical_batteries_run=False,
                  floating_cross_architecture_bit_identity=False, source_sha256=hashes())
    try:
        versions = {name: importlib.metadata.version(name) for name in PINS}
        if versions != PINS:
            raise RuntimeError(f"Version-pinned protocol: expected {PINS}; found {versions}")
        for name in ("libtpu", "scikit-learn", "matplotlib"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        import jax
        jax.config.update("jax_enable_x64", False)
        jax.config.update("jax_threefry_partitionable", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        if jax.default_backend() != args.backend:
            raise RuntimeError("Requested backend unavailable; CPU fallback forbidden")
        report.update(versions=versions, python=platform.python_version(), platform=platform.platform(), device=str(jax.devices()[0]),
                      device_kind=jax.devices()[0].device_kind, single_device=True,
                      jax_configuration=dict(enable_x64=False, threefry_partitionable=True,
                                             default_matmul_precision="highest"),
                      scope="CPU functional pilot" if args.backend == "cpu" else "native TPU prototype")
        for key, command in (("git_commit", ["git", "rev-parse", "HEAD"]),
                             ("git_status", ["git", "status", "--porcelain"])):
            r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            report[key] = r.stdout.strip() if r.returncode == 0 else None
        print(f"Backend: {args.backend}; {report['device_kind']}", flush=True)
        with jax.default_device(jax.devices()[0]):
            yield report, out
        report["status"] = "pass"
        report["native_tpu_measurements"] = args.backend == "tpu" and not args.verify_only
    except BaseException as exc:
        report.update(status="fail", error=repr(exc))
        (out / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save(out / "report.json", report)
        with tarfile.open(out.with_suffix(".tar.gz"), "w:gz") as tar:
            tar.add(out, arcname=out.name,
                    filter=lambda i: None if i.name.endswith((".so", ".dylib")) else i)
        print(f"Status: {report['status']}; archive: {out.with_suffix('.tar.gz')}", flush=True)
