"""Finite cross-implementation conformance checks; no TPU speed claims."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)
import montmory_jax as m

ROOT = Path(__file__).resolve().parent
MAX64 = (1 << 64)-1


def pair(values):
    a = np.asarray(values, dtype=np.uint64)
    return jnp.asarray(a.astype(np.uint32)), jnp.asarray((a >> np.uint64(32)).astype(np.uint32))


def unpack(words):
    lo, hi = (np.asarray(jax.device_get(a), dtype=np.uint64) for a in words)
    return lo | (hi << np.uint64(32))


def indices(base, count):
    return np.arange(count, dtype=np.uint64) + np.uint64(base)


def load_oracle(outdir):
    compiler = shlex.split(os.environ.get("CC", "cc"))
    env = dict(os.environ)
    if sys.platform == "darwin" and "CC" not in os.environ:
        compiler = [subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()]
        env["SDKROOT"] = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    target = outdir / ("oracle.dylib" if sys.platform == "darwin" else "oracle.so")
    command = compiler + ["-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC",
                          "-dynamiclib" if sys.platform == "darwin" else "-shared",
                          str(ROOT / "reference/oracle.c"), "-o", str(target)]
    subprocess.run(command, env=env, check=True)
    lib = ctypes.CDLL(str(target))
    arr = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
    lib.reference_words.argtypes = [ctypes.c_uint64, arr, arr, ctypes.c_size_t]
    lib.reference_words.restype = None
    lib.reference_lanes.argtypes = [arr, arr, arr, arr, ctypes.c_size_t]
    lib.reference_lanes.restype = None
    return lib, command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Compare deterministic output fingerprint to a prior report")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "scope": "JAX CPU and Pallas CPU interpretation only",
              "native_tpu_tested": False, "statistical_batteries_run": False,
              "logical_lanes": m.LANES, "draws_per_lane": m.DRAWS,
              "jax": jax.__version__, "jaxlib": jax.lib.__version__, "numpy": np.__version__,
              "python": sys.version, "platform": platform.platform(),
              "machine": platform.machine(), "devices": [str(d) for d in jax.devices()],
              "jax_enable_x64": bool(jax.config.jax_enable_x64), "counts": {}, "checks": 0}
    fingerprint = hashlib.sha256()
    checks = []

    def check(family, label, actual, expected):
        actual = np.asarray(actual, dtype=np.uint64)
        expected = np.asarray(expected, dtype=np.uint64)
        if actual.shape != expected.shape:
            raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
        mismatch = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
        if mismatch.size:
            i = int(mismatch[0])
            raise AssertionError(f"{label}: word {i}, actual={int(actual.reshape(-1)[i]):016x}, expected={int(expected.reshape(-1)[i]):016x}")
        fingerprint.update(label.encode("utf-8") + b"\0")
        fingerprint.update(actual.astype("<u8").tobytes())
        report["counts"][family] = report["counts"].get(family, 0) + int(actual.size)
        report["checks"] += 1
        checks.append({"family": family, "label": label, "words": int(actual.size)})

    try:
        if any(d.platform != "cpu" for d in jax.devices()):
            raise RuntimeError("This validation protocol explicitly requires the CPU backend")
        oracle, command = load_oracle(args.out)
        report["oracle_compile_command"] = command
        report["reference_sha256"] = {name: hashlib.sha256((ROOT / "reference" / name).read_bytes()).hexdigest()
                                      for name in ("mmc_core.c", "mmc_core.h", "golden-vectors.txt")}
        print(f"Backend CPU: {report['devices']}; JAX {jax.__version__}; calcul uint32, x64 desactive.", flush=True)

        def reference(seed, ix):
            ix = np.ascontiguousarray(ix, dtype=np.uint64)
            out = np.empty(ix.size, dtype=np.uint64)
            oracle.reference_words(seed, ix, out, ix.size)
            return out

        golden = np.array([[int(x, 16) for x in line.split()]
                           for line in (ROOT / "reference/golden-vectors.txt").read_text().splitlines()], dtype=np.uint64)
        seeds, lanes, draws, raw, mantissa = [np.ascontiguousarray(golden[:, i]) for i in range(5)]
        native = np.empty(raw.size, dtype=np.uint64)
        oracle.reference_lanes(seeds, draws, lanes, native, raw.size)
        check("golden_native", "C vs frozen x86 vectors", native, raw)
        methods = ("portable", "mulhi")
        jitted_lanes = jax.jit(m.lane_words, static_argnames=("method",))
        jitted_stream = jax.jit(m.stream_words, static_argnames=("method",))
        jitted_mantissa = jax.jit(m.mantissa_words)
        for method in methods:
            for execution, fn in (("eager", m.lane_words), ("jit", jitted_lanes)):
                words = fn(pair(seeds), pair(draws), pair(lanes), method=method)
                check("golden_jax", f"golden {method} {execution} raw", unpack(words), raw)
                check("golden_mantissa", f"golden {method} {execution} mantissa53", unpack(jitted_mantissa(words)), mantissa)
        print("1600 vecteurs historiques: C, JAX eager/JIT, deux multiplications, raw64 et mantissa53 PASS.", flush=True)

        rng = np.random.Generator(np.random.PCG64(0x4D4F4E54))
        a = rng.integers(0, 1 << 32, 100_000, dtype=np.uint32)
        b = rng.integers(0, 1 << 32, 100_000, dtype=np.uint32)
        edge = np.array([0, 1, 65535, 65536, (1 << 31)-1, 1 << 31, (1 << 32)-1], dtype=np.uint32)
        aa, bb = np.meshgrid(edge, edge)
        a = np.concatenate((a, aa.ravel()))
        b = np.concatenate((b, bb.ravel()))
        expected_high = (a.astype(np.uint64) * b.astype(np.uint64)) >> np.uint64(32)
        for method in methods:
            actual = jax.jit(m.high32, static_argnames=("method",))(jnp.asarray(a), jnp.asarray(b), method=method)
            check("mulhi_components", f"high32 {method}", actual, expected_high)

        # Random seeds and full-width lane IDs exercise initialization and rotations.
        seeds_random = rng.bit_generator.random_raw(32768)
        draws_random = rng.bit_generator.random_raw(32768)
        lanes_random = rng.bit_generator.random_raw(32768)
        expected = np.empty(seeds_random.size, dtype=np.uint64)
        oracle.reference_lanes(seeds_random, draws_random, lanes_random, expected, expected.size)
        for method in methods:
            words = jitted_lanes(pair(seeds_random), pair(draws_random), pair(lanes_random), method=method)
            check("random_lane_words", f"random full-width lanes {method}", unpack(words), expected)

        ix = rng.bit_generator.random_raw(16384)
        boundary = [0, 4096, 8388608, 1 << 32, 1 << 63, MAX64]
        edge_indices = [(x + d) & MAX64 for x in boundary for d in (-2, -1, 0, 1, 2)]
        ix[:len(edge_indices)] = np.asarray(edge_indices, dtype=np.uint64)
        test_seeds = [0, 1, 42, MAX64, 1 << 63] + [int(x) for x in rng.bit_generator.random_raw(7)]
        for seed in test_seeds:
            expected = reference(seed, ix)
            for method in methods:
                actual = jitted_stream(m.constant(seed), pair(ix), method=method)
                check("indexed_words", f"indexed {seed} {method}", unpack(actual), expected)
        print("Produits hauts, seeds et indices 64 bits, lanes et limites de compteurs: PASS.", flush=True)

        bases = [0, 4095, 8388605, (1 << 32)-3, (1 << 63)-3, MAX64-65535]
        for seed in (0, 42, MAX64):
            for base in bases:
                expected = reference(seed, indices(base, 65536))
                for method in methods:
                    actual = m.generate(seed, base, 65536, method=method)
                    check("range_words", f"range {seed} {base} {method}", unpack(actual), expected)
                    check("range_mantissa", f"mantissa {seed} {base} {method}", unpack(jitted_mantissa(actual)), expected >> np.uint64(11))
        print("Blocs de 65536 mots, debordements internes et mantisses exactes: PASS.", flush=True)

        base, count, seed = (1 << 32)-4100, 16385, MAX64
        lengths = [0, 1, 127, 3968, 4096, 17, 8176]
        starts = np.cumsum([0] + lengths[:-1]).tolist()
        expected = reference(seed, indices(base, count))
        for method in methods:
            assembled = np.empty(count, dtype=np.uint64)
            for start, n in reversed(list(zip(starts, lengths))):
                assembled[start:start+n] = unpack(m.generate(seed, base+start, n, method=method))
            check("partition_words", f"reversed partitions {method}", assembled, expected)

        invalid = [(-1, 0, 1), (1 << 64, 0, 1), (0, -1, 1), (0, 0, -1), (0, MAX64, 2), (0, 0, 1 << 32)]
        rejected = 0
        for parameters in invalid:
            for fn in (m.generate, m.generate_pallas):
                try:
                    fn(*parameters)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError(f"Invalid range accepted: {parameters}")
        report["invalid_ranges_rejected"] = rejected
        print("Decoupages executes dans l'ordre inverse et rejets des indices invalides: PASS.", flush=True)

        for block_rows in (8, 16, 32):
            for seed, base in ((42, 8388608-4097), (0, (1 << 32)-4097), (MAX64, MAX64-8192)):
                expected = reference(seed, indices(base, 8193))
                for method in methods:
                    actual = m.generate_pallas(seed, base, 8193, block_rows=block_rows, method=method, interpret=True)
                    check("pallas_interpreted_words", f"pallas {block_rows} {seed} {base} {method}", unpack(actual), expected)
            print(f"Pallas interprete sur CPU, blocs {block_rows} x 128: PASS (aucune mesure TPU).", flush=True)

        report["output_fingerprint_sha256"] = fingerprint.hexdigest()
        if args.compare:
            previous = json.loads(args.compare.read_text())
            if previous.get("status") != "pass" or previous.get("output_fingerprint_sha256") != report["output_fingerprint_sha256"]:
                raise AssertionError("Cross-machine report fingerprint mismatch")
            report["cross_machine_comparison"] = str(args.compare)
        report["status"] = "pass"
        print("PASS: conformite locale. TPU natif, performances TPU et split/fold_in restent a valider.", flush=True)
    except BaseException as error:
        report["status"] = "fail"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_validation_seconds_not_benchmark"] = time.monotonic()-started
        report["checked_cases"] = checks
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
