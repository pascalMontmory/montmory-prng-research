"""Run actual-backend conformance; CPU mode interprets the Pallas kernels."""
import argparse
import functools
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cpu", "tpu"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
os.environ["JAX_PLATFORMS"] = args.backend

import jax
import jax.numpy as jnp
import numpy as np

import montmory_jax as m
from support import ROOT, Oracle, assert_equal, unpack, write_json

jax.config.update("jax_enable_x64", False)


def pair(x):
    return jnp.asarray(x.astype(np.uint32)), jnp.asarray((x >> np.uint64(32)).astype(np.uint32))


def main():
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "requested_backend": args.backend, "jax": jax.__version__,
              "word_comparisons": 0, "cases": [], "statistical_tests_run": False}
    try:
        report.update(backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                      native_pallas_tpu=args.backend == "tpu", interpreted_pallas=args.backend == "cpu")
        if jax.default_backend() != args.backend:
            raise RuntimeError("Requested backend unavailable")
        oracle = Oracle(args.out)
        golden = np.array([[int(x, 16) for x in line.split()]
                           for line in (ROOT / "reference/golden-vectors.txt").read_text().splitlines()], dtype=np.uint64)
        seeds, lanes, draws, raw, mant = [np.ascontiguousarray(golden[:, i]) for i in range(5)]
        for method in ("portable", "mulhi"):
            words = jax.jit(m.lane_words, static_argnames=("method",))(pair(seeds), pair(draws), pair(lanes), method=method)
            assert_equal(unpack(words), raw, "historical raw64")
            assert_equal(unpack(jax.jit(m.mantissa_words)(words)), mant, "historical mantissa")
            report["word_comparisons"] += 2*len(raw)
        print(f"{args.backend}: historical vectors PASS.", flush=True)
        functions = [("jax_portable", functools.partial(m.prepared_range, count=8193, method="portable")),
                     ("jax_mulhi", functools.partial(m.prepared_range, count=8193, method="mulhi"))]
        for rows in (8, 16, 32):
            functions.append((f"pallas_{rows}", functools.partial(m._pallas_range, count=8193,
                              block_rows=rows, method="portable", interpret=args.backend == "cpu")))
        p = m.pack_parameters(m.constant(42), m.constant(0))
        compiled = [(name, jax.jit(fn).lower(p).compile()) for name, fn in functions]
        convert = jax.jit(m.mantissa_words)
        for seed in (0, 42, m.MAX64, (1 << 63)+1):
            for base in (0, 4095, 8388607, (1 << 32)-7, (1 << 63)-7, m.MAX64-8192):
                expected = oracle.range(seed, base, 8193)
                p = m.pack_parameters(m.constant(seed), m.constant(base))
                for name, fn in compiled:
                    words = fn(p)
                    assert_equal(unpack(words), expected, name)
                    assert_equal(unpack(convert(words)), expected >> np.uint64(11), name + " mantissa")
                    report["word_comparisons"] += 2*8193
                    report["cases"].append({"seed": seed, "base": base, "engine": name, "words": 8193})
            print(f"{args.backend}: seed {seed}, boundary checks PASS.", flush=True)
        report["status"] = "pass"
    except BaseException as error:
        report.update(status="fail", error=repr(error))
        raise
    finally:
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
