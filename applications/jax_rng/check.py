#!/usr/bin/env python3
"""Check application boundaries, failure detection and interpreted Pallas wiring.

CPU correctness only. Never use these interpreted timings as TPU measurements.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from pathlib import Path
import json
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import benchmark as b

sys.path.insert(0, str(b.KIT))
import kernels as k
from support import Oracle

b.jax, b.jnp, b.np, b.k = jax, jnp, np, k
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_threefry_partitionable", True)


def main():
    words = (jnp.array([0, 1, 511, 512, 0xFFFFFFFF], dtype=jnp.uint32),
             jnp.array([0xFFFFFFFF, 512, 511, 1, 0], dtype=jnp.uint32))
    u = jax.jit(b.uniforms)(words)
    np.testing.assert_array_equal(u, b.numpy_reference(tuple(np.asarray(w) for w in words), "uniform"))
    if not np.all((np.asarray(u) > 0) & (np.asarray(u) < 1)):
        raise AssertionError("Uniform endpoints are not strictly open")
    z = jax.jit(b.normals)(words)
    b.check(z, b.numpy_reference(tuple(np.asarray(w) for w in words), "normal"), "normal")
    rejected = 0
    for broken in (u.at[0, 0].set(0), u[:, :2], u.at[0, 0].set(jnp.nan)):
        try:
            b.check(broken, np.asarray(u), "uniform")
        except AssertionError:
            rejected += 1
    if rejected != 3:
        raise AssertionError("Output checks accepted corrupted outputs")
    cases = 0
    with tempfile.TemporaryDirectory(prefix="montmory-application-check-") as folder:
        oracle = Oracle(folder)
        for seed in (0, (1 << 64)-1):
            for name in ("pallas_original_32", "pallas_carry_64", "threefry_pallas_128"):
                spec = k.candidates()[name]
                # Non-tile-aligned length exercises the slicing path.
                count = 4097
                p = k.arguments(spec, seed, b.BASE)
                raw = jax.jit(k.function(spec, count, interpret=True))(p)
                if spec["family"] == "montmory":
                    expected64 = oracle.range(seed, b.BASE, count)
                    expected = (expected64.astype(np.uint32),
                                (expected64 >> np.uint64(32)).astype(np.uint32))
                else:
                    reference_spec = k.candidates()["threefry2x32"]
                    expected = k.function(reference_spec, count)(k.arguments(reference_spec, seed, b.BASE))
                for actual, ref in zip(raw, expected):
                    np.testing.assert_array_equal(actual, ref)
                for workload in b.WORKLOADS:
                    def applied(params):
                        w = k.function(spec, count, interpret=True)(params)
                        if workload == "uniform":
                            return b.uniforms(w)
                        n = b.normals(w)
                        return n if workload == "normal" else jnp.mean(jnp.exp(-jnp.float32(0.5)*n*n))
                    result = jax.jit(applied)(p)
                    b.check(result, b.numpy_reference(tuple(np.asarray(w) for w in expected), workload), workload)
                    cases += 1
    # A synthetic paired example where ratio of aggregate medians differs from
    # median of paired ratios. Detect a change to the specified summary method.
    rows = [dict(count64=65536, workload="uniform", engine=engine, seed=str(seed), seconds=t)
            for engine, times in (("jax_original", (1, 10, 100)), ("threefry2x32", (1, 20, 50)))
            for seed, t in enumerate(times)]
    summary = b.summarize(rows)
    item = next(x for x in summary if x["engine"] == "threefry2x32")
    if item["paired_speed_ratios"]["jax_original"]["median_ratio"] != 1:
        raise AssertionError("Paired ratio computation changed")
    print(json.dumps(dict(status="pass", endpoint_checks=True, corruptions_rejected=rejected,
                          interpreted_pallas_application_cases=cases, paired_summary_checked=True,
                          native_tpu_run=False)))


if __name__ == "__main__":
    main()
