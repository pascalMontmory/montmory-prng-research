"""Independent C/JAX mask checks and forward/backward/training-step checks."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

import dropout as d
import model
from support import Oracle, assert_equal


def c_mask(oracle, seed, base, count):
    words = oracle.range(seed, base, (count + 1) // 2)
    halves = np.empty(words.size * 2, dtype=np.uint32)
    halves[0::2] = (words & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    halves[1::2] = (words >> np.uint64(32)).astype(np.uint32)
    return halves[:count] < d.THRESHOLD


def run(out, *, interpret, selected_rows=32):
    oracle = Oracle(Path(out) / "oracle")
    report = {"mode": "pallas_cpu_interpretation" if interpret else "native_tpu",
              "mask_elements": 0, "gradient_elements": 0, "cases": [],
              "training_step_comparisons": []}
    cases = ((0, 0, 1), (42, (1 << 32) - 7, 4097),
             ((1 << 64) - 1, (1 << 64) - 8192, 8193),
             ((1 << 63) + 1, 8388607, 127))
    for engine in ("threefry", "montmory"):
        for seed, base, count in cases:
            p = d.root_parameters(engine, seed, base)
            if engine == "montmory":
                expected = c_mask(oracle, seed, base, count)
            else:
                key = jax.random.wrap_key_data(p, impl="threefry2x32")
                expected = np.asarray(jax.random.bernoulli(key, d.KEEP, (count,), mode="low"))
            assert_equal(d.reference_mask(engine, p, (count,)), expected, "reference mask")
            tiles = sorted({8, 32, selected_rows})
            for rows in tiles:
                drop = d.make_dropout(engine, "pallas", rows, interpret)
                # A dropout of ones exposes the actual fused kernel's mask.
                output = jax.jit(drop)(jnp.ones(count, jnp.float32), p)
                assert_equal(np.asarray(output) != 0, expected, "fused mask")
                x = jnp.linspace(-1, 1, count, dtype=jnp.float32)
                upstream = jnp.linspace(-0.3, 0.7, count, dtype=jnp.float32)
                actual = jax.jit(drop)(x, p)
                reference = np.where(expected, np.asarray(x) / np.float32(d.KEEP), np.float32(0))
                np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-7)
                gradient = jax.jit(jax.grad(lambda z: jnp.vdot(drop(z, p), upstream)))(x)
                expected_gradient = np.where(expected, np.asarray(upstream) / np.float32(d.KEEP), 0)
                np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-7, atol=1e-7)
                report["mask_elements"] += count
                report["gradient_elements"] += count
            report["cases"].append({"engine": engine, "seed": seed, "base": base,
                                    "count": count, "tiles": tiles})
        print(f"{engine}: masks, tails, tile invariance and gradients PASS", flush=True)

    # Check the experiment's step/layer address schedule, including 32-bit carries.
    root = d.root_parameters("montmory", 42)
    for step in (0, 1, (1 << 32) - 1):
        for layer in (0, 1):
            p = d.layer_parameters("montmory", root, jnp.uint32(step), layer, 4097)
            actual = np.asarray(p)
            base = d.BASE + (step * 2 + layer) * 4097
            assert int(actual[11]) | (int(actual[12]) << 32) == base
            expected = c_mask(oracle, 42, base, 131)
            assert_equal(d.reference_mask("montmory", p, (131,)), expected, "layer schedule")
    report["schedule_checks"] = 6

    (x, y), _ = model.dataset()
    x, y = jnp.asarray(x[:16]), jnp.asarray(y[:16])
    for engine in ("threefry", "montmory"):
        root = d.root_parameters(engine, 19)
        weights = model.initialize(32, 7)
        reference = model.training_step(engine, "jax")
        fused = model.training_step(engine, "pallas", rows=8, interpret=interpret)
        a = reference(weights, x, y, root, jnp.uint32(17))
        b = fused(weights, x, y, root, jnp.uint32(17))
        worst = 0.0
        for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-6)
            worst = max(worst, float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
        report["training_step_comparisons"].append({"engine": engine, "max_absolute_difference": worst})
        print(f"{engine}: full MLP update matches reference (max abs {worst:.3g})", flush=True)
    report["status"] = "pass"
    return report
