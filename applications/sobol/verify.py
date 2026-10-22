"""Cross-check SciPy, host recurrence, JAX and Pallas, including high indices."""
import numpy as np
import jax

import kernels as k
import reference as ref


def verify(backend):
    interpret = backend == "cpu"
    report = {"status": "running", "backend": backend,
              "pallas_execution": "CPU interpreter" if interpret else "native TPU",
              "scipy_public_words": 0, "accelerator_words": 0,
              "recurrence_words": 0, "cases": [], "statistical_tests_run": False}
    starts = [0, 1, 127, 65535, (1 << 31) - 256, (1 << 32) - 513]
    for seed in (None, 0, 42, (1 << 64) - 1):
        p = ref.parameters(seed)
        public = ref.scipy_words(seed, 4096)
        np.testing.assert_array_equal(ref.words(p, 0, 4096), public)
        report["scipy_public_words"] += public.size
        if seed is None:
            known = np.array([[0, 0, 0], [4, 4, 4], [6, 2, 2], [2, 6, 6],
                              [3, 3, 5], [7, 7, 1], [5, 1, 7], [1, 5, 3]], dtype=np.uint32)
            np.testing.assert_array_equal(public[:8, :3], known << np.uint32(29))
        v, shift = jax.device_put((p.directions, p.shift))
        tables = {t: jax.device_put(ref.local_table(p, t)) for t in (128, 256, 512)}
        for start in starts:
            n = 513
            expected = ref.recurrence_words(p, start, n)
            np.testing.assert_array_equal(expected, ref.words(p, start, n))
            report["recurrence_words"] += expected.size
            for name in k.VARIANTS:
                got = k.make_kernel(name, n, interpret=interpret)(
                    v, shift, tables[k.tile_for(name)], np.array([start], dtype=np.uint32))
                got = np.asarray(got)
                np.testing.assert_array_equal(got, expected, err_msg=f"{name}, {seed}, {start}")
                report["accelerator_words"] += got.size
            report["cases"].append({"seed": seed, "start": start, "points": n, "variants": len(k.VARIANTS)})
        print(f"Conformance: seed={seed}, SciPy prefix and boundary indices PASS", flush=True)

    # Irregular tails, aligned fast path, arbitrary partitions and final point.
    p = ref.parameters(42)
    v, shift = jax.device_put((p.directions, p.shift))
    tables = {t: jax.device_put(ref.local_table(p, t)) for t in (128, 256, 512)}
    for name in k.VARIANTS:
        table = tables[k.tile_for(name)]
        for n in (1, 127, 128, 129, 512):
            start = 1 << 31
            got = np.asarray(k.make_kernel(name, n, aligned=True, interpret=interpret)(
                v, shift, table, np.array([start], dtype=np.uint32)))
            np.testing.assert_array_equal(got, ref.words(p, start, n))
            report["accelerator_words"] += got.size
        last = np.asarray(k.make_kernel(name, 1, interpret=interpret)(
            v, shift, table, np.array([(1 << 32) - 1], dtype=np.uint32)))
        np.testing.assert_array_equal(last, ref.recurrence_words(p, (1 << 32) - 1, 1))
        report["accelerator_words"] += last.size
        start = (1 << 31) - 137
        pieces = []
        for n in (1, 127, 129, 512):
            piece = np.asarray(k.make_kernel(name, n, interpret=interpret)(
                v, shift, table, np.array([start], dtype=np.uint32)))
            pieces.append(piece)
            start += n
        joined = np.concatenate(pieces)
        np.testing.assert_array_equal(joined, ref.words(p, (1 << 31) - 137, 769))
        report["accelerator_words"] += joined.size
    raw = ref.words(p, (1 << 32) - 1024, 1024)
    raw[0, :4] = [0, 1, 0xffffff00, 0xffffffff]
    mapped = np.asarray(jax.jit(k.uniform24)(raw))
    expected = (raw >> np.uint32(8)).astype(np.float32) * np.float32(2.0 ** -24)
    np.testing.assert_array_equal(mapped.view(np.uint32), expected.view(np.uint32))
    assert np.all((mapped >= 0) & (mapped < 1))
    for start, n in ((-1, 1), (0, 0), ((1 << 32) - 1, 2), (1 << 32, 1), (True, 1)):
        try:
            ref.check_range(start, n)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid range accepted")
    report.update(status="pass", partition_checks=len(k.VARIANTS),
                  invalid_range_checks=5, uniform24_words=int(raw.size))
    return report
