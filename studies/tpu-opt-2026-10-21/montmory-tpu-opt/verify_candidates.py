"""Bit-exact checks of new candidates against the unchanged local C oracle."""
import argparse
import json
import os
from pathlib import Path
import sys


def verify(out, interpret):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import kernels as k
    import montmory_jax as m
    from support import Oracle, unpack, assert_equal
    out.mkdir(parents=True, exist_ok=True)
    oracle = Oracle(out)
    report = dict(status="running", backend=jax.default_backend(), interpreted_pallas=interpret,
                  words_compared=0, candidates=[], component_values=0, compile_rejections=[])
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        rng = np.random.default_rng(20261021)
        edges = np.array([0,1,2,3,0x55555554,0x55555555,0x55555556,0x7ffffffe,
                          0x7fffffff,0x80000000,0x80000001,0xaaaaaaa9,0xaaaaaaaa,
                          0xaaaaaaab,0xfffffffe,0xffffffff], dtype=np.uint32)
        x = np.concatenate((edges, rng.integers(0, 1 << 32, 1048576, dtype=np.uint32)))
        expected = x.astype(np.uint64)*(1+2*(x.astype(np.uint64)&1))
        assert_equal(unpack(jax.jit(k.small_product)(jnp.asarray(x))), expected, "exact small product")
        report["component_values"] = len(x)
        cases = ((0,4095), (42,(1 << 32)-7), (m.MAX64,m.MAX64-8192),
                 ((1 << 63)+1,8388607), (0xD1B54A32D192ED03,0), (19,(1 << 63)-7))
        count = 8193
        expected_cases = {(s,b): oracle.range(s,b,count) for s,b in cases}
        for name, spec in k.candidates().items():
            if spec["kind"] in ("vendor", "write"):
                continue
            print(f"Conformite {name}...", flush=True)
            arg = k.arguments(spec, 42)
            try:
                fn = jax.jit(k.function(spec, count, interpret)).lower(arg).compile()
            except Exception as error:
                if interpret or name in ("jax_original", "pallas_original_32"):
                    raise
                report["compile_rejections"].append(dict(engine=name, error=repr(error)))
                print(f"  Compilation refusee, candidat exclu: {error}", flush=True)
                save()
                continue
            if spec["family"] == "montmory":
                for seed, base in cases:
                    words = fn(k.arguments(spec, seed, base))
                    expected = expected_cases[seed,base]
                    assert_equal(unpack(words), expected, f"{name} raw")
                    assert_equal(unpack(m.mantissa_words(words)), expected >> np.uint64(11), f"{name} mantissa")
                    report["words_compared"] += 2*count
            else:
                vendor = k.candidates()["threefry2x32"]
                reference = jax.jit(k.function(vendor, count))
                for seed in (0, 42, m.MAX64):
                    assert_equal(unpack(fn(k.arguments(spec,seed))), unpack(reference(k.arguments(vendor,seed))), name)
                    report["words_compared"] += count
            report["candidates"].append(name)
            save()
        # Check tails at sizes distinct from the common 8193-word block.
        for name in ("jax_carry", "pallas_carry_32", "threefry_pallas_32"):
            if name not in report["candidates"]:
                continue
            spec = k.candidates()[name]
            for count in (1,127,4097):
                actual = unpack(jax.jit(k.function(spec,count,interpret))(k.arguments(spec,42)))
                expected = (oracle.range(42,k.BASE,count) if spec["family"] == "montmory" else
                            unpack(jax.jit(k.function(k.candidates()["threefry2x32"],count))(
                                k.arguments(k.candidates()["threefry2x32"],42))))
                assert_equal(actual,expected,f"{name} tail {count}")
                report["words_compared"] += count
        report["status"] = "pass"
        return report
    except BaseException as error:
        report.update(status="fail", error=repr(error))
        raise
    finally:
        save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu","tpu"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args=p.parse_args()
    os.environ["JAX_PLATFORMS"]=args.backend
    import jax
    jax.config.update("jax_enable_x64",False)
    jax.config.update("jax_threefry_partitionable",True)
    if jax.default_backend()!=args.backend:
        raise RuntimeError("Backend mismatch")
    report=verify(args.out,args.backend=="cpu")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
