"""Check global stream indices under JAX device partitioning, without split keys."""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import montmory_jax as m
from support import Oracle, assert_equal, unpack, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    oracle = Oracle(args.out)
    report = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.local_devices()],
              "status": "running", "cases": [], "split_fold_in_tested": False}
    try:
        for number in (1, 2, 4):
            if number > len(jax.local_devices()):
                continue
            mesh = Mesh(np.array(jax.local_devices()[:number]), ("devices",))
            sharding = NamedSharding(mesh, P("devices"))
            replicated = NamedSharding(mesh, P())
            def body(lo, hi, seed):
                return m.stream_words((seed[0], seed[1]), (lo, hi), method="portable")
            fn = jax.jit(body, in_shardings=(sharding, sharding, replicated), out_shardings=(sharding, sharding))
            for seed in (42, m.MAX64):
                for base in (4093, 8388607, (1 << 32)-7, m.MAX64-32767):
                    ix = np.arange(32768, dtype=np.uint64) + np.uint64(base)
                    lo = jax.device_put(ix.astype(np.uint32), sharding)
                    hi = jax.device_put((ix >> np.uint64(32)).astype(np.uint32), sharding)
                    key = jax.device_put(np.array([seed & m.MASK32, seed >> 32], np.uint32), replicated)
                    actual = fn(lo, hi, key)
                    if len(actual[0].addressable_shards) != number:
                        raise AssertionError("Unexpected shard count")
                    assert_equal(unpack(actual), oracle.indices(seed, ix), "sharded stream")
                    report["cases"].append({"devices": number, "seed": seed, "base": base, "words": 32768})
            print(f"Partition sur {number} devices {jax.default_backend()}: PASS.", flush=True)
        report["status"] = "pass"
    except BaseException as error:
        report.update(status="fail", error=repr(error))
        raise
    finally:
        report["word_comparisons"] = sum(r["words"] for r in report["cases"])
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
