"""Freeze CPU output hashes before any TPU selection or confirmation."""
import os
os.environ["JAX_PLATFORMS"]="cpu"
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
import kernels as k
from support import Oracle, unpack

jax.config.update("jax_enable_x64",False)
jax.config.update("jax_threefry_partitionable",True)
root=Path(__file__).resolve().parent
oracle=Oracle(root.parent/"results/reference-build")
records={}
for count in k.SIZES:
    compiled={name:jax.jit(k.function(k.candidates()[name],count)) for name in (*k.VENDORS,"write_NOT_PRNG")}
    for seed in (*k.SEARCH_SEEDS,*k.CONFIRM_SEEDS):
        values=oracle.range(seed,k.BASE,count)
        records[f"montmory/{count}/{seed}"]=hashlib.sha256(values.astype("<u8",copy=False).tobytes()).hexdigest()
        del values
        for name,fn in compiled.items():
            values=unpack(fn(k.arguments(k.candidates()[name],seed)))
            records[f"{name}/{count}/{seed}"]=hashlib.sha256(values.astype("<u8",copy=False).tobytes()).hexdigest()
            del values
        print(f"CPU references {count}, seed {seed}: PASS",flush=True)
    jax.clear_caches()
(root/"reference/optimization-hashes.json").write_text(json.dumps(dict(jax=jax.__version__,jaxlib=jax.lib.__version__,
    base=k.BASE,sizes=k.SIZES,search_seeds=k.SEARCH_SEEDS,confirm_seeds=k.CONFIRM_SEEDS,sha256=records,
    scope="Montmory C reference; competitors JAX CPU API, not independent Random123 KATs"),indent=2)+"\n")
