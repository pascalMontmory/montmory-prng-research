"""Application mappings over unchanged Montmory/Threefry integer engines.

Pair i uses Montmory word base+i (low then high), or JAX partitionable
Threefry uint32 positions 2*i and 2*i+1. These are application contracts,
not new PRNG algorithms. Developed for ENINCA CONSULTING; see LICENSE.
"""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np

KIT = Path(__file__).resolve().parents[2] / "studies/tpu-opt-2026-10-21/montmory-tpu-opt"
sys.path.insert(0, str(KIT))
import kernels as k
import montmory_jax as m
from support import Oracle

U32 = jnp.uint32
BASE = (1 << 32) - 4097


def parameters(engine, seed, base=BASE):
    if engine not in ("montmory", "threefry") or not 0 <= seed < 1 << 64 or not 0 <= base < 1 << 64:
        raise ValueError("engine or uint64 seed/base outside contract")
    if engine == "montmory":
        return m.pack_parameters(m.constant(seed), m.constant(base))
    key = jax.random.key(U32(seed & m.MASK32), impl="threefry2x32")
    return jax.random.key_data(jax.random.fold_in(key, U32(seed >> 32)))


def at_step(engine, p, step, pairs):
    if engine == "threefry":
        return jax.random.key_data(jax.random.fold_in(jax.random.wrap_key_data(p, impl="threefry2x32"), step))
    base = m.add((p[11], p[12]), m.multiply((step, U32(0)), m.constant(pairs), "portable"))
    return p.at[11].set(base[0]).at[12].set(base[1])


def pair_words(engine, p, indices):
    if engine == "threefry":
        return k.threefry_bits_at(p, indices * U32(2)), k.threefry_bits_at(p, indices * U32(2) + U32(1))
    state, base = m.unpack_parameters(p)
    return k.stream(state, m.add(base, (indices, U32(0))), carry=True, method="portable")


def raw_jax(engine, p, pairs):
    if not 0 < pairs < 1 << 30:
        raise ValueError("require 0 < pairs < 2**30")
    if engine == "threefry":
        bits = jax.random.bits(jax.random.wrap_key_data(p, impl="threefry2x32"), (pairs, 2), dtype=U32)
        return bits[:, 0], bits[:, 1]
    return pair_words(engine, p, jnp.arange(pairs, dtype=U32))


def uniform(word):
    # Exact midpoint float32 grid with 23 bits; excludes both endpoints.
    return ((word >> U32(9)).astype(jnp.float32) + jnp.float32(.5)) * jnp.float32(2**-23)


def normal_pair(lo, hi):
    radius = jnp.sqrt(-jnp.float32(2) * jnp.log(uniform(lo)))
    angle = jnp.float32(2 * np.pi) * uniform(hi)
    return radius * jnp.cos(angle), radius * jnp.sin(angle)


def interleave(a, b):
    return jnp.stack((a, b), axis=-1).reshape(-1)


def normals_jax(engine, p, count):
    return interleave(*normal_pair(*raw_jax(engine, p, (count + 1)//2)))[:count]


def host_words(engine, seed, base, pairs, oracle):
    if engine == "montmory":
        raw = oracle.range(seed, base, pairs)
        return raw.astype(np.uint32), (raw >> np.uint64(32)).astype(np.uint32)
    p = parameters(engine, seed, base)
    bits = np.asarray(jax.random.bits(jax.random.wrap_key_data(p, impl="threefry2x32"), (pairs, 2), dtype=U32))
    return bits[:, 0], bits[:, 1]


def host_normals(lo, hi):
    u, v = [((w >> np.uint32(9)).astype(np.float64) + .5) * 2**-23 for w in (lo, hi)]
    radius = np.sqrt(-2*np.log(u))
    angle = float(np.float32(2*np.pi))*v
    return np.stack((radius*np.cos(angle), radius*np.sin(angle)), axis=-1).reshape(-1)
