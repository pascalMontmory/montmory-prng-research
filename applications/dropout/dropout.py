"""Fixed-rate dropout experiment; original PRNG integer engines are unchanged.

Developed by Pascal Montmory for ENINCA CONSULTING. See ../../LICENSE.
Threefry attribution and the imported implementation remain in the frozen kit.
"""
from functools import lru_cache
from pathlib import Path
import math
import sys

import jax
import jax.numpy as jnp

KIT = Path(__file__).resolve().parents[2] / "studies/tpu-opt-2026-10-21/montmory-tpu-opt"
sys.path.insert(0, str(KIT))
import kernels as k
import montmory_jax as m

KEEP = 0.75
THRESHOLD = 0xC0000000
BASE = (1 << 32) - 4097
U32 = jnp.uint32


def root_parameters(engine, seed, base=BASE):
    if not 0 <= seed < 1 << 64 or not 0 <= base < 1 << 64:
        raise ValueError("seed/base must be unsigned 64-bit integers")
    if engine == "montmory":
        return m.pack_parameters(m.constant(seed), m.constant(base))
    if engine != "threefry":
        raise ValueError(engine)
    key = jax.random.key(U32(seed & m.MASK32), impl="threefry2x32")
    return jax.random.key_data(jax.random.fold_in(key, U32(seed >> 32)))


def layer_parameters(engine, root, step, layer, stride_words):
    """Two dropout sites per step; Montmory reserves distinct index intervals.

    Caller validates step in [0, 2**32), layer in {0,1}, and no uint64
    interval overflow. stride_words >= ceil(activation_count / 2).
    """
    if engine == "threefry":
        key = jax.random.wrap_key_data(root, impl="threefry2x32")
        key = jax.random.fold_in(jax.random.fold_in(key, step), U32(layer))
        return jax.random.key_data(key)
    block = m.add(m.add((step, U32(0)), (step, U32(0))), m.constant(layer))
    offset = m.multiply(block, m.constant(stride_words), "portable")
    base = m.add((root[11], root[12]), offset)
    return root.at[11].set(base[0]).at[12].set(base[1])


def mask_at(engine, parameters, offsets):
    if engine == "threefry":
        bits = k.threefry_bits_at(parameters, offsets)
    else:
        state, base = m.unpack_parameters(parameters)
        # Each unchanged 64-bit word supplies low32 then high32 to two elements.
        index = m.add(base, (offsets >> U32(1), U32(0)))
        low, high = k.stream(state, index, carry=True, method="portable")
        bits = jnp.where((offsets & U32(1)) == 0, low, high)
    return bits < U32(THRESHOLD)


def reference_mask(engine, parameters, shape):
    if engine == "threefry":
        key = jax.random.wrap_key_data(parameters, impl="threefry2x32")
        return jax.random.bernoulli(key, KEEP, shape, mode="low")
    return mask_at(engine, parameters, jnp.arange(math.prod(shape), dtype=U32)).reshape(shape)


def reference_dropout(engine, x, parameters):
    return jnp.where(reference_mask(engine, parameters, x.shape), x / KEEP, jnp.zeros_like(x))


def pallas_dropout(engine, x, parameters, rows, interpret):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu

    count = math.prod(x.shape)
    if x.dtype != jnp.float32 or not 0 < count < (1 << 31):
        raise ValueError("this experiment requires float32 and 0 < elements < 2**31")
    tile = rows * 128
    blocks = (count + tile - 1) // tile
    padding = blocks * tile - count
    flat = jnp.pad(x.reshape(-1), (0, padding)).reshape(blocks * rows, 128)

    def body(p_ref, x_ref, y_ref):
        offsets = (pl.program_id(0).astype(U32) * U32(tile)
                   + jnp.arange(rows, dtype=U32)[:, None] * U32(128)
                   + jnp.arange(128, dtype=U32)[None, :])
        mask = mask_at(engine, p_ref, offsets)
        y_ref[...] = jnp.where(mask, x_ref[...] / KEEP, jnp.zeros_like(x_ref[...]))

    spec = pl.BlockSpec((rows, 128), lambda i, p: (i, 0))
    result = pl.pallas_call(
        body, out_shape=jax.ShapeDtypeStruct(flat.shape, x.dtype), interpret=interpret,
        grid_spec=tpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(blocks,),
                                           in_specs=(spec,), out_specs=spec),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"{engine}_dropout_{rows}")(parameters, flat)
    return result.reshape(-1)[:count].reshape(x.shape)


@lru_cache(maxsize=None)
def make_dropout(engine, implementation, rows=32, interpret=False):
    if engine not in ("montmory", "threefry") or rows not in (8, 16, 32, 64, 128):
        raise ValueError("unknown engine or tile")
    if implementation == "jax":
        return lambda x, p: reference_dropout(engine, x, p)
    if implementation != "pallas":
        raise ValueError(implementation)

    @jax.custom_vjp
    def apply(x, p):
        return pallas_dropout(engine, x, p, rows, interpret)

    def forward(x, p):
        return pallas_dropout(engine, x, p, rows, interpret), p

    def backward(p, gradient):
        # Regenerate exactly the forward mask. The dropout residual is only p.
        return pallas_dropout(engine, gradient, p, rows, interpret), None

    apply.defvjp(forward, backward)
    return apply
