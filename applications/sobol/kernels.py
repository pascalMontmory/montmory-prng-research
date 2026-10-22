"""Exact Sobol32 JAX and experimental Pallas TPU kernels, 128 dimensions.

For aligned b and 0 <= r < B, B a power of two:
  Gray(b+r) = Gray(b) XOR Gray(r).
The block kernel computes the common term once per block, not once per point.
This standard digital-net identity is not a new Sobol construction.

Developed by Pascal Montmory for ENINCA CONSULTING. See ../../LICENSE.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp

DIMENSIONS = 128
U32 = jnp.uint32
VARIANTS = ("jax_gray", "jax_block_128", "jax_block_256", "jax_block_512",
            "pallas_gray_32", "pallas_block_128", "pallas_block_256", "pallas_block_512")


def tile_for(name):
    if name not in VARIANTS:
        raise ValueError(name)
    return int(name.rsplit("_", 1)[1]) if "block" in name else 128


def gray_words(directions, shift, indices):
    gray = indices ^ (indices >> U32(1))
    out = jnp.broadcast_to(shift, (*indices.shape, DIMENSIONS))
    for bit in range(32):
        out = out ^ jnp.where(((gray >> U32(bit)) & U32(1))[..., None] != 0,
                              directions[bit], U32(0))
    return out


def jax_block(directions, shift, table, base, blocks, tile):
    indices = base + jnp.arange(blocks, dtype=U32) * U32(tile)
    gray = indices ^ (indices >> U32(1))
    common = jnp.broadcast_to(shift, (blocks, DIMENSIONS))
    # An aligned binary base can have a Gray bit at log2(B)-1.
    for bit in range(tile.bit_length() - 2, 32):
        common ^= jnp.where(((gray >> U32(bit)) & U32(1))[:, None] != 0,
                            directions[bit], U32(0))
    return (common[:, None, :] ^ table[None, :, :]).reshape(blocks * tile, DIMENSIONS)


def pallas_words(directions, shift, table, base, count, tile, block, interpret):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu

    blocks = (count + tile - 1) // tile

    def body(base_ref, v_ref, s_ref, t_ref, out_ref):
        index = base_ref[0] + pl.program_id(0).astype(U32) * U32(tile)
        if block:
            gray = index ^ (index >> U32(1))
            common = s_ref[0, :]
            for bit in range(tile.bit_length() - 2, 32):
                common ^= jnp.where(((gray >> U32(bit)) & U32(1)) != 0,
                                    v_ref[bit, :], U32(0))
            out_ref[...] = t_ref[...] ^ common[None, :]
        else:
            indices = index + jnp.arange(tile, dtype=U32)
            gray = indices ^ (indices >> U32(1))
            value = jnp.broadcast_to(s_ref[0, :], (tile, DIMENSIONS))
            for bit in range(32):
                value ^= jnp.where(((gray >> U32(bit)) & U32(1))[:, None] != 0,
                                   v_ref[bit, :], U32(0))
            out_ref[...] = value

    # Scalar base in SMEM; direction/shift/table tiles in VMEM. All are inputs,
    # not compile-time seed-specific constants.
    out_spec = pl.BlockSpec((tile, DIMENSIONS), lambda i, base: (i, 0))
    specs = (pl.BlockSpec((32, DIMENSIONS), lambda i, base: (0, 0)),
             pl.BlockSpec((1, DIMENSIONS), lambda i, base: (0, 0)),
             pl.BlockSpec((tile, DIMENSIONS), lambda i, base: (0, 0)))
    return pl.pallas_call(
        body, out_shape=jax.ShapeDtypeStruct((blocks * tile, DIMENSIONS), U32),
        grid_spec=tpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(blocks,),
                                           in_specs=specs, out_specs=out_spec),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret, name=f"sobol_{'block' if block else 'gray'}_{tile}")(
            jnp.reshape(base, (1,)), directions, shift[None, :], table[:tile])


@lru_cache(maxsize=None)
def make_kernel(name, count, aligned=False, interpret=False):
    """Return a jitted function(v, shift, local_table, uint32 start[1]).

    Validate logical ranges on the host before calling. aligned=True is only
    legal for starts divisible by the variant's tile (direct paths: no rule).
    Padding may wrap past 2**32, but those points are always discarded.
    """
    if name not in VARIANTS or type(count) is not int or count < 1:
        raise ValueError("unknown variant or invalid count")
    block = "block" in name
    tile = tile_for(name) if block else (32 if name.startswith("pallas") else 128)

    def generate(v, shift, table, start):
        base = start[0]
        if name == "jax_gray":
            return gray_words(v, shift, base + jnp.arange(count, dtype=U32))
        offset = U32(0) if aligned or not block else base & U32(tile - 1)
        base = base - offset
        padded = ((count + (0 if aligned or not block else tile - 1) + tile - 1) // tile) * tile
        if name.startswith("jax"):
            out = jax_block(v, shift, table, base, padded // tile, tile)
        else:
            out = pallas_words(v, shift, table, base, padded, tile, block, interpret)
        if aligned or not block:
            return out[:count]
        return jax.lax.dynamic_slice(out, (offset.astype(jnp.int32), jnp.int32(0)),
                                     (count, DIMENSIONS))

    return jax.jit(generate)


def uniform24(raw):
    """Optional exact float32 dyadic mapping in [0,1); discards low eight bits.

    This is NOT SciPy's full 32-bit float64 coordinate conversion.
    """
    return (raw >> U32(8)).astype(jnp.float32) * jnp.float32(2.0 ** -24)
