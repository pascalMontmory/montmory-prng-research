"""Exact Montmory stream, uint32 pairs only; no new split/fold_in semantics.

Words are (low32, high32). Stream v3 logical geometry is fixed at T=2048,
D=4096. Hardware block sizes do not enter the logical lane calculation.
The C reference remains the authority. Native TPU execution is NOT certified.
"""
from functools import partial

import jax
import jax.numpy as jnp

U32 = jnp.uint32
MASK32 = (1 << 32) - 1
MAX64 = (1 << 64) - 1
LANES, DRAWS = 2048, 4096


def constant(value):
    return U32(value & MASK32), U32((value >> 32) & MASK32)


def add(a, b):
    lo = a[0] + b[0]
    return lo, a[1] + b[1] + (lo < a[0]).astype(U32)


def high32(a, b, method="portable"):
    if method == "mulhi":
        return jax.lax.mulhi(a, b)
    if method != "portable":
        raise ValueError("Unknown multiplication method")
    # Exact 16-bit partial products. Each intermediate fits uint32.
    a0, a1 = a & U32(65535), a >> U32(16)
    b0, b1 = b & U32(65535), b >> U32(16)
    t = a1 * b0 + ((a0 * b0) >> U32(16))
    u = a0 * b1 + (t & U32(65535))
    return a1 * b1 + (t >> U32(16)) + (u >> U32(16))


def multiply(a, b, method="portable"):
    return (a[0] * b[0],
            high32(a[0], b[0], method) + a[1] * b[0] + a[0] * b[1])


def xor_right(a, shift):
    # All callers use fixed shifts strictly between 0 and 32.
    return (a[0] ^ ((a[0] >> U32(shift)) | (a[1] << U32(32-shift))),
            a[1] ^ (a[1] >> U32(shift)))


def splitmix(a, method):
    a = add(a, constant(0x9E3779B97F4A7C15))
    a = multiply(xor_right(a, 30), constant(0xBF58476D1CE4E5B9), method)
    a = multiply(xor_right(a, 27), constant(0x94D049BB133111EB), method)
    return xor_right(a, 31)


def round_word(word, key, odd_rotation, even_rotation, method):
    x = word[0]
    odd = (x & U32(1)) != U32(0)
    factor = jnp.where(odd, U32(3), U32(1))
    y = add((x * factor, high32(x, factor, method)), key)
    rotation = jnp.where(odd, odd_rotation, even_rotation)
    swap = (rotation & U32(32)) != U32(0)
    a = jnp.where(swap, y[1], y[0])
    b = jnp.where(swap, y[0], y[1])
    r = rotation & U32(31)  # Seed rotations are odd, so r is in 1..31.
    f = ((a << r) | (b >> (U32(32)-r))) ^ jnp.where(odd, x >> U32(1), x * U32(5))
    return word[1] ^ f, word[0]


def mix(word, method):
    rotation = word[1] >> U32(27)
    w = (word[0] ^ word[1], word[1])
    w = multiply(xor_right(w, 25), constant(0xA24BAED496EE9B5F), method)
    w = multiply(xor_right(w, 28), constant(0x9FB21C651E98DF25), method)
    w = xor_right(w, 27)
    complement = (U32(32)-rotation) & U32(31)
    # Avoid relying on compiler behavior for a shift by the word width.
    low_cross = jnp.where(rotation != U32(0), w[1] << complement, U32(0))
    high_cross = jnp.where(rotation != U32(0), w[0] << complement, U32(0))
    return (w[0] >> rotation) | low_cross, (w[1] >> rotation) | high_cross


def prepare_state(seed, method="portable"):
    """Seed-only arithmetic, reusable across every block in the same stream."""
    k0 = splitmix(seed, method)
    k1 = splitmix(k0, method)
    k2 = splitmix(k1, method)
    rotations = splitmix(k2, method)[0]
    r0 = (rotations & U32(63)) | U32(1)
    r1 = ((rotations >> U32(8)) & U32(63)) | U32(1)
    r2 = ((rotations >> U32(16)) & U32(63)) | U32(1)
    return seed, k0, k1, k2, (r0, r1, r2)


def lane_words_prepared(state, index, lane, method="portable"):
    seed, k0, k1, k2, (r0, r1, r2) = state
    twice_lane = lane[0] << U32(1), (lane[1] << U32(1)) | (lane[0] >> U32(31))
    gamma = add(constant(0x9E3779B97F4A7C15), twice_lane)
    word = add(seed, multiply(add(index, constant(1)), gamma, method))
    word = round_word(word, k0, r0, r1, method)
    word = round_word(word, k1, r1, r2, method)
    word = round_word(word, k2, r2, r0, method)
    return mix(word, method)


def lane_words(seed, index, lane, method="portable"):
    """Original per-lane engine at zero-based draw index; all args uint32 pairs."""
    return lane_words_prepared(prepare_state(seed, method), index, lane, method)


def stream_words(seed, index, method="portable"):
    # floor(index/4096) % 2048 depends only on low bits 12..22.
    lane = (index[0] >> U32(12)) & U32(2047), U32(0)
    return lane_words(seed, index, lane, method)


def mantissa_words(word):
    """Exact integer raw64 >> 11, NOT a float32 approximation."""
    return (word[0] >> U32(11)) | (word[1] << U32(21)), word[1] >> U32(11)


@partial(jax.jit, static_argnames=("count", "method"))
def _range(seed, base, *, count, method):
    offset = jnp.arange(count, dtype=U32)
    return stream_words(seed, add(base, (offset, U32(0))), method)


def validate_range(seed, base, count):
    if type(seed) is not int or not 0 <= seed <= MAX64:
        raise ValueError("seed must be an unsigned 64-bit Python integer")
    if type(base) is not int or not 0 <= base <= MAX64:
        raise ValueError("base must be an unsigned 64-bit Python integer")
    if type(count) is not int or not 0 <= count < (1 << 32):
        raise ValueError("count must be in [0, 2**32)")
    if count and count - 1 > MAX64 - base:
        raise ValueError("global index would overflow")


def generate(seed, base, count, *, method="portable"):
    """Generate a contiguous range as two uint32 arrays, without global state."""
    validate_range(seed, base, count)
    return _range(constant(seed), constant(base), count=count, method=method)


@partial(jax.jit, static_argnames=("method",))
def pack_parameters(seed, base, method="portable"):
    state = prepare_state(seed, method)
    return jnp.stack((*state[0], *state[1], *state[2], *state[3], *state[4], *base))


def unpack_parameters(p):
    state = ((p[0], p[1]), (p[2], p[3]), (p[4], p[5]), (p[6], p[7]), (p[8], p[9], p[10]))
    return state, (p[11], p[12])


def prepared_stream(state, index, method="portable"):
    lane = (index[0] >> U32(12)) & U32(2047), U32(0)
    return lane_words_prepared(state, index, lane, method)


@partial(jax.jit, static_argnames=("count", "method"))
def prepared_range(p, *, count, method="portable"):
    state, base = unpack_parameters(p)
    offset = jnp.arange(count, dtype=U32)
    return prepared_stream(state, add(base, (offset, U32(0))), method)


@partial(jax.jit, static_argnames=("count", "block_rows", "method", "interpret"))
def _pallas_range(parameters, *, count, block_rows, method, interpret):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    block_words = block_rows * 128
    blocks = (count + block_words - 1) // block_words

    def kernel(p_ref, low_ref, high_ref):
        state, base = unpack_parameters(p_ref)
        offsets = (jnp.arange(block_rows, dtype=U32)[:, None] * U32(128)
                   + jnp.arange(128, dtype=U32)[None, :]
                   + pl.program_id(0).astype(U32) * U32(block_words))
        low, high = prepared_stream(state, add(base, (offsets, U32(0))), method)
        low_ref[...] = low
        high_ref[...] = high

    spec = pl.BlockSpec((block_rows, 128), lambda i, parameters_ref: (i, 0))
    shape = jax.ShapeDtypeStruct((blocks * block_rows, 128), U32)
    low, high = pl.pallas_call(
        kernel, out_shape=(shape, shape), interpret=interpret,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(blocks,), out_specs=(spec, spec)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="montmory_uint32_pairs",
    )(parameters)
    return low.reshape(-1)[:count], high.reshape(-1)[:count]


def generate_pallas(seed, base, count, *, block_rows=8, method="portable", interpret=True):
    """Pallas prototype. Interpret mode checks semantics, not native TPU lowering."""
    validate_range(seed, base, count)
    if type(block_rows) is not int or block_rows <= 0 or block_rows % 8:
        raise ValueError("block_rows must be a positive multiple of 8")
    if not interpret and jax.default_backend() != "tpu":
        raise ValueError("native Pallas TPU mode requires a TPU backend")
    if count == 0:
        return jnp.empty((0,), U32), jnp.empty((0,), U32)
    p = pack_parameters(constant(seed), constant(base), method=method)
    return _pallas_range(p, count=count, block_rows=block_rows, method=method, interpret=interpret)
