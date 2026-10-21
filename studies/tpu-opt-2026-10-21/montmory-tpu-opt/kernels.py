"""Equivalent Montmory candidates and an API-equivalent Threefry2x32 comparator."""
from functools import partial
import jax
import jax.numpy as jnp
import montmory_jax as m

U32 = jnp.uint32
ROWS = (8, 16, 32, 64, 128, 256)
BASE = (1 << 32) - 4097
SIZES = (65536, 1048576, 8388608, 16777216)
SEARCH_SEEDS = (0, 42, (1 << 64)-1)
CONFIRM_SEEDS = (7, 19, 20261021, 0x13579BDF2468ACE0, 0xDEADBEEF01234567,
                 (1 << 64)-4, (1 << 63)+12345)
VENDORS = ("threefry2x32", "threefry4x32", "philox2x32", "philox4x32")


def small_product(x):
    """Exact x*(1 if even else 3), returned as low/high uint32."""
    parity = x & U32(1)
    low = x * (U32(1) + (parity << U32(1)))
    high = ((x >> U32(31)) & parity) + (low < x).astype(U32)
    return low, high


def round_carry(word, key, odd_rotation, even_rotation, method):
    x = word[0]
    odd = (x & U32(1)) != U32(0)
    y = m.add(small_product(x), key)
    rotation = jnp.where(odd, odd_rotation, even_rotation)
    swap = (rotation & U32(32)) != U32(0)
    a, b = jnp.where(swap, y[1], y[0]), jnp.where(swap, y[0], y[1])
    r = rotation & U32(31)
    f = ((a << r) | (b >> (U32(32)-r))) ^ jnp.where(odd, x >> U32(1), x * U32(5))
    return word[1] ^ f, word[0]


def stream(state, index, *, carry, method):
    if not carry:
        return m.prepared_stream(state, index, method)
    seed, k0, k1, k2, (r0, r1, r2) = state
    lane = (index[0] >> U32(12)) & U32(2047)
    gamma = m.add(m.constant(0x9E3779B97F4A7C15), (lane << U32(1), U32(0)))
    word = m.add(seed, m.multiply(m.add(index, m.constant(1)), gamma, method))
    for key, ro, re in ((k0, r0, r1), (k1, r1, r2), (k2, r2, r0)):
        word = round_carry(word, key, ro, re, method)
    return m.mix(word, method)


def jax_range(p, *, count, carry=False, method="portable"):
    state, base = m.unpack_parameters(p)
    index = m.add(base, (jnp.arange(count, dtype=U32), U32(0)))
    return stream(state, index, carry=carry, method=method)


def threefry_bits_at(key, index):
    """20-round Threefry2x32; JAX partitionable uint32 output is XOR of both words.

    Algorithm: Salmon et al., Random123; rotation schedule also documented in
    jax/_src/random/threefry2x32.py (Apache-2.0). No reduced-round comparator.
    Here global positions are below 2**32, hence the high counter word is zero.
    """
    keys = (key[0], key[1], key[0] ^ key[1] ^ U32(0x1BD11BDA))
    x, y = jnp.zeros_like(index) + keys[0], index + keys[1]
    rotations = ((13, 15, 26, 6), (17, 29, 16, 24))
    for group in range(5):
        for r in rotations[group % 2]:
            x = x + y
            y = ((y << U32(r)) | (y >> U32(32-r))) ^ x
        x = x + keys[(group+1) % 3]
        y = y + keys[(group+2) % 3] + U32(group+1)
    return x ^ y


def pallas_range(p, *, count, rows, carry=False, threefry=False, interpret=False):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu
    words = rows*128
    blocks = (count+words-1)//words
    def body(p_ref, low_ref, high_ref):
        offset = (jnp.arange(rows, dtype=U32)[:, None]*U32(128)
                  + jnp.arange(128, dtype=U32)[None, :] + pl.program_id(0).astype(U32)*U32(words))
        if threefry:
            key = (p_ref[0], p_ref[1])
            lo, hi = threefry_bits_at(key, offset), threefry_bits_at(key, offset+U32(count))
        else:
            state, base = m.unpack_parameters(p_ref)
            lo, hi = stream(state, m.add(base, (offset, U32(0))), carry=carry, method="portable")
        low_ref[...], high_ref[...] = lo, hi
    spec = pl.BlockSpec((rows, 128), lambda i, p_ref: (i, 0))
    shape = jax.ShapeDtypeStruct((blocks*rows, 128), U32)
    lo, hi = pl.pallas_call(body, out_shape=(shape, shape), interpret=interpret,
        grid_spec=tpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(blocks,), out_specs=(spec, spec)),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
        name="threefry20" if threefry else "montmory_carry" if carry else "montmory_original")(p)
    return lo.reshape(-1)[:count], hi.reshape(-1)[:count]


def candidates():
    result = {"jax_original": dict(family="montmory", kind="jax", carry=False, method="portable"),
              "jax_mulhi": dict(family="montmory", kind="jax", carry=False, method="mulhi"),
              "jax_carry": dict(family="montmory", kind="jax", carry=True, method="portable"),
              "jax_carry_mulhi": dict(family="montmory", kind="jax", carry=True, method="mulhi")}
    for rows in ROWS:
        for carry in (False, True):
            name = f"pallas_{'carry' if carry else 'original'}_{rows}"
            result[name] = dict(family="montmory", kind="pallas", rows=rows, carry=carry)
    for rows in (16, 32, 64, 128):
        result[f"threefry_pallas_{rows}"] = dict(family="threefry2x32", kind="pallas", rows=rows, threefry=True)
    for name in VENDORS:
        result[name] = dict(family=name, kind="vendor")
    result["write_NOT_PRNG"] = dict(family="write_NOT_PRNG", kind="write")
    result["rbg_same_backend"] = dict(family="rbg", kind="vendor")
    return result


def arguments(spec, seed, base=BASE):
    if spec["family"] == "montmory":
        value = m.pack_parameters(m.constant(seed), m.constant(base))
    elif spec["kind"] == "write":
        value = jnp.array([seed & m.MASK32, seed >> 32], dtype=U32)
    else:
        value = jax.random.key(U32(seed & m.MASK32), impl=spec["family"])
        value = jax.random.fold_in(value, U32(seed >> 32))
        if spec["kind"] == "pallas":
            value = jax.random.key_data(value)
    return jax.block_until_ready(value)


def function(spec, count, interpret=False):
    if spec["kind"] == "jax":
        return partial(jax_range, count=count, carry=spec["carry"], method=spec["method"])
    if spec["kind"] == "pallas":
        return partial(pallas_range, count=count, rows=spec["rows"], carry=spec.get("carry", False),
                       threefry=spec.get("threefry", False), interpret=interpret)
    if spec["kind"] == "write":
        def write(p):
            x = jnp.arange(count, dtype=U32)+p[0]
            return x ^ p[1], x+p[1]
        return write
    def bits(key):
        out = jax.random.bits(key, (2, count), dtype=U32)
        return out[0], out[1]
    return bits
