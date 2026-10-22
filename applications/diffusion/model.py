"""Small rectified-flow image model; original code for ENINCA CONSULTING.

Gaussian-to-data interpolation: x_t=(1-t)z+t*x, target velocity x-z.
This is a flow-matching demonstration, not a Stable Diffusion reproduction.
"""
import jax
import jax.numpy as jnp
import numpy as np
from applications.stochastic import rng


def prepare(engine, implementation, x, t, p, *, rows=8, interpret=False):
    if engine not in ("montmory", "threefry") or implementation not in ("jax", "pallas", "native"):
        raise ValueError("unknown engine or implementation")
    if x.ndim != 2 or x.shape[1] != 64 or t.shape != (x.shape[0],) or x.dtype != jnp.float32 or t.dtype != jnp.float32:
        raise ValueError("require float32 images (batch,64) and times (batch,)")
    if rows not in (8,16) or (implementation == "native" and engine != "threefry"):
        raise ValueError("unsupported tile or native comparator")
    count = x.size
    if implementation == "native":
        z = jax.random.normal(jax.random.wrap_key_data(p, impl="threefry2x32"), x.shape)
        return (1-t[:, None])*z + t[:, None]*x, x-z
    if implementation == "jax":
        z = rng.normals_jax(engine, p, count).reshape(x.shape)
        return (1-t[:, None])*z + t[:, None]*x, x-z
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu
    pairs = count//2
    tile = rows*128
    blocks = (pairs+tile-1)//tile
    pad = blocks*tile-pairs
    def pack(a):
        return jnp.pad(a, (0, pad)).reshape(blocks*rows, 128)
    a, b = pack(x.reshape(-1)[0::2]), pack(x.reshape(-1)[1::2])
    times = pack(jnp.repeat(t, x.shape[1]//2))
    def body(p_ref, a_ref, b_ref, t_ref, ya_ref, yb_ref, va_ref, vb_ref):
        i = (pl.program_id(0).astype(rng.U32)*rng.U32(tile)
             + jnp.arange(rows, dtype=rng.U32)[:, None]*rng.U32(128)
             + jnp.arange(128, dtype=rng.U32)[None, :])
        za, zb = rng.normal_pair(*rng.pair_words(engine, p_ref, i))
        time = t_ref[...]
        ya_ref[...], yb_ref[...] = (1-time)*za+time*a_ref[...], (1-time)*zb+time*b_ref[...]
        va_ref[...], vb_ref[...] = a_ref[...]-za, b_ref[...]-zb
    spec = pl.BlockSpec((rows, 128), lambda i, p: (i, 0))
    shape = jax.ShapeDtypeStruct(a.shape, jnp.float32)
    ya, yb, va, vb = pl.pallas_call(body, out_shape=(shape,)*4, interpret=interpret,
        grid_spec=tpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(blocks,),
                                           in_specs=(spec,)*3, out_specs=(spec,)*4),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"flow_noise_{engine}")(p, a, b, times)
    def unpack(a, b):
        return rng.interleave(a, b)[:count].reshape(x.shape)
    return unpack(ya, yb), unpack(va, vb)


def initialize(seed, width=128):
    r = np.random.default_rng(seed)
    return tuple((jnp.asarray(r.normal(size=(a,b))*np.sqrt(1/a), dtype=jnp.float32),
                  jnp.zeros(b, jnp.float32)) for a,b in ((80,width),(width,width),(width,64)))


def predict(weights, x, t):
    angle = t[:,None]*jnp.arange(1,9,dtype=jnp.float32)[None,:]*jnp.float32(2*np.pi)
    y = jnp.concatenate((x, jnp.sin(angle), jnp.cos(angle)), axis=1)
    for i,(w,b) in enumerate(weights):
        y = y@w+b
        if i<2:
            y = jax.nn.silu(y)
    return y


def init_state(weights):
    zero = jax.tree.map(jnp.zeros_like, weights)
    return weights, zero, zero


def step_function(engine, implementation, *, interpret=False):
    @jax.jit
    def step(state, x, t, root, number):
        p = rng.at_step(engine, root, number, x.size//2)
        noisy, target = prepare(engine, implementation, x, t, p, interpret=interpret)
        w,m,v = state
        loss,grad = jax.value_and_grad(lambda w: jnp.mean((predict(w,noisy,t)-target)**2))(w)
        m = jax.tree.map(lambda a,g: .9*a+.1*g, m,grad)
        v = jax.tree.map(lambda a,g: .999*a+.001*g*g, v,grad)
        n = number.astype(jnp.float32)+1
        w = jax.tree.map(lambda a,b,c: a-1e-3*(b/(1-.9**n))/(jnp.sqrt(c/(1-.999**n))+1e-8), w,m,v)
        return (w,m,v),loss
    return step


@jax.jit
def heldout_loss(weights, x, t, z):
    return jnp.mean((predict(weights, (1-t[:,None])*z+t[:,None]*x, t)-(x-z))**2)


@jax.jit
def generate(weights, z):
    # Fixed explicit Euler integration of dx/dt=v_theta(x,t), no clipping inside.
    def body(i, x):
        t = jnp.full((x.shape[0],), i/40, jnp.float32)
        return x+predict(weights,x,t)/40
    return jax.lax.fori_loop(0,40,body,z)
