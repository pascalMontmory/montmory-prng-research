#!/usr/bin/env python3
"""Boundary and rejection checks for the two new application contracts."""
import os
from pathlib import Path
import sys
os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jnp
import numpy as np
from applications.stochastic import rng
from applications.diffusion import model
from applications.asian_option import pricing


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid input accepted')


def main():
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_threefry_partitionable',True)
    endpoints=np.asarray(rng.uniform(jnp.array([0,2**32-1],dtype=jnp.uint32)))
    np.testing.assert_array_equal(endpoints,np.array([2**-24,1-2**-24],dtype=np.float32))
    assert np.isfinite(np.asarray(jax.scipy.special.ndtri(endpoints))).all()
    for engine in ('montmory','threefry'):
        for seed in (-1,2**64):reject(lambda:rng.parameters(engine,seed))
        for base in (-1,2**64):reject(lambda:rng.parameters(engine,0,base))
    p=rng.parameters('threefry',42)
    for count in (0,2**30):reject(lambda:rng.raw_jax('threefry',p,count))
    x=jnp.ones((2,64),jnp.float32); t=jnp.array([.2,.8],jnp.float32)
    reject(lambda:model.prepare('threefry','jax',x[:,:63],t,p))
    reject(lambda:model.prepare('threefry','jax',x,t[:1],p))
    reject(lambda:model.prepare('montmory','native',x,t,p))
    # Linear interpolation identities must hold independently of the RNG.
    y0,target0=model.prepare('threefry','jax',x,jnp.zeros_like(t),p)
    y1,target1=model.prepare('threefry','jax',x,jnp.ones_like(t),p)
    np.testing.assert_array_equal(y1,x)
    np.testing.assert_allclose(y0+target0,x,atol=2e-7)
    np.testing.assert_array_equal(target0,target1)
    # Positive-strike price is bounded by the discounted underlying mean.
    for sigma in (0.,.2,1.):
        market=pricing.MARKET.astype(np.float64); market[3]=sigma
        for dates in (1,2,128):
            price=pricing.geometric_price(market,dates)
            assert 0<=price<=100
    reject(lambda:pricing.geometric_price([0,100,.03,.2,1]))
    reject(lambda:pricing.geometric_price([100,100,.03,-.2,1]))
    print('PASS: midpoint endpoints, finite quantiles, invalid arguments, interpolation and price bounds.')


if __name__=='__main__':main()
