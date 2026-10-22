"""Discrete Asian calls under risk-neutral GBM, 128 monitoring dates.

No time-zero observation, dividends, control variate or Brownian bridge.
Geometric-average log price is Gaussian: its exact moments supply an oracle.
Original implementation for ENINCA CONSULTING, see LICENSE.
"""
import math
import numpy as np
from scipy.special import ndtr

DATES=128
MARKET=np.array([100.,100.,.03,.2,1.],dtype=np.float32)  # S0,K,r,sigma,T


def geometric_price(market, dates=DATES):
    s,k,r,sigma,t=map(float,market)
    if s<=0 or k<=0 or sigma<0 or t<=0 or dates<1:
        raise ValueError('invalid market or monitoring dates')
    # Cov(W_ti,W_tj)=min(ti,tj); E[t_i]=T(n+1)/(2n).
    mean=math.log(s)+(r-.5*sigma*sigma)*t*(dates+1)/(2*dates)
    var=sigma*sigma*t*(dates+1)*(2*dates+1)/(6*dates*dates)
    if var==0:
        return math.exp(-r*t)*max(math.exp(mean)-k,0.)
    d2=(mean-math.log(k))/math.sqrt(var)
    return math.exp(-r*t)*(math.exp(mean+var/2)*ndtr(d2+math.sqrt(var))-k*ndtr(d2))


def numpy_payoffs(z,market):
    s,k,r,sigma,t=map(float,market)
    z=np.asarray(z,dtype=np.float64)
    dt=t/z.shape[1]
    logs=math.log(s)+np.cumsum((r-.5*sigma*sigma)*dt+sigma*math.sqrt(dt)*z,axis=1)
    arithmetic=np.maximum(np.exp(logs).mean(axis=1)-k,0)*math.exp(-r*t)
    geometric=np.maximum(np.exp(logs.mean(axis=1))-k,0)*math.exp(-r*t)
    return np.stack((arithmetic,geometric),axis=1)


def payoffs(z,market):
    import jax.numpy as jnp
    s,k,r,sigma,t=market
    dt=t/z.shape[1]
    logs=jnp.log(s)+jnp.cumsum((r-.5*sigma*sigma)*dt+sigma*jnp.sqrt(dt)*z,axis=1)
    a=jnp.maximum(jnp.mean(jnp.exp(logs),axis=1)-k,0)*jnp.exp(-r*t)
    g=jnp.maximum(jnp.exp(jnp.mean(logs,axis=1))-k,0)*jnp.exp(-r*t)
    return jnp.stack((a,g),axis=1)
