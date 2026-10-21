# mmc_numba.py
import numpy as np
from numba import njit, prange

@njit(inline='always')
def rotl64(x, r):
    r &= 63
    return ((x << r) | (x >> ((64 - r) & 63))) & ((1<<64)-1)

@njit(inline='always')
def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & ((1<<64)-1)
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & ((1<<64)-1)
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & ((1<<64)-1)
    return (x ^ (x >> 31)) & ((1<<64)-1)

@njit(inline='always')
def pcg_dxsm(x):
    w = ((x >> 32) ^ x) & ((1<<64)-1)
    rot = x >> 59
    w ^= (w >> 25)
    w = (w * 0xA24BAED496EE9B5F) & ((1<<64)-1)
    w ^= (w >> 28)
    w = (w * 0x9FB21C651E98DF25) & ((1<<64)-1)
    w ^= (w >> 27)
    return ((w >> rot) | (w << ((-rot) & 63))) & ((1<<64)-1)

@njit(inline='always')
def f_collatz(x, k, r0, r1):
    if x & 1:
        y = (rotl64((x * 3 + k) & ((1<<64)-1), r0) ^ (x >> 1)) & ((1<<64)-1)
    else:
        y = (rotl64((x + k) & ((1<<64)-1), r1) ^ ((x * 5) & ((1<<64)-1))) & ((1<<64)-1)
    return y & 0xFFFFFFFF

@njit(inline='always')
def permute3(x, k0,k1,k2, r0,r1,r2):
    L = (x >> 32) & 0xFFFFFFFF
    R = x & 0xFFFFFFFF
    F = f_collatz(R, k0, r0, r1); L ^= F; L, R = R, L
    F = f_collatz(R, k1, r1, r2); L ^= F; L, R = R, L
    F = f_collatz(R, k2, r2, r0); L ^= F; L, R = R, L
    return ((L & 0xFFFFFFFF) << 32) | (R & 0xFFFFFFFF)

@njit
def init_state(seed, lane):
    s = seed & ((1<<64)-1)
    gamma = (0x9E3779B97F4A7C15 + (lane<<1)) | 1  # impair par lane
    gamma &= ((1<<64)-1)
    k0 = splitmix64(seed)
    k1 = splitmix64(k0)
    k2 = splitmix64(k1)
    rseed = splitmix64(k2)
    r0 = (rseed      & 63) | 1
    r1 = ((rseed>>8) & 63) | 1
    r2 = ((rseed>>16)& 63) | 1
    return s, gamma, k0,k1,k2, r0,r1,r2

@njit(parallel=True)
def generate_parallel(n_total, seed=42, lanes=8):
    out = np.empty(n_total, dtype=np.float64)
    chunk = n_total // lanes
    for lane in prange(lanes):
        start = lane*chunk
        end   = (lane+1)*chunk if lane < lanes-1 else n_total
        s, g, k0,k1,k2, r0,r1,r2 = init_state(seed, lane)
        for i in range(start, end):
            s = (s + g) & ((1<<64)-1)
            y = permute3(s, k0,k1,k2, r0,r1,r2)
            y = pcg_dxsm(y)
            mant = (y >> 11) & ((1<<53)-1)
            out[i] = mant * (1.0/(1<<53))
    return out

if __name__ == "__main__":
    x = generate_parallel(10_000_000, seed=42, lanes=8)
    print("mean", x.mean(), "var", x.var())
