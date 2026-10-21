"""Exact integer model of the recovered Metal 3R engine, not a new PRNG.

Python integers explicitly reduced modulo 2**64. No statistical assumptions.
"""

MOD = 1 << 64
MASK = MOD - 1
MASK32 = (1 << 32) - 1
GAMMA = 0x9E3779B97F4A7C15
A = 0xA24BAED496EE9B5F
B = 0x9FB21C651E98DF25
A_INV = pow(A, -1, MOD)
B_INV = pow(B, -1, MOD)


def rol(x, r):
    r &= 63
    return ((x << r) | (x >> ((64 - r) & 63))) & MASK


def undo_xor_right(y, shift):
    x = y
    for k in range(shift, 64, shift):
        x ^= y >> k
    return x


def splitmix(x):
    x = (x + GAMMA) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def params(seed):
    k0 = splitmix(seed)
    k1 = splitmix(k0)
    k2 = splitmix(k1)
    r = splitmix(k2)
    return (k0, k1, k2), tuple(((r >> n) & 63) | 1 for n in (0, 8, 16))


def round_f(x, key, r0, r1):
    if x & 1:
        y = rol((3 * x + key) & MASK, r0) ^ (x >> 1)
    else:
        y = rol((x + key) & MASK, r1) ^ ((5 * x) & MASK)
    return y & MASK32


def feistel(x, seed, inverse=False):
    keys, rotations = params(seed)
    left, right = x >> 32, x & MASK32
    for i in (range(2, -1, -1) if inverse else range(3)):
        r0, r1 = rotations[i], rotations[(i + 1) % 3]
        if inverse:
            left, right = right ^ round_f(left, keys[i], r0, r1), left
        else:
            left, right = right, left ^ round_f(right, keys[i], r0, r1)
    return (left << 32) | right


def mixer(x):
    w = x ^ (x >> 32)
    w ^= w >> 25
    w = (w * A) & MASK
    w ^= w >> 28
    w = (w * B) & MASK
    w ^= w >> 27
    return rol(w, -(x >> 59))


def inverse_unrotated_mixer(y):
    y = undo_xor_right(y, 27)
    y = (y * B_INV) & MASK
    y = undo_xor_right(y, 28)
    y = (y * A_INV) & MASK
    y = undo_xor_right(y, 25)
    return undo_xor_right(y, 32)


def preimages(y):
    """Complete inverse fiber: the rotation selector has exactly 32 values."""
    result = []
    for rotation in range(32):
        x = inverse_unrotated_mixer(rol(y, rotation))
        if x >> 59 == rotation:
            result.append(x)
    return result


def state(seed, lane, n):
    gamma = (GAMMA + 2 * lane) & MASK
    return (seed + n * gamma) & MASK


def at(seed, lane, n):
    """n=1 denotes the first draw (advance, then output)."""
    return mixer(feistel(state(seed, lane, n), seed))


def metal_mantissa(seed, threads, draws, position):
    """Absolute position, zero based, including repeated GPU refills."""
    if threads <= 0 or draws <= 0 or position < 0:
        raise ValueError('positive dimensions and nonnegative position required')
    lane = (position // draws) % threads
    return at(seed, lane, position + 1) >> 11
