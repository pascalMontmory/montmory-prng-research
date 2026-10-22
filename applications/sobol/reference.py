"""Sobol32 input contract and host oracles; no accelerator dependency.

Developed by Pascal Montmory for ENINCA CONSULTING. See ../../LICENSE.
Direction numbers and LMS+shift parameters are supplied by installed SciPy.
"""
from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
from scipy.stats import qmc

DIMENSIONS = 128
BITS = 32
LIMIT = 1 << BITS


def check_range(start, count):
    if type(start) is not int or type(count) is not int:
        raise ValueError("start/count must be Python integers")
    if start < 0 or count < 1 or start + count > LIMIT:
        raise ValueError("require 0 <= start < start+count <= 2**32")
    if count * DIMENSIONS >= 1 << 31:
        raise ValueError("prototype limits output to fewer than 2**31 elements")


@dataclass
class Parameters:
    directions: np.ndarray  # [bit, dimension], uint32, bit 0 first
    shift: np.ndarray       # [dimension], uint32
    seed: int | None

    def digest(self):
        return hashlib.sha256(self.directions.astype("<u4").tobytes()
                              + self.shift.astype("<u4").tobytes()).hexdigest()

    def save(self, path):
        np.savez(path, directions=self.directions, shift=self.shift)


def scipy_engine(seed):
    if seed is not None and (type(seed) is not int or not 0 <= seed < 1 << 64):
        raise ValueError("seed must be None or an unsigned 64-bit Python integer")
    return qmc.Sobol(DIMENSIONS, bits=BITS, scramble=seed is not None,
                     rng=np.random.default_rng(0 if seed is None else seed))


def parameters(seed):
    engine = scipy_engine(seed)
    # Private data extraction is intentional and version-pinned by run.py.
    # Public SciPy output is independently checked in verify.py.
    directions = np.ascontiguousarray(engine._sv.T, dtype=np.uint32)
    shift = np.ascontiguousarray(engine._shift, dtype=np.uint32)
    if directions.shape != (BITS, DIMENSIONS) or shift.shape != (DIMENSIONS,):
        raise RuntimeError("unexpected SciPy Sobol layout")
    return Parameters(directions, shift, seed)


def scipy_words(seed, count, start=0):
    check_range(start, count)
    engine = scipy_engine(seed)
    if start:
        engine.fast_forward(start)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        points = engine.random(count)
    # SciPy's 32-bit dyadic coordinates are represented exactly in float64.
    return np.ldexp(points, BITS).astype(np.uint32)


def words(p, start, count):
    """Host random-access oracle, Gray code formula, entirely uint32."""
    check_range(start, count)
    index = np.arange(count, dtype=np.uint32) + np.uint32(start)
    gray = index ^ (index >> np.uint32(1))
    out = np.broadcast_to(p.shift, (count, DIMENSIONS)).copy()
    for bit in range(BITS):
        out ^= np.where(((gray >> np.uint32(bit)) & np.uint32(1))[:, None],
                        p.directions[bit], np.uint32(0))
    return out


def recurrence_words(p, start, count):
    """Second host route: one scalar Gray evaluation, then Sobol recurrence."""
    check_range(start, count)
    state = p.shift.copy()
    gray = start ^ (start >> 1)
    for bit in range(BITS):
        if gray & (1 << bit):
            state ^= p.directions[bit]
    out = np.empty((count, DIMENSIONS), dtype=np.uint32)
    out[0] = state
    for row in range(1, count):
        index = start + row
        changed_bit = (index & -index).bit_length() - 1
        state ^= p.directions[changed_bit]
        out[row] = state
    return out


def local_table(p, tile):
    if tile not in (128, 256, 512):
        raise ValueError("unsupported tile")
    # No digital shift: it is applied exactly once in the common contribution.
    return words(p, 0, tile) ^ p.shift
