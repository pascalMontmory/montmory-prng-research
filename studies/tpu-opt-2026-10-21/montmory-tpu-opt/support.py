import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def unpack(words):
    lo, hi = (np.asarray(jax.device_get(w), dtype=np.uint64) for w in words)
    return lo | (hi << np.uint64(32))


def assert_equal(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shapes differ")
    mismatch = np.flatnonzero(actual != expected)
    if mismatch.size:
        i = int(mismatch[0])
        raise AssertionError(f"{label}: first mismatch at {i}: {actual.flat[i]} != {expected.flat[i]}")


class Oracle:
    def __init__(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / ("oracle.dylib" if sys.platform == "darwin" else "oracle.so")
        compiler = shlex.split(os.environ.get("CC", "cc"))
        env = dict(os.environ)
        if sys.platform == "darwin":
            if "CC" not in os.environ:
                compiler = [subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()]
            env["SDKROOT"] = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        command = compiler + ["-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC",
                              "-dynamiclib" if sys.platform == "darwin" else "-shared",
                              str(ROOT / "reference/oracle.c"), "-o", str(target)]
        subprocess.run(command, env=env, check=True)
        self.lib = ctypes.CDLL(str(target.resolve()))
        array = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C_CONTIGUOUS")
        self.lib.reference_words.argtypes = [ctypes.c_uint64, array, array, ctypes.c_size_t]
        self.lib.reference_words.restype = None

    def indices(self, seed, indices):
        ix = np.ascontiguousarray(indices, dtype=np.uint64)
        output = np.empty(ix.size, dtype=np.uint64)
        self.lib.reference_words(seed, ix, output, ix.size)
        return output

    def range(self, seed, base, count):
        return self.indices(seed, np.arange(count, dtype=np.uint64) + np.uint64(base))
