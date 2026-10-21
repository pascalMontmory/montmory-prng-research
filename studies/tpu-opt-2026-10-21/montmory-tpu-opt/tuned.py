"""Use a confirmed per-size TPU configuration; portable JAX elsewhere."""
from functools import lru_cache
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import kernels as k
import montmory_jax as m


@lru_cache(maxsize=64)
def _function(name,count):
    return jax.jit(k.function(k.candidates()[name],count))


def load_tpu_selection(report_path):
    """Explicitly load a passing native campaign for this JAX/device combination."""
    report=json.loads(Path(report_path).read_text())
    if report["status"]!="pass" or report["cpu_smoke"] or report["backend"]!="tpu":
        raise ValueError("A passing native TPU campaign is required")
    if report["jax"]!=jax.__version__ or report["jaxlib"]!=jax.lib.__version__:
        raise ValueError("JAX version differs from the measured configuration")
    if jax.default_backend()!="tpu" or jax.devices()[0].device_kind!=report["device_kind"]:
        raise ValueError("TPU kind differs from the measured configuration")
    selected={int(n):s["recommended_after_confirmation"] for n,s in report["selection"].items()}
    if any(name not in k.candidates() or k.candidates()[name]["family"]!="montmory" for name in selected.values()):
        raise ValueError("Only Montmory candidates preserve the requested stream")
    return selected


def generate(seed,base,count,*,selection=None):
    """Exact low/high uint32 arrays. No hardware-dependent changes to logical lanes.

    A TPU selection applies to the measured sizes only. Other sizes/backends
    retain portable JAX; no unmeasured optimum is claimed.
    """
    m.validate_range(seed,base,count)
    if count==0: return jnp.empty(0,jnp.uint32),jnp.empty(0,jnp.uint32)
    name="jax_original"
    if selection is not None and jax.default_backend()=="tpu":
        name=selection.get(count,name)
    spec=k.candidates().get(name)
    if spec is None or spec["family"]!="montmory": raise ValueError("Invalid Montmory configuration")
    return _function(name,count)(k.arguments(spec,seed,base))
