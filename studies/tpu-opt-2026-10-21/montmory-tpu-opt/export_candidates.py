"""TPU-targeted IR export on CPU; not native TPU compilation or timing."""
import json
from pathlib import Path
import sys
import jax
from jax import export
import kernels as k

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
records=[]
for name,spec in k.candidates().items():
    if spec["kind"] in ("write","vendor"):
        continue
    arg=k.arguments(spec,42)
    record=dict(engine=name)
    try:
        result=export.export(jax.jit(k.function(spec,8193)),platforms=("tpu",))(jax.ShapeDtypeStruct(arg.shape,arg.dtype))
        (out/(name+".mlir")).write_text(result.mlir_module())
        record["status"]="export_pass"
    except Exception as error:
        record.update(status="fail",error=repr(error))
    print(name,record["status"],flush=True)
    records.append(record)
(out/"report.json").write_text(json.dumps(dict(native_tpu_execution=False,cases=records),indent=2)+"\n")
if any(r["status"]!="export_pass" for r in records):
    raise SystemExit(1)
