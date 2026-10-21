"""Export TPU-targeted IR on CPU; this is not native TPU machine compilation."""
import argparse
import functools
import hashlib
import json
from pathlib import Path

import jax
from jax import export
import montmory_jax as m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"stage": "TPU-targeted StableHLO/Mosaic export only", "jax": jax.__version__,
              "native_tpu_compilation": False, "native_tpu_execution": False, "cases": []}
    cases = []
    for method in ("portable", "mulhi"):
        cases.append((f"jax_prepared_{method}", functools.partial(m.prepared_range, count=8193, method=method), True))
    for rows in (8, 16, 32):
        cases.append((f"pallas_portable_{rows}", functools.partial(m._pallas_range, count=8193,
                      block_rows=rows, method="portable", interpret=False), True))
    cases.append(("pallas_mulhi_unsupported_probe", functools.partial(m._pallas_range, count=8193,
                  block_rows=8, method="mulhi", interpret=False), False))
    signature = jax.ShapeDtypeStruct((13,), m.U32)
    for name, function, required in cases:
        row = {"name": name, "required": required}
        try:
            exported = export.export(jax.jit(function), platforms=("tpu",))(signature)
            ir = exported.mlir_module()
            (args.out / (name + ".mlir")).write_text(ir)
            row.update(status="export_pass", sha256=hashlib.sha256(ir.encode()).hexdigest(), characters=len(ir))
        except Exception as error:
            row.update(status="export_failed", error_type=type(error).__name__, error=str(error))
        report["cases"].append(row)
        print(name, row["status"], row.get("error", ""), flush=True)
    report["required_exports_passed"] = all(r["status"] == "export_pass" for r in report["cases"] if r["required"])
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["required_exports_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
