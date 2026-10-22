#!/usr/bin/env python3
"""Independently audit embedded application sources and retained CPU summaries."""
import ast
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def close(left, right):
    assert math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12), (left, right)


def main():
    checked_files = notebooks = 0
    for app in ("diffusion", "asian_option"):
        folder = ROOT / "applications" / app
        for doc in folder.glob("*.md"):
            for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', doc.read_text()):
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                    continue
                assert (doc.parent / unquote(target.split('#', 1)[0])).exists(), (doc, target)
        for file in (folder / "notebooks").glob("*.ipynb"):
            nb = json.loads(file.read_text())
            payloads = []
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                assert cell["execution_count"] is None and not cell["outputs"]
                source = "".join(cell["source"])
                compile(source, str(file), "exec")
                for node in ast.walk(ast.parse(source)):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "b64decode"):
                        payloads.append(base64.b64decode(ast.literal_eval(node.args[0]), validate=True))
            assert len(payloads) == 1, file
            with tarfile.open(fileobj=io.BytesIO(payloads[0]), mode="r:gz") as tar:
                names = tar.getnames()
                assert len(names) == len(set(names))
                assert all(m.isfile() for m in tar.getmembers())
                manifest = json.load(tar.extractfile("manifest.json"))
                assert set(names) == set(manifest) | {"manifest.json"}
                for name, digest in manifest.items():
                    path = (ROOT / name).resolve()
                    assert path.is_relative_to(ROOT)
                    embedded = tar.extractfile(name).read()
                    assert hashlib.sha256(embedded).hexdigest() == digest
                    assert embedded == path.read_bytes(), (file, name)
                    checked_files += 1
            notebooks += 1
        report = json.loads((folder / "validation/cpu/report.json").read_text())
        assert report["status"] == "pass" and report["protocol"]["backend"] == "cpu"
        assert not report["native_tpu_measurements"]
        assert report["verification"]["pallas_interpreted"]
        assert not report["statistical_batteries_run"]
        for name, digest in report["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    folder = ROOT / "applications/diffusion/validation/cpu"
    report = json.loads((folder / "report.json").read_text())
    steps = csv_rows(folder / "steps.csv")
    curves = csv_rows(folder / "learning.csv")
    assert len(steps) == report["timed_steps"] == 1800
    for run in report["training"]:
        selected = [r for r in steps if r["variant"] == run["variant"] and int(r["seed"]) == run["seed"]]
        assert len(selected) == len({r["step"] for r in selected}) == report["configuration"]["steps"]
        close(statistics.median(int(r["elapsed_ns"]) for r in selected)/1e6, run["median_step_ms"])
        selected = [r for r in curves if r["variant"] == run["variant"] and int(r["seed"]) == run["seed"]]
        close(selected[0]["heldout_mse"], run["initial_heldout_mse"])
        close(selected[-1]["heldout_mse"], run["final_heldout_mse"])

    folder = ROOT / "applications/asian_option/validation/cpu"
    report = json.loads((folder / "report.json").read_text())
    rows = csv_rows(folder / "measurements.csv")
    estimates = csv_rows(folder / "estimates.csv")
    assert len(rows) == report["timed_outputs_checked"] == 96
    for estimate in estimates:
        selected = [r for r in rows if all(r[k] == estimate[k] for k in ("paths", "seed", "variant"))]
        assert len(selected) == len({r["repeat"] for r in selected}) == report["configuration"]["repetitions"]
        close(statistics.median(int(r["elapsed_ns"]) for r in selected)/1e9, estimate["median_seconds"])
        for r in selected:
            for key in ("arithmetic", "geometric"):
                close(r[key], estimate[key])
    for summary in report["summary"]:
        selected = [r for r in estimates if r["variant"] == summary["variant"] and int(r["paths"]) == summary["paths"]]
        assert len(selected) == len({r["seed"] for r in selected}) == summary["replicates"]
        for key, ref, result in (("geometric", "geometric_analytic", "geometric_rmse"),
                                 ("arithmetic", "arithmetic", "arithmetic_rms_reference_difference")):
            value = math.sqrt(statistics.mean((float(r[key])-report["reference"][ref])**2 for r in selected))
            close(value, summary[result])
    assert notebooks == 4
    print(f"PASS: {notebooks} notebooks, {checked_files} embedded file comparisons, source hashes and CPU summaries.")


if __name__ == "__main__":
    main()
