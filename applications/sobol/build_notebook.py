#!/usr/bin/env python3
"""Build FR/EN Sobol notebooks with an embedded, hashed source kit."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def cell(kind, text, number):
    out = dict(cell_type=kind, id=f"sobol-{number}", metadata={}, source=text.splitlines(keepends=True))
    if kind == "code":
        compile(text, f"sobol-cell-{number}", "exec")
        out.update(execution_count=None, outputs=[])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    sources = sorted([*here.glob("*.py"), *here.glob("README*.md"), root / "LICENSE",
                      root / "LICENSE.fr.md", root / "THIRD_PARTY_NOTICES.md",
                      root / "THIRD_PARTY_NOTICES.fr.md"])
    sources += sorted(p for p in (here / "validation/cpu").glob("*") if p.is_file())
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path, data in [(p.relative_to(root).as_posix(), p.read_bytes()) for p in sources] + [
                ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())]:
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(data))
    payload = gzip.compress(buffer.getvalue(), mtime=0)
    digest = hashlib.sha256(payload).hexdigest()
    setup = '''import datetime, importlib.metadata, os, subprocess, sys
from pathlib import Path
TPU_READY = False
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
PREFLIGHT_LOG = Path.cwd() / ("sobol-tpu-preflight-" + stamp + ".log")
PREFLIGHT_LOG.write_text("Python: " + sys.version + "\\n")

def logged(command, env=None):
    with PREFLIGHT_LOG.open("a") as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        log.write("Exit: " + str(code) + "\\n")
    if code:
        raise RuntimeError("TPU setup failed. Full log above and in " + str(PREFLIGHT_LOG)
                           + ". Use cell 4 to download diagnostics. No benchmark started.")

required = {"jax": "0.10.2", "jaxlib": "0.10.2", "numpy": "2.4.6", "scipy": "1.17.1"}
try:
    ready = all(importlib.metadata.version(k) == v for k, v in required.items())
    ready = ready and bool(importlib.metadata.version("libtpu"))
except importlib.metadata.PackageNotFoundError:
    ready = False
if not ready:
    logged([sys.executable, "-m", "pip", "install", "--upgrade", "jax[tpu]==0.10.2",
            "numpy==2.4.6", "scipy==1.17.1"])
probe = "import jax; assert jax.default_backend() == 'tpu'; print(jax.__version__, jax.devices())"
logged([sys.executable, "-u", "-c", probe], env={**os.environ, "JAX_PLATFORMS": "tpu"})
TPU_READY = True
'''
    extract = f'''import base64, hashlib, io, json, tarfile, datetime
from pathlib import Path
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
KIT = Path("/content") / ("eninca-sobol-" + stamp)
KIT.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
    for entry in tar.getmembers():
        target = (KIT / entry.name).resolve()
        if not entry.isfile() or not target.is_relative_to(KIT.resolve()):
            raise ValueError("Unexpected archive entry: " + entry.name)
        if target.exists():
            raise ValueError("Duplicate archive entry: " + entry.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(entry) as source:
            target.write_bytes(source.read())
manifest = json.loads((KIT / "manifest.json").read_text())
for rel, expected in manifest.items():
    assert hashlib.sha256((KIT / rel).read_bytes()).hexdigest() == expected, rel
print("Source hashes PASS:", len(manifest))
'''
    run = '''import subprocess, sys, datetime
if not globals().get("TPU_READY", False):
    raise RuntimeError("Cell 1 must complete successfully before the TPU run.")
# True: first native pilot. False: 3 volumes, 3 search + 5 confirmation seeds.
QUICK = True
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RESULTS = KIT / "results" / (stamp + "-sobol-tpu")
command = [sys.executable, "-u", str(KIT / "applications/sobol/run.py"),
           "--backend", "tpu", "--out", str(RESULTS)]
if QUICK:
    command.append("--quick")
print("Conformance, compilation, search, then independent confirmation.", flush=True)
with (KIT / "run.log").open("w") as log:
    with subprocess.Popen(command, cwd=KIT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
print("Results:", RESULTS, flush=True)
if code:
    raise RuntimeError("Native trial failed. Use cell 4 to download diagnostics.")
'''
    download = '''import tarfile
from google.colab import files
preflight = globals().get("PREFLIGHT_LOG")
kit = globals().get("KIT")
results = globals().get("RESULTS")
if not globals().get("TPU_READY", False) and preflight is not None and preflight.exists():
    files.download(str(preflight))
elif results is not None and results.exists():
    archive = results.with_name(results.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(results, arcname=results.name)
        for name in ("run.log", "manifest.json"):
            if kit is not None and (kit / name).exists():
                tar.add(kit / name, arcname=name)
        if preflight is not None and preflight.exists():
            tar.add(preflight, arcname=preflight.name)
    files.download(str(archive))
elif preflight is not None and preflight.exists():
    files.download(str(preflight))
else:
    raise RuntimeError("No diagnostics found. Run cell 1 first.")
'''
    text = {
        "fr": ["# Sobol32 exact sur TPU — prototype ENINCA\n\n128 dimensions, entiers 32 bits, paramètres SciPy exportés. Validation CPU et interprétation Pallas réalisées ; compilation et performances natives TPU restent à vérifier. Aucun gain TPU ni applicatif revendiqué. Le noyau JAX de comparaison utilise aussi l’optimisation par blocs. Sélectionner **Exécution > Modifier le type d’environnement > TPU**, puis exécuter les quatre cellules. Ce notebook utilise les ressources du compte de son exécutant. Il ne réserve pas de TPU et ne transmet pas de résultats automatiquement.",
               "## 1. Vérifier l’accès TPU\nInstalle si nécessaire les versions figées JAX, NumPy et SciPy. Affiche et conserve les messages d’erreur complets. En cas de quota Colab ou d’absence de TPU, ne pas forcer un repli CPU : télécharger les diagnostics à l’étape 4 et reprendre lorsque l’accès est rétabli.",
               "## 2. Extraire les sources\nLe kit est intégré et son empreinte vérifiée. Aucun accès GitHub n’est nécessaire. Le dossier contient les documentations française et anglaise.",
               "## 3. Vérifier et mesurer\nLe mode pilote `QUICK=True` utilise deux volumes. La conformité et toutes les compilations précèdent les mesures ; elles peuvent prendre plusieurs minutes. Après un pilote réussi, `QUICK=False` lance une campagne plus complète avec cinq nouveaux cas de confirmation. Chaque sortie chronométrée est comparée exactement à sa référence. Les préparations et transferts initiaux sont enregistrés séparément du débit à paramètres réutilisés.",
               "## 4. Télécharger résultats ou diagnostics\nÀ exécuter aussi après un échec de la première ou troisième cellule. Les résultats incluent les paramètres, temps bruts, choix de configuration, versions et empreintes. Un résultat CPU ou une interprétation ne constitue jamais une mesure TPU."],
        "en": ["# Exact Sobol32 on TPU — ENINCA prototype\n\n128 dimensions, 32-bit words, exported SciPy parameters. CPU and interpreted Pallas validation completed; native TPU compilation and performance remain unverified. No TPU or application speedup claimed. The JAX comparator also includes block optimization. Select **Runtime > Change runtime type > TPU**, then execute the four cells. This notebook uses its runner’s account resources. It does not reserve a TPU or automatically transmit results.",
               "## 1. Check TPU access\nInstalls pinned JAX, NumPy and SciPy if needed. Displays and saves complete error messages. On a Colab quota or missing TPU, do not force CPU fallback: download diagnostics in step 4 and resume when access is restored.",
               "## 2. Extract sources\nThe kit is embedded and its hash is checked. No GitHub access is needed. It includes French and English documentation.",
               "## 3. Verify and measure\nThe `QUICK=True` pilot uses two volumes. Conformance and all compilations precede timing and may take several minutes. After a successful pilot, `QUICK=False` runs the fuller campaign with five new confirmation cases. Every timed output is checked exactly against its reference. Setup and initial uploads are recorded separately from reused-parameter throughput.",
               "## 4. Download results or diagnostics\nAlso run after failure of the first or third cell. Results include parameters, raw times, configuration selections, versions and hashes. CPU or interpreter output is never a native TPU measurement."]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eninca-sobol-tpu-kit.tar.gz").write_bytes(payload)
    for language, prose in text.items():
        cells = [cell("markdown", prose[0], 0)]
        for description, code in zip(prose[1:], (setup, extract, run, download), strict=True):
            cells.append(cell("markdown", description, len(cells)))
            cells.append(cell("code", code, len(cells)))
        notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
        suffix = ".en" if language == "en" else ""
        (args.out / f"Eninca_Sobol_TPU{suffix}.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(dict(directory=str(args.out), kit_sha256=digest, files=len(manifest))))


if __name__ == "__main__":
    main()
