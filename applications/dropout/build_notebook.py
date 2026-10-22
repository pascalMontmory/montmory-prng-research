#!/usr/bin/env python3
"""Package only the experiment and its frozen dependencies for Colab."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    kit = root / "studies/tpu-opt-2026-10-21/montmory-tpu-opt"
    sources = [here / n for n in ("dropout.py", "model.py", "run.py", "verify.py", "README.md", "README.fr.md")]
    sources += [kit / n for n in ("kernels.py", "montmory_jax.py", "support.py",
                                  "reference/oracle.c", "reference/mmc_core.c", "reference/mmc_core.h")]
    sources += [root / "LICENSE", root / "THIRD_PARTY_NOTICES.md", root / "THIRD_PARTY_NOTICES.fr.md"]
    sources += [root / "third_party/Random123-LICENSE"]
    sources += [p for p in (here / "validation/cpu").glob("*") if p.suffix in (".json", ".csv")]
    # Include third-party license texts referenced by the frozen kit.
    sources += list((kit / "licenses").rglob("*")) if (kit / "licenses").exists() else []
    sources = sorted({p for p in sources if p.is_file()})
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for rel, content in [(str(p.relative_to(root)), p.read_bytes()) for p in sources] + [
            ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())]:
            info = tarfile.TarInfo(rel)
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
    payload = gzip.compress(buffer.getvalue(), mtime=0)
    digest = hashlib.sha256(payload).hexdigest()
    setup = '''import datetime, importlib.metadata, os, subprocess, sys
from pathlib import Path
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
TPU_PREFLIGHT_LOG = Path.cwd() / ("montmory-tpu-preflight-" + stamp + ".log")
TPU_PREFLIGHT_OK = False
TPU_PREFLIGHT_LOG.write_text("Python: " + sys.version + "\\n")
print("Diagnostic log:", TPU_PREFLIGHT_LOG, flush=True)

def run_logged(command, env=None):
    with TPU_PREFLIGHT_LOG.open("a") as log:
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
        log.write("Exit status: " + str(code) + "\\n")
    if code:
        raise RuntimeError("Setup / TPU access check failed (exit " + str(code)
                           + "). See the output above. Diagnostic log: " + str(TPU_PREFLIGHT_LOG)
                           + ". Use cell 4 to download it. No benchmark was started.")

try:
    ready = (importlib.metadata.version("jax") == "0.10.2"
             and importlib.metadata.version("jaxlib") == "0.10.2"
             and bool(importlib.metadata.version("libtpu")))
except importlib.metadata.PackageNotFoundError:
    ready = False
if not ready:
    run_logged([sys.executable, "-m", "pip", "install", "--upgrade", "jax[tpu]==0.10.2"])
probe = "import jax; assert jax.default_backend() == 'tpu'; print(jax.__version__, jax.devices())"
run_logged([sys.executable, "-u", "-c", probe], env={**os.environ, "JAX_PLATFORMS": "tpu"})
TPU_PREFLIGHT_OK = True
'''
    extract = f'''import base64, hashlib, io, json, tarfile
from pathlib import Path
import datetime
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
KIT = Path("/content") / ("montmory-dropout-" + stamp)
KIT.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
    for member in tar.getmembers():
        target = (KIT / member.name).resolve()
        if not member.isfile() or not target.is_relative_to(KIT.resolve()):
            raise ValueError("Unexpected archive member: " + member.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source:
            target.write_bytes(source.read())
manifest = json.loads((KIT / "manifest.json").read_text())
for rel, expected in manifest.items():
    assert hashlib.sha256((KIT / rel).read_bytes()).hexdigest() == expected, rel
print("Source hashes PASS:", len(manifest), "files")
'''
    run = '''import subprocess, sys
if not globals().get("TPU_PREFLIGHT_OK", False):
    raise RuntimeError("Run cell 1 successfully before starting the TPU benchmark.")
RESULTS = KIT / "results" / "dropout-tpu"
RESULT_ARCHIVE = RESULTS.with_suffix(".tar.gz")
command = [sys.executable, "-u", str(KIT / "applications/dropout/run.py"),
           "--backend", "tpu", "--out", str(RESULTS), "--steps", "150"]
print("Checks, compilation, then 150 updates per seed and implementation.", flush=True)
with (KIT / "run.log").open("w") as log:
    with subprocess.Popen(command, cwd=KIT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
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
print("Archive:", RESULT_ARCHIVE)
if code:
    raise RuntimeError("Run failed. Use the next cell to download diagnostics.")
'''
    download = '''import tarfile
from google.colab import files
# Cell 1 can fail before KIT / RESULTS exist.
preflight = globals().get("TPU_PREFLIGHT_LOG")
results = globals().get("RESULTS")
kit = globals().get("KIT")
if not globals().get("TPU_PREFLIGHT_OK", False) and preflight is not None and preflight.exists():
    files.download(str(preflight))
elif results is not None and results.exists():
    archive = results.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(results, arcname=results.name, filter=lambda i: None if i.name.endswith((".so", ".dylib")) else i)
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
    texts = {
        "fr": ["# Petit réseau, dropout Montmory et Threefry sur TPU\n\nPrototype local validé sur CPU ; compilation et performances natives TPU à vérifier. Deux couches cachées de 128 neurones, 21 252 paramètres, données synthétiques, dropout à 25 %. Trois graines, 150 étapes par variante. Choisir **Exécution > Modifier le type d’environnement > TPU**, puis exécuter les cellules dans l’ordre. Aucun accès à votre compte GitHub ni envoi automatique de résultats.\n\nLe contrat exact porte sur les entiers et les masques ; l’entraînement flottant n’est pas garanti bit à bit entre architectures.",
               "## 1. Vérifier le TPU\nCette cellule installe JAX 0.10.2 et libtpu si nécessaire, puis refuse tout repli CPU. Elle affiche et conserve la sortie complète du sous-processus. En cas d’échec, passer directement à la cellule 4 pour télécharger le diagnostic. Si aucun TPU n’est alloué, reconnecter un environnement TPU avant de continuer ; un quota Colab exige d’attendre le rétablissement de l’accès ou de revoir l’offre de calcul.",
               "## 2. Extraire les sources contrôlées\nLe kit est intégré à ce notebook ; aucune publication GitHub n’est nécessaire.",
               "## 3. Contrôler et entraîner\nLes compilations précèdent les mesures. Prévoir plusieurs minutes, selon l’allocation et la compilation. Les quatre chemins sont JAX Threefry, JAX Montmory, Pallas Threefry et Pallas Montmory. Les masques Pallas doivent correspondre à leur propre référence. Un échec de compilation TPU reste possible à ce stade du prototype.",
               "## 4. Télécharger résultats ou diagnostics\nExécuter aussi après une erreur à l’étape 1 ou 3. Si la vérification initiale échoue, cette cellule télécharge son journal même sans kit extrait. Aucun chiffre de vitesse CPU n’est présenté comme une mesure TPU."],
        "en": ["# Small-network training with Montmory and Threefry dropout on TPU\n\nPrototype validated locally on CPU; native TPU compilation and performance remain to be checked. Two hidden layers of 128 neurons, 21,252 parameters, synthetic data, 25% dropout. Three seeds, 150 updates per variant. Select **Runtime > Change runtime type > TPU**, then run the cells in order. No GitHub account access or automatic results transmission.\n\nThe exact contract covers integers and masks; floating-point training is not guaranteed bitwise identical across architectures.",
               "## 1. Check the TPU\nInstalls JAX 0.10.2 and libtpu if needed, then refuses CPU fallback. Displays and saves the complete subprocess output. On failure, go straight to cell 4 to download diagnostics. If no TPU is allocated, reconnect to a TPU runtime before continuing; a Colab quota requires waiting for access to resume or reviewing the compute plan.",
               "## 2. Extract checked sources\nThe kit is embedded in this notebook; no GitHub publication is required.",
               "## 3. Verify and train\nCompilations precede measurements. Allow several minutes, depending on allocation and compilation. The four paths are JAX Threefry, JAX Montmory, Pallas Threefry and Pallas Montmory. Pallas masks must match their own reference. Native TPU compilation may still fail at this prototype stage.",
               "## 4. Download results or diagnostics\nAlso run this after an error in step 1 or 3. If the initial check fails, this cell downloads its log even without an extracted kit. No CPU speed is presented as a TPU measurement."]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "montmory-dropout-kit.tar.gz").write_bytes(payload)
    for lang, prose in texts.items():
        cells = [cell("markdown", prose[0])]
        for text, code in zip(prose[1:], (setup, extract, run, download), strict=True):
            compile(code, f"notebook-{lang}", "exec")
            cells += [cell("markdown", text), cell("code", code)]
        for i, notebook_cell in enumerate(cells):
            notebook_cell["id"] = f"dropout-cell-{i}"
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
        name = "Montmory_Dropout_TPU" + (".en" if lang == "en" else "") + ".ipynb"
        (args.out / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"directory": str(args.out), "kit_sha256": digest, "source_files": len(manifest)}))


if __name__ == "__main__":
    main()
