#!/usr/bin/env python3
"""Build self-contained bilingual TPU notebooks for both application prototypes."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def cell(kind,source,index):
    value=dict(cell_type=kind,id=f'application-cell-{index}',metadata={},source=source.splitlines(keepends=True))
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    paths=[]
    for app in ('stochastic','diffusion','asian_option','sobol'):
        folder=ROOT/'applications'/app
        paths += list(folder.glob('*.py'))+list(folder.glob('*.md'))+list(folder.glob('*.txt'))
    kit=ROOT/'studies/tpu-opt-2026-10-21/montmory-tpu-opt'
    paths += [kit/n for n in ('kernels.py','montmory_jax.py','support.py','reference/oracle.c','reference/mmc_core.c','reference/mmc_core.h')]
    paths += [ROOT/n for n in ('LICENSE','THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.fr.md','third_party/Random123-LICENSE')]
    paths=sorted(set(paths)); manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as tar:
        for name,data in [(str(p.relative_to(ROOT)),p.read_bytes()) for p in paths]+[('manifest.json',json.dumps(manifest,indent=2).encode())]:
            item=tarfile.TarInfo(name); item.size=len(data); item.mode=0o644; tar.addfile(item,io.BytesIO(data))
    payload=gzip.compress(buffer.getvalue(),mtime=0); digest=hashlib.sha256(payload).hexdigest()
    setup='''import datetime, importlib.metadata, os, subprocess, sys
from pathlib import Path
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
PREFLIGHT_LOG = Path.cwd() / ("applications-tpu-preflight-" + stamp + ".log")
TPU_READY = False
def logged(command, env=None):
    with PREFLIGHT_LOG.open("a") as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True); log.write(line); log.flush()
                code = process.wait()
            except BaseException:
                process.terminate(); process.wait(); raise
    if code:
        raise RuntimeError("Setup / TPU access failed. Download diagnostics in cell 4. No CPU fallback.")
pins = {"jax":"0.10.2", "jaxlib":"0.10.2", "numpy":"2.4.6", "scipy":"1.17.1", "scikit-learn":"1.7.2", "matplotlib":"3.10.7"}
try:
    ready = all(importlib.metadata.version(k) == v for k,v in pins.items()) and bool(importlib.metadata.version("libtpu"))
except importlib.metadata.PackageNotFoundError:
    ready = False
if not ready:
    logged([sys.executable,"-m","pip","install","--upgrade","jax[tpu]==0.10.2"] + [k+"=="+v for k,v in pins.items() if k!="jax"])
logged([sys.executable,"-u","-c","import jax; assert jax.default_backend() == 'tpu'; print(jax.__version__, jax.devices())"], env={**os.environ,"JAX_PLATFORMS":"tpu"})
TPU_READY = True
'''
    extract=f'''import base64, hashlib, io, json, tarfile
from pathlib import Path
import datetime
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
KIT = Path("/content") / ("eninca-applications-" + stamp)
KIT.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(payload),mode="r:gz") as tar:
    seen = set()
    for member in tar.getmembers():
        target = (KIT / member.name).resolve()
        if not member.isfile() or not target.is_relative_to(KIT.resolve()) or member.name in seen:
            raise ValueError("Invalid archive member")
        seen.add(member.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(tar.extractfile(member).read())
manifest = json.loads((KIT / "manifest.json").read_text())
assert seen == set(manifest) | {{"manifest.json"}}
for rel,digest in manifest.items():
    assert hashlib.sha256((KIT / rel).read_bytes()).hexdigest() == digest, rel
print("Source hashes PASS:", len(manifest))
'''
    download='''import tarfile
from google.colab import files
log = globals().get("PREFLIGHT_LOG")
results = globals().get("RESULTS")
if not globals().get("TPU_READY",False) and log is not None and log.exists():
    files.download(str(log))
elif results is not None and results.exists():
    archive = results.with_suffix(".tar.gz")
    with tarfile.open(archive,"w:gz") as tar:
        tar.add(results,arcname=results.name,filter=lambda i: None if i.name.endswith((".so",".dylib")) else i)
        for path in (KIT / "run.log", KIT / "manifest.json", log):
            if path is not None and path.exists(): tar.add(path,arcname=path.name)
    files.download(str(archive))
elif log is not None and log.exists():
    files.download(str(log))
else:
    raise RuntimeError("No diagnostics yet; start with cell 1.")
'''
    titles={'diffusion':('Petit modèle génératif de chiffres — diffusion / flow matching','Small digit generator — diffusion / flow matching','Eninca_Diffusion_TPU'),
            'asian_option':('Options asiatiques — Monte-Carlo et Sobol randomisé','Asian options — Monte Carlo and randomized Sobol','Eninca_Asian_Option_TPU')}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'eninca-applications-kit.tar.gz').write_bytes(payload)
    for app,(fr,en,filename) in titles.items():
        run=f'''import subprocess, sys
if not globals().get("TPU_READY",False):
    raise RuntimeError("Cell 1 must succeed before running a TPU benchmark.")
QUICK = True  # False: longer, prespecified protocol; not an optimization sweep.
RESULTS = KIT / "results" / "{app}-tpu"
RESULTS.parent.mkdir(exist_ok=True)
command = [sys.executable,"-u",str(KIT / "applications/{app}/run.py"),"--backend","tpu","--out",str(RESULTS)]
if QUICK: command.append("--quick")
with (KIT / "run.log").open("w") as log:
    with subprocess.Popen(command,cwd=KIT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
        try:
            for line in process.stdout:
                print(line,end="",flush=True); log.write(line); log.flush()
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
print("Results:",RESULTS)
if code: raise RuntimeError("Experiment failed. Download diagnostics using cell 4.")
from IPython.display import display, Image
for name in ("learning.png","samples.png","error-vs-time.png"):
    if (RESULTS / name).exists(): display(Image(filename=str(RESULTS / name)))
'''
        for lang,title in [('fr',fr),('en',en)]:
            if lang=='fr':
                prose=[f'# {title}\n\nPar Pascal Montmory pour ENINCA CONSULTING. Prototype contrôlé sur CPU ; compilation native et performances TPU à vérifier. Aucun gain commercial n’est garanti. La reproductibilité exacte concerne les entiers, pas les transformations flottantes ou un entraînement complet. Le pilote court est activé par défaut.\n\nChoisir **Exécution > Modifier le type d’environnement > TPU**. L’allocation dépend de votre compte et de ses quotas. Aucun identifiant GitHub ni envoi automatique des résultats.',
                       '## 1. Dépendances et accès TPU\nInstalle les versions épinglées et vérifie un vrai TPU. En cas d’échec, télécharger le diagnostic avec la cellule 4 ; aucun repli CPU.',
                       '## 2. Sources et empreintes\nKit autonome intégré au notebook, avec contrôles SHA-256.',
                       '## 3. Vérifications puis expérience\nLes compilations précèdent les chronométrages. Garder QUICK=True pour le premier essai. Le protocole complet utilise les paramètres définis avant les mesures. Les durées CPU ne prédisent pas les durées TPU.',
                       '## 4. Télécharger résultats ou diagnostics\nUtiliser aussi après un échec de la première ou de la troisième cellule.']
            else:
                prose=[f'# {title}\n\nBy Pascal Montmory for ENINCA CONSULTING. CPU-checked prototype; native TPU compilation and performance remain unverified. No commercial speedup is guaranteed. The exact contract covers integers, not floating-point transforms or complete training. The short pilot is enabled by default.\n\nSelect **Runtime > Change runtime type > TPU**. Allocation depends on your account and quotas. No GitHub credential or automatic results transmission.',
                       '## 1. Dependencies and TPU access\nInstall pinned versions and require an actual TPU. On failure download diagnostics with cell 4; no CPU fallback.',
                       '## 2. Sources and hashes\nStandalone embedded kit with SHA-256 checks.',
                       '## 3. Checks then experiment\nCompilation precedes timing. Keep QUICK=True for the first run. The full protocol uses settings fixed before measurements. CPU durations do not predict TPU durations.',
                       '## 4. Download results or diagnostics\nAlso use after a failure in the first or third cell.']
            cells=[cell('markdown',prose[0],0)]
            for text,code in zip(prose[1:],(setup,extract,run,download)):
                compile(code,'notebook','exec')
                cells += [cell('markdown',text,len(cells)),cell('code',code,len(cells)+1)]
            nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=cells)
            name=filename+('.en' if lang=='en' else '')+'.ipynb'
            (args.out/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps(dict(files=len(paths),kit_sha256=digest,output=str(args.out))))


if __name__=='__main__':main()
