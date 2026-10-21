"""Package the checked kit and a standalone Colab notebook with streamed logs."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

study=Path(__file__).resolve().parent
root=study/"montmory-tpu-opt"
files=sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="manifest.json")
manifest={"protocol":"montmory-tpu-opt-v1","sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
archive=study.parent/"montmory-tpu-opt-kit.tar.gz"
with tarfile.open(archive,"w:gz") as tar:
    for p in [*files,root/"manifest.json"]: tar.add(p,arcname=str(p.relative_to(study)),recursive=False)
data=archive.read_bytes();sha=hashlib.sha256(data).hexdigest()
archive.with_name(archive.name+".sha256").write_text(sha+"  "+archive.name+"\n")
payload=base64.b64encode(data).decode()
def markdown(source): return dict(cell_type="markdown",metadata={},source=source)
def code(source):
    ast.parse(source)
    return dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=source)
extract=f'''import base64, hashlib, io, os, pathlib, subprocess, sys, tarfile
ROOT = pathlib.Path(os.environ.get("MONTMORY_NOTEBOOK_ROOT", "/content")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
data = base64.b64decode({payload!r})
assert hashlib.sha256(data).hexdigest() == {sha!r}, "Archive incorrecte"
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
    for member in archive.getmembers():
        target = (ROOT / member.name).resolve()
        if not member.isfile() or not target.is_relative_to(ROOT / "montmory-tpu-opt"):
            raise RuntimeError("Entree d'archive invalide")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
KIT = ROOT / "montmory-tpu-opt"
subprocess.run([sys.executable, str(KIT / "check_manifest.py")], cwd=KIT, check=True,
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print("Kit extrait et empreintes verifiees :", KIT)
'''
install='''command = [sys.executable, "-m", "pip", "install", "-r", str(KIT / "requirements-tpu.txt")]
with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError("Installation echouee : copier le message ci-dessus.")
print("Installation terminee. Passer a la cellule suivante.")
'''
execute='''import datetime
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS = KIT / "results" / (stamp + "-tpu-opt")
RESULT_ARCHIVE = RESULTS.parent / (RESULTS.name + ".tar.gz")
command = [sys.executable, "-u", str(KIT / "run.py"), "--out", str(RESULTS)]
print("Demarrage : conformite, recherche, puis confirmation. Les compilations peuvent prendre plusieurs minutes.", flush=True)
with subprocess.Popen(command, cwd=KIT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
print("Archive :", RESULT_ARCHIVE, flush=True)
if code:
    raise RuntimeError("Essai non valide. Executer la cellule de telechargement pour transmettre les diagnostics.")
print("Campagne terminee : resultats et sorties controles. Telecharger l'archive ci-dessous.")
'''
download='''from google.colab import files
if not RESULT_ARCHIVE.exists():
    raise RuntimeError("Archive absente. Copier les dernieres lignes de l'erreur affichee.")
files.download(str(RESULT_ARCHIVE))
'''
notebook=dict(nbformat=4,nbformat_minor=5,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"colab":{"name":"Montmory_TPU_optimisation.ipynb"}},cells=[
    markdown("# Montmory : optimisation TPU sans changer les bits\n\nChoisir **Exécution → Modifier le type d’exécution → TPU** avant de commencer. Exécuter les quatre cellules de code dans l’ordre. Tous les fichiers sont intégrés au notebook. Aucun compte cloud payant n’est créé, aucun résultat n’est envoyé automatiquement.\n\n26 configurations : 16 Montmory, 4 Threefry Pallas, 4 API JAX, RBG et témoin. Quatre volumes jusqu’à 128 Mio. Les finalistes sont figés avant la confirmation sur sept nouvelles seeds. Une sortie différente de la référence arrête le test."),
    markdown("## 1. Préparer les fichiers"),code(extract),
    markdown("## 2. Installer les versions prévues de JAX\nAttendre la fin de l’installation."),code(install),
    markdown("## 3. Vérifier, optimiser et confirmer sur TPU\nLes messages s’affichent en direct. Laisser terminer les compilations et les trois blocs de confirmation. En cas d’erreur, exécuter quand même la cellule suivante pour télécharger les diagnostics."),code(execute),
    markdown("## 4. Télécharger les résultats\nJoindre ensuite le fichier `.tar.gz` à la conversation, ou le transférer sur le VPS."),code(download),
    markdown("Les débits incluent le lancement et l’attente depuis Python. Ils excluent compilation, préparation des clés et transferts de contrôle. Une configuration n’est recommandée que pour les volumes, la génération TPU et la version JAX mesurés. La campagne vérifie les sorties entières ; elle ne lance pas de nouvelle batterie TestU01/TACM.")])
for i,cell in enumerate(notebook["cells"]): cell["id"]=f"montmory-opt-{i}"
path=study.parent/"Montmory_TPU_optimisation.ipynb"
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
(study/"results/delivery.json").write_text(json.dumps(dict(archive=str(archive),sha256=sha,bytes=len(data),
    manifest_files=len(files),notebook=str(path),notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+"\n")
print(json.dumps(dict(archive=str(archive),sha256=sha,manifest_files=len(files),notebook=str(path)),indent=2))
