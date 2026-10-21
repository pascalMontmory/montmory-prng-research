"""Run on an existing TPU, stream diagnostics, archive every completed/failed attempt."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import signal

ROOT=Path(__file__).resolve().parent

def interrupted(signum, frame):
    raise KeyboardInterrupt("Runner interrupted")


def main():
    signal.signal(signal.SIGTERM, interrupted)
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--cpu-smoke",action="store_true")
    args=p.parse_args()
    backend="cpu" if args.cpu_smoke else "tpu"
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=(args.out or ROOT/"results"/(stamp+"-tpu-opt")).resolve()
    out.mkdir(parents=True,exist_ok=True)
    status=dict(status="running",backend_requested=backend,completed=[],cpu_smoke=args.cpu_smoke)
    commands=[["check_manifest.py"],
              ["verify_native.py","--backend",backend,"--out",str(out/"historical-conformance")],
              ["campaign.py","--out",str(out/"campaign")]+(["--cpu-smoke"] if args.cpu_smoke else [])]
    try:
        with (out/"run.log").open("w") as log:
            for command in commands:
                heading="Etape: "+command[0]+"\n"
                print(heading,end="",flush=True);log.write(heading);log.flush()
                process=subprocess.Popen([sys.executable,"-u",*command],cwd=ROOT,
                    env=dict(os.environ,JAX_PLATFORMS=backend),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                try:
                    for line in process.stdout:
                        print(line,end="",flush=True);log.write(line);log.flush()
                    code=process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=20)
                        except subprocess.TimeoutExpired: process.kill();process.wait()
                    raise
                if code: raise RuntimeError(f"{command[0]} failed, exit={code}; consulter run.log")
                status["completed"].append(command[0])
        status["status"]="pass"
    except BaseException as error:
        status.update(status="fail",error=repr(error));raise
    finally:
        (out/"run-status.json").write_text(json.dumps(status,indent=2)+"\n")
        archive=out.parent/(out.name+".tar.gz")
        with tarfile.open(archive,"w:gz") as tar: tar.add(out,arcname=out.name)
        print("Archive:",archive,flush=True)


if __name__=="__main__": main()
