"""Run on an already allocated TPU only. Never provisions cloud resources."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-rbg", action="store_true")
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out or ROOT / "results" / (stamp + "-tpu")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JAX_PLATFORMS="tpu")
    status = {"status": "running", "backend_requested": "tpu", "completed": []}
    benchmark = ["benchmark.py", "--backend", "tpu", "--out", str(out / "benchmark")]
    if args.quick:
        benchmark.append("--quick")
    if args.include_rbg:
        benchmark.append("--include-rbg")
    commands = [
        ["check_manifest.py"],
        ["verify_native.py", "--backend", "tpu", "--out", str(out / "conformance")],
        ["verify_sharding.py", "--out", str(out / "sharding")],
        benchmark,
    ]
    try:
        with (out / "run.log").open("w") as log:
            for command in commands:
                print("Running", command[0], flush=True)
                process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise RuntimeError(f"{command[0]} exited with status {code}")
                status["completed"].append(command[0])
        status["status"] = "pass"
    except BaseException as error:
        status.update(status="fail", error=repr(error))
        raise
    finally:
        (out / "run-status.json").write_text(json.dumps(status, indent=2) + "\n")
        archive = out.parent / (out.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out, arcname=out.name)
        print(f"Diagnostics: {archive}", flush=True)


if __name__ == "__main__":
    main()
