"""Local conformance, TPU IR export, virtual-device checks and CPU timings."""
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
    out = (args.out or ROOT / "results" / (stamp + "-cpu")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    status = {"status": "running", "backend_requested": "cpu", "completed": []}
    bench = ["benchmark.py", "--backend", "cpu", "--out", str(out / "benchmark")]
    if args.quick:
        bench.append("--quick")
    if args.include_rbg:
        bench.append("--include-rbg")
    commands = [
        ["check_manifest.py"],
        ["verify_cpu.py", "--out", str(out / "original-protocol"), "--compare", "reference/vps-report.json"],
        ["check_target.py", "--out", str(out / "tpu-export")],
        ["verify_native.py", "--backend", "cpu", "--out", str(out / "native-harness-on-cpu")],
        ["verify_sharding.py", "--out", str(out / "virtual-cpu-sharding")],
        bench,
    ]
    try:
        with (out / "run.log").open("w") as log:
            for command in commands:
                command_env = dict(env)
                if command[0] == "verify_sharding.py":
                    command_env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
                print("Running", command[0], flush=True)
                process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, env=command_env,
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
