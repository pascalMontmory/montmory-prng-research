"""Search, freeze, then confirm on disjoint seeds; check every timed output."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics as st
import time
import signal

p=argparse.ArgumentParser()
p.add_argument("--out",type=Path,required=True)
p.add_argument("--cpu-smoke",action="store_true",help="Harness test only, no TPU performance claim")
args=p.parse_args()
os.environ["JAX_PLATFORMS"]="cpu" if args.cpu_smoke else "tpu"
import jax
import numpy as np
import kernels as k
from support import unpack
from verify_candidates import verify

jax.config.update("jax_enable_x64",False)
jax.config.update("jax_threefry_partitionable",True)
ROOT=Path(__file__).resolve().parent

def interrupted(signum, frame):
    raise KeyboardInterrupt("Campaign interrupted")


def digest(value):
    return hashlib.sha256(np.asarray(unpack(value),dtype="<u8").tobytes()).hexdigest()


def main():
    signal.signal(signal.SIGTERM, interrupted)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    report=dict(status="running",backend_requested="cpu" if args.cpu_smoke else "tpu",cpu_smoke=args.cpu_smoke,
                jax=jax.__version__,jaxlib=jax.lib.__version__,platform=platform.platform(),
                logical_lanes=2048,draws_per_lane=4096,base=k.BASE,
                timing="host wall time: dispatch/allocation/wait included; keys/compilation/transfers/checks excluded",
                comparisons="same eight output bytes; Montmory and Threefry Pallas preserve their respective reference streams",
                compilation=[],selection={},summary=[],samples=0,checked_words=0)
    def save(): (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    rows=[]
    try:
        print("Initialisation du backend demande...",flush=True)
        devices=jax.devices()
        report.update(backend=jax.default_backend(),devices=[str(d) for d in devices],device_kind=devices[0].device_kind)
        if report["backend"]!=report["backend_requested"]: raise RuntimeError("Unexpected backend")
        if jax.__version__!="0.10.2" or jax.lib.__version__!="0.10.2": raise RuntimeError("Pinned JAX versions required")
        print("Backend:",report["backend"],report["device_kind"],flush=True)
        if args.cpu_smoke:
            available={n:s for n,s in k.candidates().items() if s["kind"]!="pallas"}
        else:
            checked=verify(out/"candidate-conformance",False)
            available={n:s for n,s in k.candidates().items() if s["kind"] in ("vendor","write") or n in checked["candidates"]}
        refs=json.loads((ROOT/"reference/optimization-hashes.json").read_text())
        if refs["jax"]!=jax.__version__ or refs["jaxlib"]!=jax.lib.__version__: raise RuntimeError("Reference version mismatch")
        baseline="jax_original" if args.cpu_smoke else "pallas_original_32"
        sizes=k.SIZES[:1] if args.cpu_smoke else k.SIZES
        search_seeds=k.SEARCH_SEEDS[:2] if args.cpu_smoke else k.SEARCH_SEEDS
        confirm_seeds=k.CONFIRM_SEEDS[:2] if args.cpu_smoke else k.CONFIRM_SEEDS
        report.update(sizes=sizes,search_seeds=search_seeds,confirm_seeds=confirm_seeds,baseline=baseline,
                      confirmation_blocks=3,confirmation_repetitions_per_block=3)
        if set(search_seeds)&set(confirm_seeds): raise RuntimeError("Seed overlap")
        compiled={}
        fields=("phase","words","block","seed","repetition","engine","nanoseconds","output_sha256","check")
        with (out/"measurements.csv").open("w",newline="") as file,jax.default_device(devices[0]):
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
            def measure(phase,count,names,seeds,blocks,repetitions):
                for block in range(blocks):
                    for si,seed in enumerate(seeds):
                        arguments={name:k.arguments(available[name],seed) for name in names}
                        expected={}
                        for name in names:
                            family=available[name]["family"]
                            target=None if family=="rbg" else refs["sha256"][f"{family}/{count}/{seed}"]
                            for _ in range(2):
                                actual=compiled[count,name](arguments[name]);jax.block_until_ready(actual)
                                value=digest(actual)
                                if target is None: target=value
                                if value!=target: raise AssertionError(f"Output mismatch warmup {name}/{count}/{seed}")
                            expected[name]=target
                        rng=random.Random(20261021+count+block*1009+si+(0 if phase=="search" else 1000003))
                        for repetition in range(repetitions):
                            order=list(names);rng.shuffle(order)
                            for name in order:
                                start=time.perf_counter_ns()
                                actual=compiled[count,name](arguments[name]);jax.block_until_ready(actual)
                                elapsed=time.perf_counter_ns()-start
                                value=digest(actual)
                                if value!=expected[name]: raise AssertionError(f"Output mismatch timed {name}/{count}/{seed}")
                                row=dict(phase=phase,words=count,block=block,seed=seed,repetition=repetition,engine=name,
                                         nanoseconds=elapsed,output_sha256=value,
                                         check="same_backend_repeatability" if available[name]["family"]=="rbg" else "fixed_cpu_reference")
                                writer.writerow(row);rows.append(row);report["samples"]+=1;report["checked_words"]+=count
                        file.flush();save()
                        print(f"{phase}: {count} mots, bloc {block+1}/{blocks}, seed {si+1}/{len(seeds)}: sorties PASS",flush=True)
            for count in sizes:
                print(f"Recherche: compilation de {len(available)} configurations pour {count} mots",flush=True)
                usable=[]
                for name,spec in available.items():
                    print("  Compilation",name,flush=True)
                    entry=dict(words=count,engine=name);start=time.perf_counter()
                    try:
                        compiled[count,name]=jax.jit(k.function(spec,count)).lower(k.arguments(spec,0)).compile()
                        entry["status"]="pass";usable.append(name)
                    except Exception as error:
                        entry.update(status="rejected",error=repr(error))
                        if spec["kind"]!="pallas" or name==baseline: raise
                    finally:
                        entry["seconds"]=time.perf_counter()-start;report["compilation"].append(entry);save()
                measure("search",count,usable,search_seeds,1,3)
                def search_score(name):
                    return st.median(st.median(r["nanoseconds"] for r in rows if r["phase"]=="search" and r["words"]==count
                                              and r["engine"]==name and r["seed"]==seed) for seed in search_seeds)
                mont=sorted((n for n in usable if available[n]["family"]=="montmory"),key=search_score)
                three=sorted((n for n in usable if available[n]["family"]=="threefry2x32"),key=search_score)
                finalists=list(dict.fromkeys([*mont[:2],baseline,"jax_original",*k.VENDORS,three[0],"write_NOT_PRNG","rbg_same_backend"]))
                report["selection"][str(count)]=dict(primary=mont[0],secondary=mont[1],threefry=three[0],finalists=finalists)
                # Release unselected compilation objects, leaving selected kernels precompiled.
                for name in usable:
                    if name not in finalists: del compiled[count,name]
                save()
            selection_path=out/"frozen-selection.json"
            selection_path.write_text(json.dumps(report["selection"],indent=2)+"\n")
            report["selection_sha256"]=hashlib.sha256(selection_path.read_bytes()).hexdigest()
            print("Selection figee. Debut de la confirmation sur des seeds distinctes.",flush=True)
            for count in sizes:
                selection=report["selection"][str(count)]
                measure("confirmation",count,selection["finalists"],confirm_seeds,3,3)
                primary=selection["primary"]
                times={name:{seed:st.median(r["nanoseconds"] for r in rows if r["phase"]=="confirmation" and r["words"]==count
                                          and r["engine"]==name and r["seed"]==seed) for seed in confirm_seeds}
                       for name in selection["finalists"]}
                for name in selection["finalists"]:
                    ratios=[times[name][seed]/times[primary][seed] for seed in confirm_seeds]
                    item=dict(words=count,engine=name,GB_per_second=8*count/st.median(times[name].values()),
                              primary=primary,primary_speedup_vs_engine=st.median(ratios),
                              primary_faster_seeds=sum(v>1 for v in ratios),seed_count=len(confirm_seeds),ratios=ratios)
                    report["summary"].append(item)
                    print(f"{name:28s} {item['GB_per_second']:9.3f} GB/s; primaire/ref={st.median(ratios):.4f}; {item['primary_faster_seeds']}/{len(confirm_seeds)}",flush=True)
                base_ratios=[times[baseline][s]/times[primary][s] for s in confirm_seeds]
                # A rule fixed before confirmation, never replacing primary by another held-out winner.
                adopted=primary if primary!=baseline and st.median(base_ratios)>=1.03 and all(v>1 for v in base_ratios) else baseline
                selection["recommended_after_confirmation"]=adopted
                selection["rule"]="Use search primary only if >=3% median gain and faster on every held-out seed; otherwise retain baseline"
                save()
            if hashlib.sha256(selection_path.read_bytes()).hexdigest()!=report["selection_sha256"]:
                raise RuntimeError("Frozen selection was modified")
        report["status"]="pass"
    except BaseException as error:
        report.update(status="fail",error=repr(error));raise
    finally:
        save()


if __name__=="__main__": main()
