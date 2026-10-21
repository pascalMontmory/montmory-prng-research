#!/usr/bin/env python3
"""Execute the frozen TACM comparison. Raw estimates retained, no automatic ranking."""
import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta, binom, chi2

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
PROTOCOL_SHA=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
RESULTS=ROOT/'results'
NATIVE={name:i for i,name in enumerate(P['engines'][:5])}
NUMPY={'numpy_PCG64':np.random.PCG64,'numpy_PCG64DXSM':np.random.PCG64DXSM,
       'numpy_MT19937':np.random.MT19937,'numpy_SFC64':np.random.SFC64,'numpy_Philox4x64':np.random.Philox}
LIB=ct.CDLL(str(ROOT/'build/libtacm.so'))
PTR=ct.POINTER(ct.c_uint64)
LIB.tacm_fill.argtypes=[ct.c_int,ct.c_uint64,ct.c_uint64,ct.c_size_t,PTR]
LIB.tacm_fill.restype=ct.c_int
LIB.tacm_gather.argtypes=[ct.c_int,ct.c_uint64,ct.c_size_t,PTR,PTR]
LIB.tacm_gather.restype=ct.c_int
LIB.tacm_check.argtypes=[ct.c_char_p]
LIB.tacm_check.restype=ct.c_int


def seed(namespace,group,replicate=0):
    key=f'TACM-20261021-v1|{namespace}|{group}|{replicate}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')


def raw(engine,s,n):
    if engine in NATIVE:
        out=np.empty(n,dtype=np.uint64)
        code=LIB.tacm_fill(NATIVE[engine],s,0,n,out.ctypes.data_as(PTR))
        if code: raise RuntimeError(f'fill error {code}')
        return out
    g=NUMPY[engine](s)
    if engine=='numpy_MT19937':
        out=g.random_raw(n*2)
        return out[::2] | (out[1::2]<<np.uint64(32))
    return g.random_raw(n)


def uniforms(words):
    return (words>>np.uint64(11)).astype(np.float64)*2.0**-53


def histogram_p(codes,cells):
    counts=np.bincount(codes,minlength=cells)
    expected=len(codes)/cells
    statistic=float(np.sum((counts-expected)**2/expected))
    return statistic,float(chi2.sf(statistic,cells-1))


def structural(u,words=None):
    b=np.floor(u*16).astype(np.int64)
    out={'marginal_q16':histogram_p(b,16)}
    b8=np.floor(u*8).astype(np.int64)
    out['disjoint_pairs_q8']=histogram_p(8*b8[::2]+b8[1::2],64)
    if words is not None:
        out['raw_low_nibble_q16']=histogram_p((words&np.uint64(15)).astype(np.int64),16)
    return out


def lane_joint(u):
    b=np.floor(u*8).astype(np.int64)
    return histogram_p((8*b[::2,:]+b[1::2,:]).reshape(-1),64)


def cp_bounds(k,n,alpha=.05):
    lo=0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    hi=1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return lo,hi


def calibration():
    cfg=P['calibration']; rows=[]
    for engine in cfg['sources']:
        counts={name:0 for name in ['marginal_q16','disjoint_pairs_q8','raw_low_nibble_q16','lane_joint_q8',*cfg['controls']]}
        for trial in range(cfg['trials_per_source']):
            words=raw(engine,seed('calibration-'+engine,trial),cfg['draws'])
            u=uniforms(words)
            values=structural(u,words)
            values['lane_joint_q8']=lane_joint(u.reshape(32,-1))
            for name,(_,p) in values.items(): counts[name]+=int(p<cfg['test_alpha'])
            biased=structural(u*.5)['marginal_q16'][1]
            duplicates=structural(np.repeat(u[:len(u)//2],2))['disjoint_pairs_q8'][1]
            lanes=u.reshape(32,-1).copy(); lanes[1::2]=lanes[::2]
            duplicated_lanes=lane_joint(lanes)[1]
            for name,p in zip(cfg['controls'],[biased,duplicates,duplicated_lanes]):
                counts[name]+=int(p<cfg['test_alpha'])
        for name,k in counts.items():
            lo,hi=cp_bounds(k,cfg['trials_per_source'])
            control=name in cfg['controls']
            passed=lo>cfg['power_lower_95_gate'] if control else hi<cfg['false_positive_upper_95_gate']
            rows.append(dict(source=engine,diagnostic=name,trials=cfg['trials_per_source'],rejections=k,
                             rejection_rate=k/cfg['trials_per_source'],CP95_low=lo,CP95_high=hi,
                             kind='injected_defect' if control else 'reference',gate_pass=passed))
        print(f'Calibration {engine} terminée',flush=True)
    df=pd.DataFrame(rows); df.to_csv(RESULTS/'calibration.csv',index=False)
    if not df.gate_pass.all():
        raise RuntimeError('Calibration gate failed; inspect before comparative inference')


def observations(u):
    # Disjoint pairs are separate IID observations under the stated null.
    pair=u.reshape(-1,2)
    return [('identity',u,.5,1/12,None),('quadratic',u*u,1/3,4/45,None),
            ('sin',np.sin(2*np.pi*u),0.,.5,None),
            ('rare_099',u>.99,.01,.0099,.01),
            ('pair_product',pair[:,0]*pair[:,1],.25,7/144,None),
            ('pair_rare_099',(pair[:,0]>.99)&(pair[:,1]>.99),.0001,.0001*.9999,.0001)]


def compare():
    gcount=P['groups']; rcount=P['replicates']; n=P['draws']; diag=[]; totals=[]
    for engine in P['engines']:
        rows=[]; start=time.perf_counter()
        for group in range(gcount):
            for rep in range(rcount):
                s=seed('functional',group,rep)
                words=raw(engine,s,n); u=uniforms(words)
                if rep==0:
                    for name,(stat,p) in structural(u,words).items():
                        diag.append(dict(engine=engine,group=group,diagnostic=name,statistic=stat,p_value=p))
                for name,x,mu,sigma2,prob in observations(u):
                    m=len(x); estimate=float(np.mean(x)); variance=float(np.var(x,ddof=1))
                    k=int(np.count_nonzero(x)) if prob is not None else -1
                    if prob is None:
                        covered=abs(estimate-mu)<=1.959963984540054*math.sqrt(variance/m)
                    else:
                        # CP interval membership: equivalent to both binomial tail tests.
                        covered=(binom.cdf(k,m,prob)>=.025 and binom.sf(k-1,m,prob)>=.025)
                    rows.append(dict(engine=engine,group=group,replicate=rep,seed=str(s),workload=name,
                                     sample_count=m,estimate=estimate,target=mu,sigma2=sigma2,
                                     squared_error=(estimate-mu)**2,coverage95=int(covered),count=k))
            if (group+1)%32==0:
                print(f'{engine}: {group+1}/{gcount} groupes, {time.perf_counter()-start:.1f} s',flush=True)
        pd.DataFrame(rows).to_csv(RESULTS/f'estimates-{engine}.csv',index=False)
        totals.append(dict(engine=engine,seconds=time.perf_counter()-start,raw_words=gcount*rcount*n))
    pd.DataFrame(diag).to_csv(RESULTS/'structural.csv',index=False)
    pd.DataFrame(totals).to_csv(RESULTS/'campaign_times.csv',index=False)


def multi_lane():
    cfg=P['multi_lane']; indices=(np.arange(cfg['lanes'],dtype=np.uint64)[:,None]*4096+
                                 np.arange(cfg['positions_per_lane'],dtype=np.uint64)[None,:]*256).reshape(-1)
    rows=[]
    for engine in cfg['engines']:
        for group in range(cfg['seeds']):
            s=seed('lanes',group); words=np.empty(len(indices),dtype=np.uint64)
            code=LIB.tacm_gather(NATIVE[engine],s,len(indices),indices.ctypes.data_as(PTR),words.ctypes.data_as(PTR))
            if code: raise RuntimeError(f'gather error {code}')
            matrix=words.reshape(cfg['lanes'],cfg['positions_per_lane'])
            stat,p=lane_joint(uniforms(matrix))
            # Exact word equality of selected lane prefixes, not equality of states.
            unique=np.unique(matrix,axis=0).shape[0]
            rows.append(dict(engine=engine,group=group,seed=str(s),diagnostic='lane_joint_q8',
                             statistic=stat,p_value=p,duplicate_selected_lane_prefixes=cfg['lanes']-unique))
        print(f'Inter-lanes {engine}: {cfg["seeds"]} seeds × {cfg["lanes"]} lanes',flush=True)
    pd.DataFrame(rows).to_csv(RESULTS/'lanes.csv',index=False)


def timing():
    cfg=P['timing']; engines=P['engines']; rows=[]
    for engine in engines: uniforms(raw(engine,42,65536))
    for trial in range(cfg['repetitions']):
        order=engines[trial%len(engines):]+engines[:trial%len(engines)]
        for engine in order:
            s=seed('timing',trial)
            begin=time.perf_counter_ns()
            u=uniforms(raw(engine,s,cfg['bulk_draws']))
            elapsed=time.perf_counter_ns()-begin
            checksum=float(u.sum())
            rows.append(dict(engine=engine,trial=trial,words=len(u),nanoseconds=elapsed,
                             ns_per_word=elapsed/len(u),checksum=checksum))
    pd.DataFrame(rows).to_csv(RESULTS/'timings.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['all','calibration','compare','lanes','timing'],default='all')
    args=parser.parse_args()
    assert LIB.tacm_check(str(ROOT/'deps/random123/tests/kat_vectors').encode())==0
    record=RESULTS/'execution.json'
    if record.exists():
        prior=json.loads(record.read_text())
        if prior['protocol_sha256']!=PROTOCOL_SHA: raise RuntimeError('protocol changed during campaign')
    else:
        prior=dict(protocol_sha256=PROTOCOL_SHA,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   platform=platform.platform(),versions=dict(numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__),
                   source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['run_campaign.py','src/engines.cpp','protocol.json']},completed=[])
        record.write_text(json.dumps(prior,indent=2)+'\n')
    phases=['calibration','compare','lanes','timing'] if args.phase=='all' else [args.phase]
    for phase in phases:
        if phase in prior['completed']: raise RuntimeError(f'phase {phase} already recorded; preserve results')
        if phase!='calibration' and 'calibration' not in prior['completed']:
            raise RuntimeError('calibration must pass first')
        globals()[{'lanes':'multi_lane'}.get(phase,phase)]()
        prior['completed'].append(phase)
        prior['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        record.write_text(json.dumps(prior,indent=2)+'\n')
    print('Phases terminées: '+', '.join(prior['completed']),flush=True)


if __name__=='__main__': main()
