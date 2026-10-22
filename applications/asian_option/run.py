#!/usr/bin/env python3
"""Monte Carlo / randomized Sobol Asian-option pilot with separate accuracy evidence."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from applications.stochastic.runtime import experiment,csv_save


def raw_pallas(engine,p,pairs,interpret=False):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu
    from applications.stochastic import rng
    rows=8; tile=rows*128; blocks=(pairs+tile-1)//tile
    def body(p_ref,lo_ref,hi_ref):
        i=pl.program_id(0).astype(rng.U32)*rng.U32(tile)+jnp.arange(rows,dtype=rng.U32)[:,None]*rng.U32(128)+jnp.arange(128,dtype=rng.U32)[None,:]
        lo_ref[...],hi_ref[...]=rng.pair_words(engine,p_ref,i)
    spec=pl.BlockSpec((rows,128),lambda i,p:(i,0))
    shape=jax.ShapeDtypeStruct((blocks*rows,128),rng.U32)
    lo,hi=pl.pallas_call(body,out_shape=(shape,shape),interpret=interpret,
        grid_spec=tpu.PrefetchScalarGridSpec(num_scalar_prefetch=1,grid=(blocks,),out_specs=(spec,spec)),
        compiler_params=tpu.CompilerParams(dimension_semantics=('parallel',)),name=f'asian_raw_{engine}')(p)
    return rng.interleave(lo,hi)[:pairs*2]


def generator(name,count,interpret=False):
    import jax
    import jax.numpy as jnp
    from applications.stochastic import rng
    from applications.sobol import kernels as sk
    if name.startswith('sobol'):
        f=sk.make_kernel('pallas_block_256' if 'pallas' in name else 'jax_block_256',count,aligned=True,interpret=interpret)
        return lambda p:f(*p)
    engine=name.split('_')[0]
    if 'native' in name:
        return lambda p:jax.random.normal(jax.random.wrap_key_data(p,impl='threefry2x32'),(count,128),dtype=jnp.float32)
    if 'pallas' in name:
        return lambda p:raw_pallas(engine,p,count*64,interpret).reshape(count,128)
    return lambda p:rng.interleave(*rng.raw_jax(engine,p,count*64)).reshape(count,128)


def application(name,count,interpret=False,paths=False):
    import jax
    import jax.numpy as jnp
    from applications.stochastic import rng
    from applications.asian_option.pricing import payoffs
    f=generator(name,count,interpret)
    def run(p,market):
        values=f(p)
        z=values if 'native' in name else jax.scipy.special.ndtri(rng.uniform(values))
        outputs=payoffs(z,market)
        return outputs if paths else jnp.mean(outputs,axis=0)
    return jax.jit(run)


def inputs(name,seed):
    import jax.numpy as jnp
    from applications.stochastic import rng
    from applications.sobol import reference as sr
    if name.startswith('sobol'):
        p=sr.parameters(seed)
        return (jnp.asarray(p.directions),jnp.asarray(p.shift),jnp.asarray(sr.local_table(p,256)),jnp.array([0],jnp.uint32))
    return rng.parameters(name.split('_')[0],seed)


def reference(name,seed,count,oracle):
    import numpy as np
    from scipy.special import ndtri
    from applications.stochastic import rng
    from applications.sobol import reference as sr
    if name.startswith('sobol'):
        raw=sr.scipy_words(seed,count)
    elif 'native' in name:
        z=np.asarray(generator(name,count)(inputs(name,seed)),dtype=np.float64)
        return None,z
    else:
        lo,hi=rng.host_words(name.split('_')[0],seed,rng.BASE,count*64,oracle)
        raw=np.stack((lo,hi),axis=-1).reshape(count,128)
    u=((raw>>np.uint32(9)).astype(np.float64)+.5)*2**-23
    return raw,ndtri(u)


def verify(out,interpret):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from applications.stochastic import rng
    from applications.asian_option import pricing as pr
    oracle=rng.Oracle(out/'oracle')
    n=pr.DATES
    times=np.arange(1,n+1)/n
    covariance=np.minimum.outer(times,times)
    var=.2**2*covariance.sum()/n**2
    np.testing.assert_allclose(var,.2**2*(n+1)*(2*n+1)/(6*n*n),rtol=1e-14)
    # n=1 reduces to a vanilla call; use numerical integration as another route.
    from scipy.integrate import quad
    for dates in (1,128):
        mean=np.log(100)+(.03-.5*.2**2)*(dates+1)/(2*dates)
        variance=.2**2*(dates+1)*(2*dates+1)/(6*dates*dates)
        integral=quad(lambda z:np.exp(-.03)*max(np.exp(mean+np.sqrt(variance)*z)-100,0)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-12,12,epsabs=1e-9)[0]
        np.testing.assert_allclose(pr.geometric_price(pr.MARKET,dates),integral,atol=2e-6,rtol=2e-6)
    zero=pr.MARKET.copy(); zero[3]=0
    deterministic=pr.numpy_payoffs(np.zeros((2,128)),zero)
    np.testing.assert_allclose(deterministic[:,1],pr.geometric_price(zero),atol=1e-12)
    checked=0
    for name in ('threefry_pallas','montmory_pallas','sobol_pallas'):
        for seed in (0,2**64-1):
            count=256; p=inputs(name,seed)
            raw,z=reference(name,seed,count,oracle)
            actual=jax.jit(generator(name,count,interpret))(p)
            np.testing.assert_array_equal(actual,raw); checked+=raw.size
            expected=pr.numpy_payoffs(z,pr.MARKET)
            actual=application(name,count,interpret,paths=True)(p,jnp.asarray(pr.MARKET))
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
            # For every path, arithmetic average >= geometric average.
            assert np.all(np.asarray(actual)[:,0]+1e-4>=np.asarray(actual)[:,1])
    return dict(status='pass',uint32_words=checked,analytic_quadrature_cases=2,
                deterministic_case=True,pallas_interpreted=interpret)


def run(args,report,out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import qmc,t as student
    from scipy.special import ndtri
    from applications.stochastic import rng
    from applications.sobol import reference as sr
    from applications.asian_option import pricing as pr
    report['verification']=verify(out,args.backend=='cpu')
    print('Analytic oracle, raw words and per-path payoffs PASS',flush=True)
    if args.verify_only:return
    counts=(512,2048) if args.quick else (4096,16384,65536)
    seeds=(101,211,307,401) if args.quick else tuple(1009+997*i for i in range(16))
    repetitions=3 if args.quick else 5
    names=['threefry_jax','montmory_jax','sobol_jax','threefry_native']
    if args.backend=='tpu':names+=['threefry_pallas','montmory_pallas','sobol_pallas']
    report['configuration']=dict(paths=counts,seeds=seeds,repetitions=repetitions,dates=128,
        market=pr.MARKET.tolist(),order='chronological Brownian increments; no bridge or control variate',
        normal_mapping='23-bit midpoint inverse normal CDF; native comparator uses jax.random.normal',
        sobol='LMS+shift, independent scrambles, first 2**m points including point zero',
        primary_accuracy='geometric Asian RMSE against analytic continuous-normal price',
        arithmetic_accuracy='RMS difference to independent float64 RQMC reference; reference uncertainty reported',
        timing='resident parameters, RNG + normal conversion + paths + payoffs + means, call-to-ready; setup and reference checks excluded',
        accuracy_targets=[.1,.05,.02],estimated_cloud_cost=False)
    # Independent reference uses different seeds and full 32-bit dyadic coordinates.
    ref_count=8192 if args.quick else 32768
    ref_reps=8 if args.quick else 16
    ref_values=[]; started=time.perf_counter()
    for i in range(ref_reps):
        sobol=qmc.Sobol(128,bits=32,scramble=True,rng=np.random.default_rng(900001+7919*i))
        u=sobol.random_base2(int(np.log2(ref_count)))
        # Clip only the zero endpoint to the half-bin. Upper endpoint is <1.
        z=ndtri(np.maximum(u,2**-33))
        ref_values.append(pr.numpy_payoffs(z,pr.MARKET).mean(axis=0).tolist())
    arithmetic=float(np.mean(np.array(ref_values)[:,0]))
    half=float(student.ppf(.975,ref_reps-1)*np.std(np.array(ref_values)[:,0],ddof=1)/np.sqrt(ref_reps))
    report['reference']=dict(geometric_analytic=pr.geometric_price(pr.MARKET),arithmetic=arithmetic,
        arithmetic_95_half_width=half,replicates=ref_values,paths_per_replicate=ref_count,
        setup_seconds=time.perf_counter()-started,note='Student interval across independent scrambles, not a certified bound')
    oracle=rng.Oracle(out/'campaign-oracle')
    market=jnp.asarray(pr.MARKET); rows=[]; estimates=[]; setup=[]
    report['compilation_seconds']={}
    for count in counts:
        compiled={}
        for name in names:
            p=inputs(name,seeds[0]); jax.block_until_ready(p)
            start=time.perf_counter(); compiled[name]=application(name,count).lower(p,market).compile()
            report['compilation_seconds'][f'{count}:{name}']=time.perf_counter()-start
        for seed in seeds:
            params={}; refs={}
            for name in names:
                start=time.perf_counter(); p=inputs(name,seed); jax.block_until_ready(p)
                setup.append(dict(paths=count,seed=seed,variant=name,seconds=time.perf_counter()-start))
                params[name]=p
                raw,z=reference(name,seed,count,oracle)
                if raw is not None:
                    actual=jax.jit(generator(name,count))(p)
                    np.testing.assert_array_equal(actual,raw)
                refs[name]=pr.numpy_payoffs(z,pr.MARKET).mean(axis=0)
                for _ in range(2):jax.block_until_ready(compiled[name](p,market))
            sp=sr.parameters(seed)
            sp.save(out/f'sobol-parameters-{seed}.npz')
            order=np.random.default_rng(seed+count)
            for rep in range(repetitions):
                for pos,choice in enumerate(order.permutation(len(names))):
                    name=names[int(choice)]
                    started=time.perf_counter_ns(); result=compiled[name](params[name],market)
                    jax.block_until_ready(result); elapsed=time.perf_counter_ns()-started
                    values=np.asarray(result)
                    np.testing.assert_allclose(values,refs[name],atol=1e-4,rtol=1e-4)
                    rows.append(dict(paths=count,seed=seed,variant=name,repeat=rep,order=pos,elapsed_ns=elapsed,
                                     arithmetic=float(values[0]),geometric=float(values[1])))
            for name in names:
                selected=[r for r in rows if r['paths']==count and r['seed']==seed and r['variant']==name]
                assert len({(r['arithmetic'],r['geometric']) for r in selected})==1
                estimates.append(dict(paths=count,seed=seed,variant=name,arithmetic=selected[0]['arithmetic'],
                    geometric=selected[0]['geometric'],median_seconds=float(np.median([r['elapsed_ns'] for r in selected]))/1e9))
            print(f'{count} paths, seed {seed}: all timed outputs checked',flush=True)
        csv_save(out/'measurements.csv',rows); csv_save(out/'estimates.csv',estimates); csv_save(out/'setup.csv',setup)
    summary=[]
    for name in names:
        for count in counts:
            selected=[r for r in estimates if r['paths']==count and r['variant']==name]
            a=np.array([r['arithmetic'] for r in selected]); g=np.array([r['geometric'] for r in selected])
            native={r['seed']:r for r in estimates if r['paths']==count and r['variant']=='threefry_native'}
            ratios=[native[r['seed']]['median_seconds']/r['median_seconds'] for r in selected]
            summary.append(dict(variant=name,paths=count,replicates=len(selected),
                median_seconds=float(np.median([r['median_seconds'] for r in selected])),
                native_reference_over_candidate=float(np.median(ratios)),
                geometric_rmse=float(np.sqrt(np.mean((g-report['reference']['geometric_analytic'])**2))),
                arithmetic_rms_reference_difference=float(np.sqrt(np.mean((a-arithmetic)**2))),
                arithmetic_reference_half_width=half))
    report['summary']=summary
    report['empirical_time_to_geometric_rmse']=[]
    for name in names:
        for target in (.1,.05,.02):
            candidates=[r for r in summary if r['variant']==name and r['geometric_rmse']<=target]
            chosen=min(candidates,key=lambda r:r['paths']) if candidates else None
            report['empirical_time_to_geometric_rmse'].append(dict(variant=name,target=target,
                tested_paths=chosen['paths'] if chosen else None,seconds=chosen['median_seconds'] if chosen else None))
    report['timed_outputs_checked']=len(rows)
    csv_save(out/'summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for name in names:
        data=[r for r in summary if r['variant']==name]
        ax.loglog([r['median_seconds'] for r in data],[r['geometric_rmse'] for r in data],'o-',label=name)
    ax.set(xlabel='Median seconds per replicate',ylabel='Geometric Asian RMSE',title=f'{args.backend.upper()} pilot — empirical error, finite replicates')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'error-vs-time.png',dpi=140); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('cpu','tpu'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    with experiment(args) as (report,out):run(args,report,out)


if __name__=='__main__':main()
