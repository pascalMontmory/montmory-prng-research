#!/usr/bin/env python3
"""Train a small Gaussian-to-digits flow; CPU checks precede TPU claims."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from applications.stochastic.runtime import experiment, csv_save


def verify(out, interpret):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from applications.stochastic import rng
    from applications.diffusion import model
    oracle = rng.Oracle(out/'oracle')
    words_checked = elements = 0
    # Non-multiple tile sizes, full-width seeds and a uint32 carry.
    for engine in ('threefry','montmory'):
        for seed,base,batch in ((0,0,1),(42,2**32-7,65),(2**64-1,2**64-8192,129)):
            p = rng.parameters(engine,seed,base)
            host = rng.host_words(engine,seed,base,batch*32,oracle)
            for a,b in zip(rng.raw_jax(engine,p,batch*32),host):
                np.testing.assert_array_equal(a,b)
                words_checked += a.size
            x = np.linspace(-1,1,batch*64,dtype=np.float32).reshape(batch,64)
            t = np.linspace(.01,.99,batch,dtype=np.float32)
            z = rng.host_normals(*host).reshape(x.shape)
            expected = ((1-t[:,None])*z+t[:,None]*x, x-z)
            for rows in (8,16):
                fn = jax.jit(lambda x,t,p: model.prepare(engine,'pallas',x,t,p,rows=rows,interpret=interpret))
                actual = fn(jnp.asarray(x),jnp.asarray(t),p)
                for a,b in zip(actual,expected):
                    np.testing.assert_allclose(a,b,atol=4e-5,rtol=4e-5)
                    elements += a.size
        # One complete Adam update including gradients, same per-engine noise.
        x=jnp.asarray(x[:8]); t=jnp.asarray(t[:8])
        state=model.init_state(model.initialize(7))
        p=rng.parameters(engine,19)
        a=model.step_function(engine,'jax')(state,x,t,p,jnp.uint32(17))
        b=model.step_function(engine,'pallas',interpret=interpret)(state,x,t,p,jnp.uint32(17))
        for left,right in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_allclose(left,right,rtol=5e-4,atol=5e-6)
    # Address reservation stays exact even across large step indices.
    p=rng.parameters('montmory',42)
    for s in (0,1,2**32-1):
        q=np.asarray(rng.at_step('montmory',p,jnp.uint32(s),4097))
        assert (int(q[11]) | int(q[12])<<32)==rng.BASE+s*4097
    return dict(status='pass',uint32_words=words_checked,fused_output_elements=elements,
                complete_adam_updates=2,address_checks=3,pallas_interpreted=interpret)


def run(args, report, out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sklearn.datasets import load_digits
    from applications.stochastic import rng
    from applications.diffusion import model
    for name,version in {'scikit-learn':'1.7.2','matplotlib':'3.10.7'}.items():
        if importlib.metadata.version(name)!=version:
            raise RuntimeError(f'Expected {name}=={version}')
    report['verification']=verify(out,args.backend=='cpu')
    print('Raw words, fused noise, tails and full Adam updates PASS',flush=True)
    if args.verify_only:
        return
    data=load_digits().data.astype(np.float32)/8-1
    order=np.random.default_rng(20261022).permutation(len(data))
    train,valid=data[order[:1400]],data[order[1400:]]
    report['dataset']=dict(name='sklearn digits (UCI optical handwritten digits)',train=len(train),
                           validation=len(valid),sha256=hashlib.sha256(data.tobytes()+order.tobytes()).hexdigest())
    steps=300 if args.quick else 1500
    seeds=(7,19) if args.quick else (7,19,42)
    batch=128
    common=np.random.default_rng(8801)
    bx=train[common.integers(len(train),size=(steps,batch))]
    bt=common.uniform(0,1,size=(steps,batch)).astype(np.float32)
    vt=common.uniform(0,1,size=len(valid)).astype(np.float32)
    vz=common.normal(size=valid.shape).astype(np.float32)
    probe=common.normal(size=(64,64)).astype(np.float32)
    report['inputs_sha256']=hashlib.sha256(bx.tobytes()+bt.tobytes()+vt.tobytes()+vz.tobytes()+probe.tobytes()).hexdigest()
    bx,bt,valid,vt,vz,probe=map(jnp.asarray,(bx,bt,valid,vt,vz,probe))
    jax.block_until_ready((bx,bt,valid,vt,vz,probe))
    variants=[('threefry','jax'),('montmory','jax'),('threefry','native')]
    if args.backend=='tpu': variants += [('threefry','pallas'),('montmory','pallas')]
    report['configuration']=dict(steps=steps,seeds=seeds,batch=batch,width=128,tile_rows=8,
        model='80-128-128-64 SiLU velocity MLP; Adam 0.001; uniform t; Euler 40',
        gaussian='23-bit midpoint Box-Muller; native comparator uses jax.random.normal',
        quality='held-out velocity MSE with fixed common noise; images illustrative, no FID claim',
        timing='resident-input full training step through block_until_ready; compile, input slicing and validation excluded')
    compiled={}; report['compilation_seconds']={}
    initial=model.init_state(model.initialize(seeds[0]))
    for e,impl in variants:
        name=f'{e}_{impl}'; p=rng.parameters(e,seeds[0]); fn=model.step_function(e,impl)
        begin=time.perf_counter(); compiled[name]=fn.lower(initial,bx[0],bt[0],p,jnp.uint32(0)).compile()
        report['compilation_seconds'][name]=time.perf_counter()-begin
        for _ in range(2): jax.block_until_ready(compiled[name](initial,bx[0],bt[0],p,jnp.uint32(0)))
    rows=[]; curves=[]; report['training']=[]; images={}
    for seed in seeds:
        initial=model.init_state(model.initialize(seed))
        states={f'{e}_{i}':initial for e,i in variants}
        params={e:rng.parameters(e,seed) for e,_ in variants}
        histories={name:[] for name in states}
        loss0=float(model.heldout_loss(initial[0],valid,vt,vz))
        for name in states: curves.append(dict(seed=seed,variant=name,step=0,heldout_mse=loss0))
        schedule=np.random.default_rng(seed)
        for number in range(steps):
            x,t,idx=bx[number],bt[number],jnp.uint32(number)
            jax.block_until_ready((x,t,idx,params,states))
            for pos,choice in enumerate(schedule.permutation(len(variants))):
                e,impl=variants[int(choice)]; name=f'{e}_{impl}'
                begin=time.perf_counter_ns()
                states[name],loss=compiled[name](states[name],x,t,params[e],idx)
                jax.block_until_ready((states[name],loss))
                ns=time.perf_counter_ns()-begin
                value=float(loss)
                if not np.isfinite(value): raise AssertionError('Nonfinite training loss')
                histories[name].append(value)
                rows.append(dict(seed=seed,variant=name,step=number,order=pos,elapsed_ns=ns,loss=value))
            if (number+1)%100==0 or number==steps-1:
                for name,state in states.items():
                    val=float(model.heldout_loss(state[0],valid,vt,vz))
                    if not np.isfinite(val): raise AssertionError('Nonfinite heldout loss')
                    curves.append(dict(seed=seed,variant=name,step=number+1,heldout_mse=val))
                print(f'Seed {seed}: {number+1}/{steps} complete',flush=True)
        for name,state in states.items():
            if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(state)):
                raise AssertionError('Nonfinite trained state')
            samples=np.asarray(model.generate(state[0],probe))
            if not np.isfinite(samples).all(): raise AssertionError('Nonfinite generated image')
            np.save(out/f'samples-{name}-{seed}.npy',samples)
            images[name,seed]=samples
            report['training'].append(dict(seed=seed,variant=name,initial_heldout_mse=loss0,
                final_heldout_mse=float(model.heldout_loss(state[0],valid,vt,vz)),
                median_step_ms=float(np.median([r['elapsed_ns'] for r in rows if r['seed']==seed and r['variant']==name]))/1e6))
        if args.backend=='tpu':
            for engine in ('montmory','threefry'):
                a,b=f'{engine}_jax',f'{engine}_pallas'
                np.testing.assert_allclose(histories[a],histories[b],atol=2e-4,rtol=2e-3)
                for left,right in zip(jax.tree.leaves(states[a]),jax.tree.leaves(states[b])):
                    np.testing.assert_allclose(left,right,atol=2e-4,rtol=2e-3)
        csv_save(out/'steps.csv',rows); csv_save(out/'learning.csv',curves)
    report['summary']=[]
    for name in compiled:
        selected=[r for r in report['training'] if r['variant']==name]
        reference={r['seed']:r for r in report['training'] if r['variant']=='threefry_native'}
        ratios=[reference[r['seed']]['median_step_ms']/r['median_step_ms'] for r in selected]
        report['summary'].append(dict(variant=name,median_seed_step_ms=float(np.median([r['median_step_ms'] for r in selected])),
            median_heldout_mse=float(np.median([r['final_heldout_mse'] for r in selected])),
            native_reference_over_candidate=float(np.median(ratios)),seeds_faster=sum(r>1 for r in ratios)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for name in compiled:
        selected=[r for r in curves if r['variant']==name and r['seed']==seeds[0]]
        ax.plot([r['step'] for r in selected],[r['heldout_mse'] for r in selected],label=name)
    ax.set(xlabel='Training updates',ylabel='Held-out velocity MSE',title=f'{args.backend.upper()} pilot — seed {seeds[0]}')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'learning.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(len(compiled),8,figsize=(10,1.5*len(compiled)))
    for row,name in enumerate(compiled):
        for col in range(8):
            axes[row,col].imshow(images[name,seeds[0]][col].reshape(8,8),cmap='gray',vmin=-1,vmax=1)
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(name,fontsize=8)
    fig.suptitle('First eight fixed probes; display clipped to [-1, 1]; no selection')
    fig.tight_layout(); fig.savefig(out/'samples.png',dpi=130); plt.close(fig)
    report['timed_steps']=len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('cpu','tpu'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    with experiment(args) as (report,out): run(args,report,out)


if __name__=='__main__': main()
