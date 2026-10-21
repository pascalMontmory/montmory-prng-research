#!/usr/bin/env python3
"""Analyze the frozen campaign without fitting tests to the observed engines."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, kstest, t, ttest_1samp

import run_campaign as c

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
P=json.loads((ROOT/'protocol.json').read_text())


def main():
    execution=json.loads((R/'execution.json').read_text())
    assert set(execution['completed'])=={'calibration','compare','lanes','timing'}
    assert c.PROTOCOL_SHA==execution['protocol_sha256']
    for name,digest in execution['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert pd.read_csv(R/'calibration.csv').gate_pass.all()
    G=P['groups']; reps=P['replicates']; N=P['draws']
    seeds=[c.seed('functional',g,r) for g in range(G) for r in range(reps)]
    assert len(set(seeds))==G*reps
    frames=[]
    for engine in P['engines']:
        df=pd.read_csv(R/f'estimates-{engine}.csv',dtype={'seed':str},float_precision='round_trip')
        assert len(df)==G*reps*6 and set(df.engine)=={engine}
        assert not df.duplicated(['group','replicate','workload']).any()
        assert set(df.workload)==set(P['functional_workloads'])
        assert set(df.group)==set(range(G)) and set(df.replicate)==set(range(reps))
        assert np.isfinite(df.select_dtypes(include='number')).all().all()
        assert all(row.seed==str(c.seed('functional',row.group,row.replicate)) for row in df.itertuples())
        assert np.allclose(df.squared_error,(df.estimate-df.target)**2,rtol=1e-12,atol=1e-30)
        assert set(df.coverage95)<={0,1}
        assert all(row.sample_count==(N//2 if row.workload.startswith('pair_') else N) for row in df.itertuples())
        frames.append(df)
    data=pd.concat(frames,ignore_index=True)
    metrics=[]; diagnostics=[]; variance_rows=[]; binomial_rows=[]

    def ks_row(engine,name,pvalues,note):
        assert np.isfinite(pvalues).all() and np.all((pvalues>=0)&(pvalues<=1))
        test=kstest(pvalues,'uniform')
        diagnostics.append(dict(engine=engine,diagnostic=name,n=len(pvalues),
                                KS_statistic=float(test.statistic),p_value=float(test.pvalue),calibration=note))

    for (engine,work),df in data.groupby(['engine','workload'],sort=False):
        df=df.sort_values(['group','replicate'])
        mu=float(df.target.iloc[0]); sigma2=float(df.sigma2.iloc[0]); m=int(df.sample_count.iloc[0])
        errors=df.estimate.to_numpy()-mu
        mse=float(np.mean(errors**2)); normalized=mse/(sigma2/m)
        grouped=df.groupby('group').estimate.var(ddof=1).reindex(range(G))
        cov=int(df.coverage95.sum()); covlo,covhi=c.cp_bounds(cov,len(df))
        metrics.append(dict(engine=engine,workload=work,replicate_estimates=len(df),sample_count=m,
                            bias=float(errors.mean()),bias_in_IID_standard_errors=float(errors.mean()/math.sqrt(sigma2/m/len(df))),
                            MSE=mse,normalized_MSE=normalized,coverage95=cov/len(df),coverage_CP95_low=covlo,
                            coverage_CP95_high=covhi,mean_replicate_variance=float(grouped.mean()),
                            normalized_replicate_variance=float(grouped.mean()/(sigma2/m)),
                            zero_group_variances=int((grouped==0).sum()),min_group_variance=float(grouped.min()),
                            max_group_variance=float(grouped.max())))
        if work in ['rare_099','pair_rare_099']:
            # Fixed auxiliary namespace, independent of generator outputs. Exact PIT
            # identity assumes independent continuous uniform auxiliaries; here the
            # auxiliaries are reproducible 53-bit PCG64DXSM pseudo-uniforms.
            probability=.01 if work=='rare_099' else .0001
            aux_seed=c.seed('binomial-PIT-'+engine+'-'+work,0)
            aux=np.random.Generator(np.random.PCG64DXSM(aux_seed)).random(len(df))
            counts=df['count'].to_numpy(dtype=np.int64)
            assert np.all(counts>=0) and np.all(counts<=m)
            assert np.array_equal(df.estimate.to_numpy(),counts/m)
            pit=binom.cdf(counts-1,m,probability)+aux*binom.pmf(counts,m,probability)
            ks_row(engine,'binomial_PIT_'+work,pit,'Binomial IID nominal; randomized PIT; finite-grid auxiliary approximation')
            for row,value,a in zip(df.itertuples(),pit,aux):
                binomial_rows.append(dict(engine=engine,workload=work,group=row.group,replicate=row.replicate,
                                           count=row.count,auxiliary=float(a),PIT=float(value)))
        else:
            pit=chi2.cdf((reps-1)*grouped.to_numpy()/(sigma2/m),reps-1)
            ks_row(engine,'variance_PIT_'+work,pit,'Chi-square variance reference exact for IID normal estimates; CLT approximation here')
            for group,value,var in zip(range(G),pit,grouped):
                variance_rows.append(dict(engine=engine,workload=work,group=group,variance=var,PIT=value))

    structural=pd.read_csv(R/'structural.csv')
    assert len(structural)==len(P['engines'])*G*3
    assert not structural.duplicated(['engine','group','diagnostic']).any()
    for (engine,name),df in structural.groupby(['engine','diagnostic']):
        assert len(df)==G
        ks_row(engine,'structural_'+name,df.p_value.to_numpy(),'Pearson chi-square asymptotic reference; disjoint observations')
    lanes=pd.read_csv(R/'lanes.csv',dtype={'seed':str})
    assert len(lanes)==3*64 and not lanes.duplicated(['engine','group']).any()
    for engine,df in lanes.groupby('engine'):
        assert len(df)==64
        ks_row(engine,'lane_joint_q8',df.p_value.to_numpy(),'Pearson chi-square asymptotic reference; disjoint adjacent lanes')
    diag=pd.DataFrame(diagnostics)
    assert len(diag)==93
    diag['p_Bonferroni']=np.minimum(1.,len(diag)*diag.p_value)
    diag['flag_005']=diag.p_Bonferroni<.05

    # The seed-group, not the individual replicate, is the paired analysis unit.
    group_mse=data.groupby(['engine','workload','group']).squared_error.mean()
    comparisons=[]
    for engine in P['engines'][1:]:
        for work in P['functional_workloads']:
            x=group_mse.loc['montmory',work].reindex(range(G)).to_numpy()
            y=group_mse.loc[engine,work].reindex(range(G)).to_numpy()
            ratio=float(x.mean()/y.mean())
            result=ttest_1samp(x-y,0.,alternative='less')
            adjusted=min(1.,float(result.pvalue)*54)
            ratio1=float(x[:G//2].mean()/y[:G//2].mean())
            ratio2=float(x[G//2:].mean()/y[G//2:].mean())
            # Additional uncertainty description: simultaneous approximate delta
            # intervals for the ratio, using the same 54-comparison family.
            influence=x/x.mean()-y/y.mean()
            log_se=float(influence.std(ddof=1)/math.sqrt(G))
            radius=float(t.ppf(1-.05/(2*54),G-1))*log_se
            comparisons.append(dict(reference=engine,workload=work,Montmory_over_reference_MSE=ratio,
                                    first_half_ratio=ratio1,second_half_ratio=ratio2,
                                    paired_t=float(result.statistic),p_one_sided=float(result.pvalue),
                                    p_Bonferroni_54=adjusted,
                                    simultaneous_ratio_low=ratio*math.exp(-radius),
                                    simultaneous_ratio_high=ratio*math.exp(radius),
                                    advantage=ratio<.90 and adjusted<.05 and ratio1<1 and ratio2<1))
    comp=pd.DataFrame(comparisons)
    assert len(comp)==54
    timings=pd.read_csv(R/'timings.csv')
    assert len(timings)==10*15 and not timings.duplicated(['engine','trial']).any()
    ts=timings.groupby('engine').ns_per_word.agg(median='median',q10=lambda x:x.quantile(.1),q90=lambda x:x.quantile(.9)).reset_index()
    ts['million_words_per_second']=1000/ts['median']
    for filename,frame in [('metrics.csv',pd.DataFrame(metrics)),('variance_pit.csv',pd.DataFrame(variance_rows)),
                            ('binomial_pit.csv',pd.DataFrame(binomial_rows)),('diagnostics.csv',diag),
                            ('MSE_comparisons.csv',comp),('timing_summary.csv',ts)]:
        frame.to_csv(R/filename,index=False)
    summary=dict(status='complete',functional_words=len(P['engines'])*G*reps*N,
                 functional_estimates=len(data),seeds_per_engine=G*reps,
                 primary_comparisons=54,advantages=int(comp.advantage.sum()),
                 diagnostic_comparisons=len(diag),diagnostic_flags=int(diag.flag_005.sum()),
                 duplicate_selected_lane_prefixes=int(lanes.duplicate_selected_lane_prefixes.sum()),
                 normalized_MSE_min=float(pd.DataFrame(metrics).normalized_MSE.min()),
                 normalized_MSE_max=float(pd.DataFrame(metrics).normalized_MSE.max()),
                 calibration_pilot_gate_pass=False,calibration_extension_gate_pass=True,
                 analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 limitations=['t tests, ratio intervals, variance PIT and Pearson calibrations are approximate',
                              'no observed advantage does not prove equivalence or rule out other workloads',
                              'seed separation is deterministic and not a mathematical independence guarantee',
                              'initial calibration gate failure and independent extension must both be reported'])
    (R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(comp.sort_values('p_one_sided').head(8).to_string(index=False))
    print(ts.sort_values('median').to_string(index=False))


if __name__=='__main__': main()
