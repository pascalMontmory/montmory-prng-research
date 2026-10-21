#!/usr/bin/env python3
"""Recompute selected saved estimates with independent scalar reductions."""
import json
import math
from fractions import Fraction
import pandas as pd
import run_campaign as c

checked=0
for engine in c.P['engines']:
    frame=pd.read_csv(c.RESULTS/f'estimates-{engine}.csv',dtype={'seed':str},float_precision='round_trip')
    for group,rep in [(0,0),(63,15),(127,15)]:
        u=c.uniforms(c.raw(engine,c.seed('functional',group,rep),c.P['draws']))
        m=len(u)
        values={
            'identity':math.fsum(u)/m,
            'quadratic':math.fsum(float(x)*float(x) for x in u)/m,
            'sin':math.fsum(math.sin(2*math.pi*float(x)) for x in u)/m,
            'rare_099':sum(float(x)>.99 for x in u)/m,
            'pair_product':math.fsum(float(u[j])*float(u[j+1]) for j in range(0,m,2))/(m//2),
            'pair_rare_099':sum(u[j]>.99 and u[j+1]>.99 for j in range(0,m,2))/(m//2)}
        rows=frame[(frame.group==group)&(frame.replicate==rep)].set_index('workload')
        for work,value in values.items():
            assert abs(rows.loc[work,'estimate']-value)<2e-15,(engine,work)
            checked+=1
proofs={'identity_variance':str(Fraction(1,3)-Fraction(1,2)**2),
        'quadratic_variance':str(Fraction(1,5)-Fraction(1,3)**2),
        'pair_product_variance':str(Fraction(1,3)**2-Fraction(1,2)**4)}
result={'status':'pass','independently_recomputed_estimates':checked,'absolute_tolerance':2e-15,'exact_rational_variances':proofs}
(c.RESULTS/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
