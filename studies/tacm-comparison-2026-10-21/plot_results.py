#!/usr/bin/env python3
"""Optional scientific figure; matplotlib is not required for the campaign."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

root=Path(__file__).resolve().parent
frame=pd.read_csv(root/'results/MSE_comparisons.csv')
names=['identity','quadratic','sin','rare_099','pair_product','pair_rare_099']
labels=['Moyenne U','Moyenne U²','sin(2πU)','U > 0,99','Produit UV','U et V > 0,99']
fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
for engine,label,offset,color in [('philox4x32_10','Philox4x32-10',-.13,'#277da1'),
                                  ('threefry2x32_20','Threefry2x32-20',.13,'#d97722')]:
    d=frame[frame.reference==engine].set_index('workload').loc[names]
    x=d.Montmory_over_reference_MSE.to_numpy()
    ax.errorbar(x,np.arange(len(names))+offset,
                xerr=np.vstack([x-d.simultaneous_ratio_low.to_numpy(),d.simultaneous_ratio_high.to_numpy()-x]),
                fmt='o',capsize=3,label=label,color=color)
ax.axvline(1,color='#333333',lw=1)
ax.axvline(.9,color='#999999',ls='--',lw=.8)
ax.set_yticks(np.arange(len(names)),labels)
ax.invert_yaxis()
ax.set_xlim(.80,1.25)
ax.set_xlabel('Rapport des erreurs quadratiques moyennes : Montmory / référence\n< 1 : favorable à Montmory ; > 1 : favorable à la référence')
ax.set_title('TACM : aucun avantage établi sur les six calculs\nIntervalles simultanés approximatifs à 95 % — famille de 54 comparaisons',fontsize=12)
ax.grid(axis='x',alpha=.2)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,fontsize=9,frameon=False)
fig.savefig(root/'results/MSE_ratios.png',dpi=180)
fig.savefig(root/'results/MSE_ratios.pdf')
