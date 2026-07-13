# -*- coding: utf-8 -*-
"""Build appendix A: classical ML compared on OFFICIAL vs RANDOM split.
Reads classic_ml_all_results.csv (already restricted to the ML classico folder)."""
import csv, io, os, math
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'savefig.dpi':300,'font.family':'DejaVu Sans','font.size':10,
    'axes.titlesize':11,'axes.titleweight':'bold','axes.labelsize':10,
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25})

B='/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'
FIG=f'{B}/figuras_artigo'; os.makedirs(FIG, exist_ok=True)
rows=list(csv.DictReader(io.open(f'{B}/classic_ml_all_results.csv',encoding='utf-8')))
def f(x):
    try: return float(x)
    except: return math.nan

def best_per(split):
    d={}
    for r in rows:
        if r.get('split')!=split: continue
        s=r['scenario']
        if s not in d or f(r['accuracy'])>f(d[s]['accuracy']): d[s]=r
    return d

bo, br = best_per('original'), best_per('aleatoria')
scen = sorted(set(bo)&set(br), key=lambda s:-f(bo[s]['accuracy']))
# ---------- figure ----------
fig, ax = plt.subplots(figsize=(11.5, 4.6))
import numpy as np
x=np.arange(len(scen)); w=0.38
ax.bar(x-w/2,[f(bo[s]['accuracy']) for s in scen],w,color='#1F4E4A',edgecolor='black',lw=0.6,label='Official split (audited holdout)')
ax.bar(x+w/2,[f(br[s]['accuracy']) for s in scen],w,color='#C99A2E',edgecolor='black',lw=0.6,label='Random split')
for i,s in enumerate(scen):
    d=f(br[s]['accuracy'])-f(bo[s]['accuracy'])
    ax.text(i, max(f(bo[s]['accuracy']),f(br[s]['accuracy']))+0.4, f'{d:+.2f}', ha='center', fontsize=8.5, color='#B22')
ax.set_xticks(x); ax.set_xticklabels([s.replace('_',' ') for s in scen], rotation=20, ha='right', fontsize=8.5)
ax.set_ylim(80,94); ax.set_ylabel('Best accuracy per scenario (%)')
ax.set_title('Classical ML: official (audited) vs random split (delta annotated in red)')
ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), frameon=False)
fig.savefig(f'{FIG}/figA1_split_comparison.png', bbox_inches='tight')
fig.savefig(f'{FIG}/figA1_split_comparison.pdf', bbox_inches='tight')
fig.savefig(f'{FIG}/figA1_split_comparison.svg', bbox_inches='tight')
print('saved figA1_split_comparison')

# ---------- markdown appendix ----------
L=['# Appendix A - Classical ML: official vs random split\n',
   '> Every row is the best (classifier x NL quantisation) for that scenario on the\n'
   '> stated split. Same 198 wavelet+GLCM features and same 19-classifier grid; only\n'
   '> the train/test partition changes. Source: `classic_ml_all_results.csv` aggregated\n'
   '> from the MATLAB `Relatorio_*.xlsx` reports in the `ML classico/` folder.\n',
   '## Table A.1. Best per scenario on each split\n',
   '| Wavelet | Skull | Feat.Sel. | Split | Classifier | NL | Acc % | F1 % | AUC % |',
   '|---------|-------|-----------|-------|-----------|----|------:|-----:|------:|']
def key_of(s):
    parts=s.split('_')
    wl='bior1.1' if parts[0]=='bior11' else parts[0]
    ss='Yes' if 'no_ss' not in s else 'No'
    fs='Yes' if s.endswith('_fs') and 'no_fs' not in s else 'No'
    return wl,ss,fs
for s in scen:
    wl,ss,fs=key_of(s)
    ro=bo[s]; rr=br[s]
    L.append(f"| {wl} | {ss} | {fs} | Official | {ro['classificador']} | {ro['NumLevels']} | {ro['accuracy']} | {ro['f1_score']} | {ro['auc']} |")
    L.append(f"| {wl} | {ss} | {fs} | Random   | {rr['classificador']} | {rr['NumLevels']} | {rr['accuracy']} | {rr['f1_score']} | {rr['auc']} |")

L.append('\n## Table A.2. Split-effect summary\n')
diffs=[f(br[s]['accuracy'])-f(bo[s]['accuracy']) for s in scen]
L.append('| Statistic | Value |')
L.append('|-----------|------:|')
L.append(f"| Mean delta (random - official) | {sum(diffs)/len(diffs):+.2f} pp |")
L.append(f"| Max delta                     | {max(diffs):+.2f} pp |")
L.append(f"| Min delta                     | {min(diffs):+.2f} pp |")
L.append(f"| Scenarios where random > official | {sum(1 for d in diffs if d>0)} / {len(diffs)} |")
L.append(f"| Best random scenario | {max(scen, key=lambda s:f(br[s]['accuracy']))} - {max(f(br[s]['accuracy']) for s in scen):.2f}% |")
L.append(f"| Best official scenario | {max(scen, key=lambda s:f(bo[s]['accuracy']))} - {max(f(bo[s]['accuracy']) for s in scen):.2f}% |")

L.append('\n## Comment\n'
 'The random split re-partitions the 5,041 audited images ignoring the official\n'
 'train/test boundary. Contrary to a common intuition, the observed effect is\n'
 'small and inconsistent: the mean delta is -0.57 pp and random beats the official\n'
 'split in only 3/8 scenarios (Table A.2). The best random result (bior1.1 no-SS\n'
 'no-FS LGBM NL16 = 91.38%) exceeds the best official result (db4 no-SS no-FS LGBM\n'
 'NL16 = 90.71%) by only 0.67 pp, well within test-set sampling noise on 678 images.\n'
 'This is itself an argument for the forensic audit: once exact and near-duplicate\n'
 'images have been removed, the choice of split contributes little to the reported\n'
 'accuracy. Only the *official* split is used in the main-text cross-approach\n'
 'comparison because the deep-learning workflows share that same 678-image test set.\n')

io.open(f'{B}/appendix_split_comparison.md','w',encoding='utf-8').write('\n'.join(L)+'\n')
print('wrote appendix_split_comparison.md')
