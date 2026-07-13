# -*- coding: utf-8 -*-
"""Fig 2 in paper-grade style (Nature/Sci Reports/Diagnostics conventions):
- Arial/Helvetica; no bold title on axes; no in-axes grid; no bar edges by default
- ColorBrewer Set2 palette (colour-blind safe); slim bars; labels above bars
- Compact caption-free axes (caption goes into article body)
- Slightly restrained CI whiskers; y-axis breaks unnecessary vertical waste
- 1.6:1 aspect (golden ratio); saved 300 dpi + PDF
"""
import csv, io, os
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np

B  = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'
FIG = f'{B}/figuras_artigo'

# Paper-grade rc
mpl.rcParams.update({
    'figure.dpi':120, 'savefig.dpi':300,
    'font.family':'sans-serif',
    'font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
    'font.size':9, 'axes.titlesize':9, 'axes.titleweight':'regular',
    'axes.labelsize':9, 'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.linewidth':0.7, 'xtick.major.width':0.6, 'ytick.major.width':0.6,
    'xtick.major.size':3, 'ytick.major.size':3,
    'axes.grid':False,
    'legend.frameon':False, 'legend.fontsize':8.5,
    'pdf.fonttype':42, 'ps.fonttype':42,
})

# ColorBrewer Set2 (colour-blind safe, muted)
PAL = {'Classical ML':'#66C2A5','CNN scratch':'#8DA0CB','Transfer':'#FC8D62',
       'RadImageNet':'#E78AC3','Hybrid':'#A6D854'}

def load_csv(p): return list(csv.DictReader(io.open(p, encoding='utf-8')))
lk = {(r['workflow'],r['dataset']):r for r in load_csv(f'{B}/leakage_backbones/all_results.csv')}
hy = load_csv(f'{B}/w11_hybrid/all_hybrid_results.csv')
def hyb(bb,c):
    for r in hy:
        if r['backbone']==bb and r['classifier']==c: return r

w7h = hyb('W7_VGG16_ImageNet','LightGBM')
rows = [
  ('Classical\nML',        90.71, None,  None,  PAL['Classical ML']),
  ('Scratch\nCNN',         93.81, 91.15, 94.69, PAL['CNN scratch']),
  ('Transfer\nlearning',   97.94, 96.61, 98.97, PAL['Transfer']),
  ('Medical\npretraining', 72.12, 69.17, 75.37, PAL['RadImageNet']),
  ('Hybrid',               float(w7h['accuracy']), float(w7h['ci95_lo']),
                                                 float(w7h['ci95_hi']), PAL['Hybrid']),
]

fig, ax = plt.subplots(figsize=(5.2, 3.4))  # ~golden ratio
xs = np.arange(len(rows))
for i,(lab,acc,lo,hi,col) in enumerate(rows):
    yerr = None if lo is None else [[acc-lo],[hi-acc]]
    ax.bar(i, acc, 0.60, color=col, edgecolor='none',
           yerr=yerr, capsize=2.2, ecolor='#333', error_kw={'linewidth':0.8})
    # value label above the CI (or above the bar if no CI)
    top = acc + ((hi-acc) if hi is not None else 0.5)
    ax.text(i, top+0.8, f'{acc:.1f}', ha='center', va='bottom', fontsize=8.5)

ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(65, 102)
ax.set_yticks([70,80,90,100])
# discreet horizontal helper lines (Sci Reports uses these often)
for y in (70,80,90,100):
    ax.axhline(y, color='#e6e6e6', lw=0.5, zorder=0)

# subtle single-panel: no title (goes to caption in the paper)
# annotate the "best" bar with a small mark
best = max(range(len(rows)), key=lambda i: rows[i][1])
ax.annotate('best', xy=(best, rows[best][1]+3), xytext=(best, 101),
            ha='center', fontsize=7.5, color='#444',
            arrowprops=dict(arrowstyle='-', lw=0.6, color='#888'))

for s in ('left','bottom'): ax.spines[s].set_color('#444')

fig.savefig(f'{FIG}/fig2_grand_comparison_paper.png', bbox_inches='tight')
fig.savefig(f'{FIG}/fig2_grand_comparison_paper.pdf', bbox_inches='tight')
print('saved fig2 paper style')
