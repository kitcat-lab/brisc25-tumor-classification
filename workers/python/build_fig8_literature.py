# -*- coding: utf-8 -*-
"""Figure 8: literature positioning vs our results, coloured by DATASET, labelled
with author+method. Reads the user's Comparison sheet + our audited results."""
import openpyxl, re, os
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'savefig.dpi':300,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
FIG='/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui/figuras_artigo'
XL='/mnt/c/Users/cbot/Desktop/BRISC pos graduação/literature/_literature_comparison.xlsx'

def parse_acc(s):
    if s is None: return None
    s=str(s)
    m=re.findall(r'(\d{2,3}\.?\d?)\s*%?', s)
    if not m: return None
    vals=[float(x) for x in m if 40<=float(x)<=100]
    return max(vals) if vals else None

def dataset_family(d):
    d=(d or '').lower()
    if 'brisc' in d: return 'BRISC 2025'
    if 'cheng' in d or 'figshare' in d: return 'Figshare/Cheng (3064)'
    if 'br35h' in d or 'navoneel' in d: return 'BR35H'
    if 'brats' in d: return 'BraTS'
    if 'kaggle' in d: return 'Kaggle'
    return 'Other / multi-source'

DS_COLOR={'BRISC 2025':'#1F4E4A','Figshare/Cheng (3064)':'#1F77B4','Kaggle':'#E08A1E',
          'BraTS':'#8E44AD','BR35H':'#C0392B','Other / multi-source':'#7F7F7F'}

wb=openpyxl.load_workbook(XL, data_only=True); ws=wb['Comparison']
rows=list(ws.iter_rows(values_only=True))
hdr=next((list(r) for r in rows if r and r[0]=='Study'),None)
ix={h:i for i,h in enumerate(hdr)}
entries=[]  # (label, dataset_family, acc, is_ours)
for r in rows:
    if not r or not r[0]: continue
    st=str(r[0]).strip()
    if st in ('Study','Reference') or (st.isupper() and (len(r)<2 or not r[1])): continue
    if 'THIS WORK' in st.upper(): continue  # our results added separately below
    acc=parse_acc(r[ix['Accuracy']]) if 'Accuracy' in ix and len(r)>ix['Accuracy'] else None
    if acc is None: continue
    ds=str(r[ix['Dataset']]) if 'Dataset' in ix and r[ix['Dataset']] else ''
    bb=str(r[ix['Backbone']]) if 'Backbone' in ix and r[ix['Backbone']] else ''
    ch=str(r[ix['Classifier head']]) if 'Classifier head' in ix and r[ix['Classifier head']] else ''
    yr=str(r[ix['Year']]) if 'Year' in ix and r[ix['Year']] else ''
    author=re.split(r'\d', st)[0].strip().rstrip('(').strip()
    method=(bb+(' + '+ch if ch and ch.lower() not in ('softmax','none','') else '')).strip(' +')
    lab=f"{author} et al. ({yr}) - {method[:30]}"
    entries.append([lab, dataset_family(ds), acc, False])
wb.close()

# our results (audited BRISC)
ours=[('This work - Classical (db4+GLCM+LGBM)',90.71),('This work - CNN scratch (W2)',93.81),
      ('This work - Transfer VGG16 (softmax)',97.94),('This work - RadImageNet',72.12),
      ('This work - Hybrid VGG16+LightGBM',98.67)]
for lab,acc in ours: entries.append([lab,'BRISC 2025',acc,True])

mpl.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.titleweight':'bold',
                     'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                     'legend.fontsize':9,'axes.grid':True,'grid.alpha':0.25})
entries.sort(key=lambda e:e[2])
fig,ax=plt.subplots(figsize=(11.5, 0.34*len(entries)+1.2))
y=range(len(entries))
for i,(lab,ds,acc,ours_) in enumerate(entries):
    ax.barh(i,acc,color=DS_COLOR[ds],edgecolor='black',lw=1.4 if ours_ else 0.5,
            hatch='//' if ours_ else None, alpha=0.95)
    ax.text(acc+0.3,i,f'{acc:.1f}',va='center',fontsize=8,fontweight='bold' if ours_ else 'normal')
ax.set_yticks(list(y)); ax.set_yticklabels([e[0] for e in entries],fontsize=7.6)
for t,e in zip(ax.get_yticklabels(),entries):
    if e[3]: t.set_fontweight('bold')
ax.set_xlim(65,103); ax.set_xlabel('Reported test accuracy (%)')
ax.set_title('Positioning vs literature (colour = dataset; hatched = this work, BRISC 2025 audited)')
handles=[plt.Rectangle((0,0),1,1,color=c) for c in DS_COLOR.values()]
# legend OUTSIDE the axes, never overlapping the bars\n
ax.legend(handles,list(DS_COLOR.keys()),title='Dataset',loc='center left',
          bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=8.5,title_fontsize=9)
ax.grid(axis='x',alpha=0.25); ax.set_axisbelow(True)
fig.savefig(f'{FIG}/fig8_literature_positioning.png',bbox_inches='tight')
fig.savefig(f'{FIG}/fig8_literature_positioning.pdf',bbox_inches='tight')
fig.savefig(f'{FIG}/fig8_literature_positioning.svg',bbox_inches='tight')
print('saved fig8; entries:',len(entries),'| datasets:',sorted(set(e[1] for e in entries)))
