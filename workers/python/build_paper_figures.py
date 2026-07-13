# -*- coding: utf-8 -*-
"""Generate the 3 LaTeX article figures in the workbooks' teal/blue-green theme.
  fig_workflows  -- study-design schematic (teal shades)
  fig_confusion  -- 3-panel confusion matrices (GnBu colourmap): Classical, Transfer, Hybrid
  fig_gradcam    -- 6x4 Grad-CAM++ grid (CNN scratch + 5 backbones x classes), overlay-only, jet
Saved to latex_artigo/figures/ as PDF (vector) + PNG.

Runtime deps: numpy, matplotlib, PIL only (NO TensorFlow / GPU). Runs from WSL,
Windows or a Jupyter kernel -- all paths are derived from this file's location.

Grad-CAM inputs: fig_gradcam reads w11_hybrid/<model>/gradcam/<class>_raw.npy for
all six rows (W3_CNN_scratch + W6..W10). These are normally already present. If the
W3 row is ever missing, regenerate it once (inference only, CPU is fine):
    python w11_gradcam.py W3_CNN_scratch <path-to>/classification_task <path-to>/w11_hybrid
"""
import os, io, csv
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# Paths derived from this file's location so the script runs from WSL, Windows
# or a Jupyter kernel without editing hard-coded absolutes.
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../brisc_gui
_ROOT = os.path.dirname(_HERE)                              # project root
B   = _HERE
LFG = os.path.join(_ROOT, 'latex_artigo', 'figures')
os.makedirs(LFG, exist_ok=True)
CLASSES = ['glioma','meningioma','no_tumor','pituitary']
CLABEL  = ['Glioma','Meningioma','No tumor','Pituitary']

# Grad-CAM grid: same audited test base + example selection as w11_gradcam.py
# (first image alphabetically in each test/<class>/). Native input size per model.
# Auto-locate the audited test set across WSL / Windows layouts; the overlay is
# only cosmetic, so if none is found fig_gradcam falls back to the bare heatmap.
_AUD_CANDIDATES = [
    '/root/brisc/data/brisc2025_clean/classification_task',
    os.path.join(_ROOT, 'classification_task'),
    os.path.join(_ROOT, 'brisc2025_clean', 'classification_task'),
]
AUD_BASE = next((p for p in _AUD_CANDIDATES if os.path.isdir(os.path.join(p, 'test'))),
                _AUD_CANDIDATES[0])
GC_IMG_SIZE = {'W3_CNN_scratch':128,'W6_ResNet50_ImageNet':224,'W7_VGG16_ImageNet':224,
               'W8_EfficientNetB2_ImageNet':260,'W9_ConvNeXtTiny_ImageNet':224,
               'W10_RadImageNet_ResNet50':224,'W15_EffNetB1_optimised':240}

# --- User palette (matches fig_workflows PDF) --------------------------------
# Bright pastel palette from the user's own workflow schematic. Rose = data,
# butter/sand = approaches, teal = classifier heads, plum/lavender = metrics,
# slate = text and axes.
PAL_ROSE   = '#F5B5C1'
PAL_ROSE_D = '#B45E70'
PAL_SAND   = '#F9E8A1'
PAL_SAND_D = '#B08B2E'
PAL_TEAL   = '#A9DDD8'
PAL_TEAL_D = '#2E7A75'
PAL_PLUM   = '#D9C4E6'
PAL_PLUM_D = '#6E4C8E'
PAL_INK    = '#2E3149'      # dark slate for text/axes
# aliases kept for the legacy code paths (any older reference to TEAL_* still resolves)
TEAL_D = PAL_TEAL_D
TEAL_M = PAL_TEAL
TEAL_L = PAL_TEAL
TEAL_XL = PAL_TEAL
mpl.rcParams.update({
    'figure.dpi':120,'savefig.dpi':300,
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':9,'axes.titlesize':9.5,'axes.titleweight':'bold','axes.labelsize':9,
    'xtick.labelsize':8.5,'ytick.labelsize':8.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.7,
    'pdf.fonttype':42,'ps.fonttype':42,
})
def save(fig,name):
    fig.savefig(f'{LFG}/{name}.pdf',bbox_inches='tight')
    fig.savefig(f'{LFG}/{name}.png',bbox_inches='tight')
    fig.savefig(f'{LFG}/{name}.svg',bbox_inches='tight')   # editable vector
    plt.close(fig); print('saved',name)

# ---------------- FIG WORKFLOWS (top-down schematic, muted publication palette) --
def fig_workflows():
    """Study-design schematic. Top-down layers: (1) dataset, (2) audit,
    (3) split, (4) four approach branches, (5) classifier heads, (6) metrics.
    Uses a muted rose+sand+teal+plum palette that reads as scientific without
    the loud pastel of the source draft."""
    # muted publication palette (softer than the source pink/yellow draft)
    P_DATA   = '#D9A5B3'    # muted rose        -- dataset / audit / split rows
    P_APPR   = '#E6D3A3'    # warm sand         -- four approach branches
    P_HEAD   = '#8FBAB4'    # dusty teal        -- classifier heads
    P_MET    = '#B8A9C9'    # muted plum        -- metrics row
    EDGE     = '#3A3F52'    # slate ink for text and arrows
    ARROW    = '#6A7180'

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.axis('off')

    def box(x, y, w, h, t, c, tc=EDGE, fs=9.5, weight='bold'):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.05,rounding_size=0.18',
            fc=c, ec=EDGE, lw=0.9))
        ax.text(x + w / 2, y + h / 2, t,
                ha='center', va='center',
                color=tc, fontsize=fs, fontweight=weight)

    def arr(x1, y1, x2, y2, lw=1.2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>',
            mutation_scale=14, lw=lw, color=ARROW))

    # Row 1 — dataset (centered, wide)
    box(6.5, 9.4, 7.0, 1.1, 'BRISC 2025 dataset — 6,000 MRI images', P_DATA, fs=11)
    # Row 2 — audit
    box(6.5, 8.0, 7.0, 1.0,
        'Forensic audit (MD5 + pHash): 6,000 $\\to$ 5,041 images', P_DATA, fs=10.5)
    # Row 3 — split
    box(6.5, 6.6, 7.0, 1.0,
        'Official split: 4,363 train / 678 test', P_DATA, fs=10.5)
    arr(10.0, 9.4, 10.0, 9.0)
    arr(10.0, 8.0, 10.0, 7.6)

    # Row 4 — four approach branches, evenly spaced
    br_x = [0.4, 5.4, 10.4, 15.4]
    br_w = 4.2
    br_titles = [
        'Wavelet + GLCM\n(198 features)',
        'CNN from scratch\n(3 blocks)',
        'Transfer backbones\nResNet50 / VGG16 /\nEffNetB2 / ConvNeXt-Tiny',
        'RadImageNet\n(medical pretrain)',
    ]
    for x, t in zip(br_x, br_titles):
        box(x, 4.4, br_w, 1.7, t, P_APPR, fs=9.5)
        arr(10.0, 6.6, x + br_w / 2, 6.1)

    # Row 5 — classifier heads
    # left-most branch feeds "19 classifiers"; the three right branches share
    # "softmax head OR 7-classifier hybrid"
    box(0.4, 2.7, br_w, 1.2, '19 classifiers', P_HEAD, fs=10.5)
    box(5.4, 2.7, br_w * 3 + 2.0, 1.2,
        'Softmax head  OR  hybrid (7 classifiers on 128-d features)',
        P_HEAD, fs=10.5)
    arr(br_x[0] + br_w / 2, 4.4, br_x[0] + br_w / 2, 3.9)
    for i in (1, 2, 3):
        arr(br_x[i] + br_w / 2, 4.4, br_x[i] + br_w / 2, 3.9)

    # Row 6 — metrics
    box(3.0, 0.8, 14.0, 1.2,
        'Acc / F1 / AUC  $\\;+\\;$  95% bootstrap CI  $\\;+\\;$  '
        'paired McNemar  $\\;+\\;$  Grad-CAM++', P_MET, fs=11)
    arr(br_x[0] + br_w / 2, 2.7, 10.0, 2.0)
    for i in (1, 2, 3):
        arr(br_x[i] + br_w / 2, 2.7, 10.0, 2.0)

    save(fig, 'fig_workflows')

# ---------------- FIG CONFUSION (3-panel, GnBu) ----------------
def load_cm(path):
    rows=list(csv.reader(io.open(path,encoding='utf-8')))
    # first row header (class names or blank+names), first col labels
    body=rows[1:]
    M=np.array([[float(x) for x in r[1:1+4]] for r in body[:4]])
    return M

def classical_cm():
    # re-fit LightGBM on db4 NL16 features (audited split) -> test confusion
    import openpyxl
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    p='/mnt/c/Users/cbot/Desktop/BRISC pos graduação/ML classico/features_BRISC_db4_preprocessed.xlsx'
    wb=openpyxl.load_workbook(p,read_only=True,data_only=True); ws=wb['NL16']
    it=ws.iter_rows(values_only=True); hdr=list(next(it)); ci={n:i for i,n in enumerate(hdr)}
    feat=[n for n in hdr if n not in ('filename','split','class','view','index')]
    ci2={c:i for i,c in enumerate(CLASSES)}
    Xtr=[];ytr=[];Xte=[];yte=[]
    def sf(v):
        try:return float(v)
        except:return 0.0
    for r in it:
        if r[ci['filename']] is None: continue
        x=[sf(r[ci[f]]) for f in feat]; lab=ci2[r[ci['class']]]
        if r[ci['split']]=='train': Xtr.append(x);ytr.append(lab)
        else: Xte.append(x);yte.append(lab)
    wb.close()
    sc=StandardScaler().fit(Xtr)
    clf=LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=63,random_state=42,verbose=-1)
    clf.fit(sc.transform(Xtr),ytr)
    yp=clf.predict(sc.transform(Xte))
    return confusion_matrix(yte,yp,labels=range(4)).astype(float), (np.array(yp)==np.array(yte)).mean()*100

def fig_confusion():
    cmC,accC=classical_cm()
    cmT=load_cm(f'{B}/leakage_backbones/W7_VGG16_ImageNet/audited/confusion_matrix.csv')
    cmH=load_cm(f'{B}/w11_hybrid/W7_VGG16_ImageNet/confusion_matrix_best.csv')
    accT=np.trace(cmT)/cmT.sum()*100; accH=np.trace(cmH)/cmH.sum()*100
    panels=[('Classical (db4+GLCM+LGBM)',cmC,accC),
            ('Transfer (VGG16 softmax)',cmT,accT),
            ('Hybrid (VGG16+LightGBM)',cmH,accH)]
    # Custom colormap from user's palette (rose highlight -> teal -> deep teal)
    from matplotlib.colors import LinearSegmentedColormap
    cmap_pal = LinearSegmentedColormap.from_list('brisc_teal',
        ['#F5F9F8', PAL_TEAL, PAL_TEAL_D])
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.7))
    for ax,(title,M,acc) in zip(axs,panels):
        Mn=M/M.sum(axis=1,keepdims=True)*100
        im=ax.imshow(Mn,cmap=cmap_pal,vmin=0,vmax=100)
        ax.set_xticks(range(4));ax.set_yticks(range(4))
        ax.set_xticklabels(['G','M','N','P'],fontsize=8,color=PAL_INK)
        ax.set_yticklabels(['G','M','N','P'],fontsize=8,color=PAL_INK)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f'{int(M[i,j])}',ha='center',va='center',fontsize=7.5,
                        color='white' if Mn[i,j]>55 else PAL_INK)
        ax.set_title(f'{title}\n{acc:.1f}%',fontsize=8,color=PAL_INK)
        ax.set_xlabel('Predicted',fontsize=8,color=PAL_INK)
        if ax is axs[0]: ax.set_ylabel('True',fontsize=8,color=PAL_INK)
        ax.grid(False)
    fig.text(0.5,-0.06,'G glioma  M meningioma  N no tumor  P pituitary',ha='center',fontsize=7.5,color=PAL_INK)
    fig.tight_layout()
    save(fig,'fig_confusion')

# ---------------- FIG GRAD-CAM (6x4 grid, overlay-only) ----------------
def _gc_example_path(cls):
    """Same example the worker used: first image alphabetically in test/<cls>/."""
    d=f'{AUD_BASE}/test/{cls}'
    fs=sorted(f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png')))
    return f'{d}/{fs[0]}'

def _gc_overlay(bb,cls):
    """Rebuild the overlay-only image (grayscale MRI + jet Grad-CAM++) at full
    cell resolution from the saved `_raw.npy` cam, matching the worker's blend.
    Falls back to the bare heatmap if the raw MRI is not reachable."""
    from matplotlib import cm
    cam=np.load(f'{B}/w11_hybrid/{bb}/gradcam/{cls}_raw.npy')   # already (sz,sz)
    heat=cm.jet(cam)[...,:3]
    try:
        sz=GC_IMG_SIZE[bb]
        gray=np.asarray(Image.open(_gc_example_path(cls)).convert('L').resize((sz,sz)),float)/255.0
        return np.clip(0.55*heat+0.45*np.stack([gray]*3,axis=-1),0,1)
    except Exception:
        return heat

def fig_gradcam():
    # Rows: CNN-from-scratch, the five uniform-protocol backbones, then the
    # headline optimised EfficientNetB1 (98.82% audited). Any row whose heatmaps
    # are not on disk is skipped (with a warning) so the figure still builds.
    ROWS=[('W3_CNN_scratch','Custom CNN'),
          ('W6_ResNet50_ImageNet','ResNet50'),
          ('W7_VGG16_ImageNet','VGG16'),
          ('W8_EfficientNetB2_ImageNet','EffNetB2'),
          ('W9_ConvNeXtTiny_ImageNet','ConvNeXt'),
          ('W10_RadImageNet_ResNet50','RadImageNet'),
          ('W15_EffNetB1_optimised','EffNetB1 opt.')]
    present=[(b,l) for b,l in ROWS
             if os.path.exists(f'{B}/w11_hybrid/{b}/gradcam/{CLASSES[0]}_raw.npy')]
    for b,l in ROWS:
        if (b,l) not in present:
            print(f'  [warn] fig_gradcam: heatmaps for {b} missing -- run '
                  f'w11_gradcam.py {b} <classification_task> <w11_hybrid> to include it.')
    bbs =[b for b,_ in present]
    blab=[l for _,l in present]
    n=len(bbs)
    # Uniform grid: same treatment as fig_gradcam_compact -- gridspec with
    # equal cell sizes + aspect='auto' on imshow so every panel has the same
    # visible size regardless of the native pixel resolution of the backbone.
    fig,axs=plt.subplots(n,4,figsize=(10.5,2.35*n),
                         gridspec_kw={'wspace':0.02,'hspace':0.05})
    axs=np.atleast_2d(axs)
    for i,bb in enumerate(bbs):
        for j,cl in enumerate(CLASSES):
            ax=axs[i,j]
            if os.path.exists(f'{B}/w11_hybrid/{bb}/gradcam/{cl}_raw.npy'):
                ax.imshow(_gc_overlay(bb,cl),aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if i==0: ax.set_title(CLABEL[j],fontsize=11,fontweight='bold',color=TEAL_D)
            if j==0: ax.text(-0.05,0.5,blab[i],transform=ax.transAxes,ha='right',va='center',
                             fontsize=10.5,fontweight='bold',color=TEAL_D,rotation=90)
    # Reserve a FIXED-INCH band for the suptitle (not a fraction of height):
    # a fractional rect (e.g. top=0.98) leaves a huge gap on tall multi-row
    # figures, since 2% of 16in is much more white space than 2% of 9in.
    # Reserve a FIXED-INCH band at the top (not a fraction of height): a
    # fractional margin leaves a huge gap on tall multi-row figures. The band
    # must fit BOTH the suptitle and the column-header row (drawn just above
    # the grid via ax.set_title), so it needs to be generous, not minimal.
    fig.tight_layout()
    fig_h = fig.get_size_inches()[1]
    top_margin_in = 0.55
    fig.subplots_adjust(top=1 - top_margin_in/fig_h, left=0.07)
    fig.suptitle('Grad-CAM++ across pretrainings (same four test images)',
                 fontsize=12,fontweight='bold',color=TEAL_D,y=1 - 0.15/fig_h)
    save(fig,'fig_gradcam')


def fig_gradcam_compact():
    """Compact 4-row Grad-CAM++ for the main article: only the four models that
    carry the study's narrative (custom CNN baseline, best ImageNet transfer,
    RadImageNet contrast, headline optimised EfficientNetB1). The full 7-row
    version stays in fig_gradcam.* for the supplementary."""
    ROWS = [
        ('W3_CNN_scratch',            'Custom CNN'),
        ('W7_VGG16_ImageNet',         'VGG16 (ImageNet)'),
        ('W10_RadImageNet_ResNet50',  'RadImageNet'),
        ('W15_EffNetB1_optimised',    'EffNetB1 opt.'),
    ]
    present = [(b, l) for b, l in ROWS
               if os.path.exists(f'{B}/w11_hybrid/{b}/gradcam/{CLASSES[0]}_raw.npy')]
    if len(present) < len(ROWS):
        missing = [b for b, _ in ROWS if (b, _) not in present]
        print(f'  [warn] fig_gradcam_compact: missing heatmaps for {missing}')
    bbs = [b for b, _ in present]
    blab = [l for _, l in present]
    n = len(bbs)
    # Uniform grid: fixed figsize with equal cells + aspect='auto' on imshow so
    # every panel is the same visible size regardless of the underlying image
    # native pixel resolution (128 vs 224 vs 240 px).
    fig, axs = plt.subplots(n, 4, figsize=(10.5, 2.35 * n),
                            gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    axs = np.atleast_2d(axs)
    for i, bb in enumerate(bbs):
        for j, cl in enumerate(CLASSES):
            ax = axs[i, j]
            if os.path.exists(f'{B}/w11_hybrid/{bb}/gradcam/{cl}_raw.npy'):
                ax.imshow(_gc_overlay(bb, cl), aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(CLABEL[j], fontsize=11, fontweight='bold', color=TEAL_D)
            if j == 0:
                ax.text(-0.05, 0.5, blab[i], transform=ax.transAxes,
                        ha='right', va='center', fontsize=10.5,
                        fontweight='bold', color=TEAL_D, rotation=90)
    fig.tight_layout()
    fig_h = fig.get_size_inches()[1]
    top_margin_in = 0.55
    fig.subplots_adjust(top=1 - top_margin_in/fig_h, left=0.07)
    fig.suptitle('Grad-CAM++ (four representative models on the same test images)',
                 fontsize=12, fontweight='bold', color=TEAL_D, y=1 - 0.15/fig_h)
    save(fig, 'fig_gradcam_compact')


# fig_workflows is now the user's hand-drawn PDF (kept only in fig_workflows.pdf);
# the Python version is retained above as a fallback but is NOT auto-run so that
# invoking the script never overwrites the authored PDF. To regenerate the Python
# fallback explicitly:  python -c "from build_paper_figures import fig_workflows; fig_workflows()"
for fn in (fig_confusion, fig_gradcam, fig_gradcam_compact):
    try: fn()
    except Exception as e:
        import traceback; print('ERR',fn.__name__); traceback.print_exc()
print('done ->',LFG)
