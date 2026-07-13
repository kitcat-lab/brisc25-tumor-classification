# -*- coding: utf-8 -*-
"""Generate ALL article + supplementary figures in one run.

Figures produced (Springer Nature NPG palette, colorblind-friendly):
  Article:
    fig_workflows           study-design schematic
    fig_confusion           3-panel confusion matrices (Classical / Transfer / Hybrid)
    fig_gradcam_compact     4-row Grad-CAM++ (CNN, VGG16, RadImageNet, EffNetB1 opt.)
  Supplementary:
    fig_gradcam             7-row Grad-CAM++ (all backbones)
    fig_training_history    learning curves: accuracy (individual + overlay)
    fig_training_loss       learning curves: loss (individual + overlay)

Output: latex_artigo/figures/ as PNG + PDF + SVG.

Runtime deps: numpy, matplotlib, Pillow, openpyxl, lightgbm, scikit-learn.
              (if openpyxl/lightgbm are missing, fig_confusion is skipped)
Runs from WSL or Windows -- paths derived from __file__.
"""
import os, io, csv, json
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image



# ══════════════════════════════════════════════════════════════════════════════
# PATHS (portable: derived from this file's location)
# ══════════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../brisc_gui
_ROOT = os.path.dirname(_HERE)                              # project root
B     = _HERE
LFG   = os.path.join(_ROOT, 'latex_artigo', 'figures')
os.makedirs(LFG, exist_ok=True)

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLABEL  = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

_AUD_CANDIDATES = [
    '/root/brisc/data/brisc2025_clean/classification_task',
    os.path.join(_ROOT, 'classification_task'),
    os.path.join(_ROOT, 'brisc2025_clean', 'classification_task'),
]
AUD_BASE = next((p for p in _AUD_CANDIDATES
                 if os.path.isdir(os.path.join(p, 'test'))), _AUD_CANDIDATES[0])

GC_IMG_SIZE = {
    'W3_CNN_scratch': 128, 'W6_ResNet50_ImageNet': 224,
    'W7_VGG16_ImageNet': 224, 'W8_EfficientNetB2_ImageNet': 260,
    'W9_ConvNeXtTiny_ImageNet': 224, 'W10_RadImageNet_ResNet50': 224,
    'W15_EffNetB1_optimised': 240,
}

# ══════════════════════════════════════════════════════════════════════════════
# SPRINGER NATURE NPG PALETTE (colorblind-friendly, no dark blue)
# ══════════════════════════════════════════════════════════════════════════════
SN_RED     = '#E64B35'
SN_TEAL    = '#00A087'
SN_ORANGE  = '#F39B7F'
SN_PURPLE  = '#8491B4'
SN_CYAN    = '#4DBBD5'
SN_MINT    = '#91D1C2'
SN_BROWN   = '#7E6148'
SN_WINE    = '#B09C85'
SN_DRED    = '#DC0000'
SN_MAGENTA = '#CC79A7'
SN_INK     = '#2E3149'

# Individual panel curves
CLR_TRAIN  = SN_RED
CLR_VAL    = SN_CYAN
CLR_MARK   = SN_MAGENTA       # head -> fine-tune dotted line

# 11-color overlay cycle (all distinct, no dark blue)
CYC = [SN_TEAL, SN_RED, SN_CYAN, SN_ORANGE, SN_PURPLE,
       SN_BROWN, SN_MAGENTA, SN_DRED, SN_MINT, SN_WINE, '#3D7068']
CYC_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', 'd', 'p']
CYC_LS      = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-']

mpl.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def save(fig, name):
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{LFG}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('  saved', name)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — WORKFLOWS (study-design schematic)
# ══════════════════════════════════════════════════════════════════════════════
def fig_workflows():
    P_DATA = SN_ORANGE;  P_APPR = SN_MINT
    P_HEAD = SN_CYAN;    P_MET  = SN_PURPLE
    EDGE   = SN_INK;     ARROW  = '#6A7180'

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.axis('off')

    def box(x, y, w, h, t, c, tc=EDGE, fs=9.5, weight='bold'):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.18',
            fc=c, ec=EDGE, lw=0.9))
        ax.text(x + w/2, y + h/2, t, ha='center', va='center',
                color=tc, fontsize=fs, fontweight=weight)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>',
            mutation_scale=14, lw=1.2, color=ARROW))

    box(6.5, 9.4, 7.0, 1.1,
        'BRISC 2025 dataset — 6,000 MRI images', P_DATA, fs=11)
    box(6.5, 8.0, 7.0, 1.0,
        'Forensic audit (MD5 + pHash): 6,000 $\\to$ 5,041 images', P_DATA, fs=10.5)
    box(6.5, 6.6, 7.0, 1.0,
        'Official split: 4,363 train / 678 test', P_DATA, fs=10.5)
    arr(10.0, 9.4, 10.0, 9.0)
    arr(10.0, 8.0, 10.0, 7.6)

    br_x = [0.4, 5.4, 10.4, 15.4]; br_w = 4.2
    br_titles = [
        'Wavelet + GLCM\n(198 features)',
        'CNN from scratch\n(3 blocks)',
        'Transfer backbones\nResNet50 / VGG16 /\nEffNetB2 / ConvNeXt-Tiny',
        'RadImageNet\n(medical pretrain)',
    ]
    for x, t in zip(br_x, br_titles):
        box(x, 4.4, br_w, 1.7, t, P_APPR, fs=9.5)
        arr(10.0, 6.6, x + br_w/2, 6.1)

    box(0.4, 2.7, br_w, 1.2, '19 classifiers', P_HEAD, fs=10.5)
    box(5.4, 2.7, br_w*3 + 2.0, 1.2,
        'Softmax head  OR  hybrid (7 classifiers on 128-d features)',
        P_HEAD, fs=10.5)
    arr(br_x[0] + br_w/2, 4.4, br_x[0] + br_w/2, 3.9)
    for i in (1, 2, 3):
        arr(br_x[i] + br_w/2, 4.4, br_x[i] + br_w/2, 3.9)

    box(3.0, 0.8, 14.0, 1.2,
        'Acc / F1 / AUC  $\\;+\\;$  95% bootstrap CI  $\\;+\\;$  '
        'paired McNemar  $\\;+\\;$  Grad-CAM++', P_MET, fs=11)
    arr(br_x[0] + br_w/2, 2.7, 10.0, 2.0)
    for i in (1, 2, 3):
        arr(br_x[i] + br_w/2, 2.7, 10.0, 2.0)

    save(fig, 'fig_workflows')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — CONFUSION MATRICES (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
def load_cm(path):
    rows = list(csv.reader(io.open(path, encoding='utf-8')))
    body = rows[1:]
    return np.array([[float(x) for x in r[1:5]] for r in body[:4]])


def classical_cm():
    import openpyxl
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix

    candidates = [
        os.path.join(_ROOT, 'ML classico',
                     'features_BRISC_db4_preprocessed.xlsx'),
        '/mnt/c/Users/cbot/Desktop/BRISC pos graduação'
        '/ML classico/features_BRISC_db4_preprocessed.xlsx',
    ]
    p = next((c for c in candidates if os.path.isfile(c)), candidates[0])
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    ws = wb['NL16']
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    ci = {n: i for i, n in enumerate(hdr)}
    feat = [n for n in hdr
            if n not in ('filename', 'split', 'class', 'view', 'index')]
    ci2 = {c: i for i, c in enumerate(CLASSES)}
    Xtr, ytr, Xte, yte = [], [], [], []

    def sf(v):
        try: return float(v)
        except Exception: return 0.0

    for r in it:
        if r[ci['filename']] is None:
            continue
        x = [sf(r[ci[f]]) for f in feat]
        lab = ci2[r[ci['class']]]
        if r[ci['split']] == 'train':
            Xtr.append(x); ytr.append(lab)
        else:
            Xte.append(x); yte.append(lab)
    wb.close()

    sc = StandardScaler().fit(Xtr)
    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                         num_leaves=63, random_state=42, verbose=-1)
    clf.fit(sc.transform(Xtr), ytr)
    yp = clf.predict(sc.transform(Xte))
    acc = (np.array(yp) == np.array(yte)).mean() * 100
    return confusion_matrix(yte, yp, labels=range(4)).astype(float), acc


def fig_confusion():
    cmC, accC = classical_cm()
    cmT = load_cm(f'{B}/leakage_backbones/W7_VGG16_ImageNet/audited/'
                  'confusion_matrix.csv')
    cmH = load_cm(f'{B}/w11_hybrid/W7_VGG16_ImageNet/'
                  'confusion_matrix_best.csv')
    accT = np.trace(cmT) / cmT.sum() * 100
    accH = np.trace(cmH) / cmH.sum() * 100

    panels = [('Classical (db4+GLCM+LGBM)', cmC, accC),
              ('Transfer (VGG16 softmax)',   cmT, accT),
              ('Hybrid (VGG16+LightGBM)',    cmH, accH)]

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'brisc_sn', ['#F5F9F8', SN_MINT, SN_TEAL])

    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.7))
    for ax, (title, M, acc) in zip(axs, panels):
        Mn = M / M.sum(axis=1, keepdims=True) * 100
        ax.imshow(Mn, cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(['G', 'M', 'N', 'P'], fontsize=8, color=SN_INK)
        ax.set_yticklabels(['G', 'M', 'N', 'P'], fontsize=8, color=SN_INK)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{int(M[i,j])}', ha='center', va='center',
                        fontsize=7.5,
                        color='white' if Mn[i, j] > 55 else SN_INK)
        ax.set_title(f'{title}\n{acc:.1f}%', fontsize=8, color=SN_INK)
        ax.set_xlabel('Predicted', fontsize=8, color=SN_INK)
        if ax is axs[0]:
            ax.set_ylabel('True', fontsize=8, color=SN_INK)
        ax.grid(False)
    fig.text(0.5, -0.06,
             'G glioma  M meningioma  N no tumor  P pituitary',
             ha='center', fontsize=7.5, color=SN_INK)
    fig.tight_layout()
    save(fig, 'fig_confusion')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — GRAD-CAM++ (7-row supplementary + 4-row compact article)
# ══════════════════════════════════════════════════════════════════════════════
def _gc_example_path(cls):
    d = f'{AUD_BASE}/test/{cls}'
    fs = sorted(f for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return f'{d}/{fs[0]}'


def _gc_overlay(bb, cls):
    from matplotlib import cm
    cam = np.load(f'{B}/w11_hybrid/{bb}/gradcam/{cls}_raw.npy')
    heat = cm.jet(cam)[..., :3]
    try:
        sz = GC_IMG_SIZE[bb]
        gray = np.asarray(
            Image.open(_gc_example_path(cls)).convert('L').resize((sz, sz)),
            float) / 255.0
        return np.clip(0.55 * heat + 0.45 * np.stack([gray]*3, axis=-1), 0, 1)
    except Exception:
        return heat


def _gradcam_grid(rows_spec, fname, suptitle):
    present = [(b, l) for b, l in rows_spec
               if os.path.exists(
                   f'{B}/w11_hybrid/{b}/gradcam/{CLASSES[0]}_raw.npy')]
    for b, l in rows_spec:
        if (b, l) not in present:
            print(f'  [warn] {fname}: heatmaps for {b} missing')
    bbs  = [b for b, _ in present]
    blab = [l for _, l in present]
    n = len(bbs)
    if n == 0:
        print(f'  [skip] {fname}: no heatmaps found'); return

    fig, axs = plt.subplots(n, 4, figsize=(10.5, 2.35 * n),
                            gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    axs = np.atleast_2d(axs)
    for i, bb in enumerate(bbs):
        for j, cl in enumerate(CLASSES):
            ax = axs[i, j]
            npy = f'{B}/w11_hybrid/{bb}/gradcam/{cl}_raw.npy'
            if os.path.exists(npy):
                ax.imshow(_gc_overlay(bb, cl), aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(CLABEL[j], fontsize=11, fontweight='bold',
                             color=SN_TEAL)
            if j == 0:
                ax.text(-0.05, 0.5, blab[i], transform=ax.transAxes,
                        ha='right', va='center', fontsize=10.5,
                        fontweight='bold', color=SN_TEAL, rotation=90)
    fig.tight_layout()
    fig_h = fig.get_size_inches()[1]
    fig.subplots_adjust(top=1 - 0.55/fig_h, left=0.07)
    fig.suptitle(suptitle, fontsize=12, fontweight='bold',
                 color=SN_TEAL, y=1 - 0.15/fig_h)
    save(fig, fname)


def fig_gradcam():
    _gradcam_grid([
        ('W3_CNN_scratch',           'Custom CNN'),
        ('W6_ResNet50_ImageNet',     'ResNet50'),
        ('W7_VGG16_ImageNet',        'VGG16'),
        ('W8_EfficientNetB2_ImageNet','EffNetB2'),
        ('W9_ConvNeXtTiny_ImageNet', 'ConvNeXt'),
        ('W10_RadImageNet_ResNet50', 'RadImageNet'),
        ('W15_EffNetB1_optimised',   'EffNetB1 opt.'),
    ], 'fig_gradcam',
       'Grad-CAM++ across pretrainings (same four test images)')


def fig_gradcam_compact():
    _gradcam_grid([
        ('W3_CNN_scratch',           'Custom CNN'),
        ('W7_VGG16_ImageNet',        'VGG16 (ImageNet)'),
        ('W10_RadImageNet_ResNet50', 'RadImageNet'),
        ('W15_EffNetB1_optimised',   'EffNetB1 opt.'),
    ], 'fig_gradcam_compact',
       'Grad-CAM++ (four representative models on the same test images)')


# ══════════════════════════════════════════════════════════════════════════════
# FIG S2/S3 — TRAINING HISTORY (accuracy + loss)
# ══════════════════════════════════════════════════════════════════════════════
HISTORY_MODELS = [
    ('Custom CNN',
     f'{B}/leakage_v3/audited/history.json',
     f'{B}/leakage_v3/audited/metrics.json'),
    ('ResNet50 (ImageNet)',
     f'{B}/leakage_backbones/W6_ResNet50_ImageNet/audited/history.json',
     f'{B}/leakage_backbones/W6_ResNet50_ImageNet/audited/metrics.json'),
    ('VGG16 (ImageNet)',
     f'{B}/leakage_backbones/W7_VGG16_ImageNet/audited/history.json',
     f'{B}/leakage_backbones/W7_VGG16_ImageNet/audited/metrics.json'),
    ('EfficientNetB0 (ImageNet)',
     f'{B}/leakage_backbones/W14_EfficientNetB0_ImageNet/audited/history.json',
     f'{B}/leakage_backbones/W14_EfficientNetB0_ImageNet/audited/metrics.json'),
    ('EfficientNetB2 (ImageNet)',
     f'{B}/leakage_backbones/W8_EfficientNetB2_ImageNet/audited/history.json',
     f'{B}/leakage_backbones/W8_EfficientNetB2_ImageNet/audited/metrics.json'),
    ('ConvNeXt-Tiny (ImageNet)',
     f'{B}/leakage_backbones/W9_ConvNeXtTiny_ImageNet/audited/history.json',
     f'{B}/leakage_backbones/W9_ConvNeXtTiny_ImageNet/audited/metrics.json'),
    ('ResNet50 (RadImageNet)',
     f'{B}/leakage_backbones/W10_RadImageNet_ResNet50/audited/history.json',
     f'{B}/leakage_backbones/W10_RadImageNet_ResNet50/audited/metrics.json'),
    ('EfficientNetB0 (opt.)',
     f'{B}/efficientnet_optimized/B0/audited/history.json',
     f'{B}/efficientnet_optimized/B0/audited/metrics.json'),
    ('EfficientNetB1 (opt.)',
     f'{B}/efficientnet_optimized/B1/audited/history.json',
     f'{B}/efficientnet_optimized/B1/audited/metrics.json'),
    ('EfficientNetB2 (opt.)',
     f'{B}/efficientnet_optimized/B2/audited/history.json',
     f'{B}/efficientnet_optimized/B2/audited/metrics.json'),
    ('Custom CNN (opt.)',
     f'{B}/scratch_cnn_optimized/audited/history.json',
     f'{B}/scratch_cnn_optimized/audited/metrics.json'),
]


def _load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _boundary(mpath):
    m = _load_json(mpath)
    return m.get('epochs_head') if m else None


def _training_figure(metric, val_metric, ylabel, ylim, fname, suptitle):
    present = [(t, h, m) for t, h, m in HISTORY_MODELS
               if (_load_json(h) or {}).get(metric)]
    if not present:
        print(f'  [skip] {fname}: no history files found'); return

    cols = 4
    rows = -(-(len(present) + 1) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows))
    axs = axs.ravel()
    for k in range(len(present) + 1, len(axs)):
        axs[k].set_visible(False)

    overlay = axs[len(present)]
    for i, (title, hpath, mpath) in enumerate(present):
        ax = axs[i]
        h = _load_json(hpath)
        tr = [v * 100 if metric == 'accuracy' else v for v in h[metric]]
        va = [v * 100 if metric == 'accuracy' else v
              for v in h.get(val_metric, [])]
        ep = range(1, len(tr) + 1)
        ax.plot(ep, tr, color=CLR_TRAIN, lw=1.6, label='train')
        if va:
            ax.plot(ep, va, color=CLR_VAL, lw=1.6, ls='--', label='val')
        b = _boundary(mpath)
        if b and 0 < b < len(tr):
            ax.axvline(b + 0.5, color=CLR_MARK, lw=0.9, ls=':', alpha=0.8)
        ax.set_title(title, fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.2, lw=0.5)
        if i % cols == 0:
            ax.set_ylabel(ylabel)
        if va:
            lbl = ('RadImageNet' if 'RadImageNet' in title
                   else title.split(' (')[0]
                        + (' opt' if 'opt.' in title else ''))
            mk = CYC_MARKERS[i % len(CYC_MARKERS)]
            ls = CYC_LS[i % len(CYC_LS)]
            overlay.plot(ep, va, color=CYC[i % len(CYC)], lw=1.4,
                         ls=ls, marker=mk, markersize=3.5,
                         markevery=max(1, len(va) // 8), label=lbl)

    axs[0].legend(loc='center right', frameon=False, fontsize=6.5)
    overlay.set_title('All models (validation)', fontsize=8)
    overlay.set_xlabel('Epoch')
    overlay.grid(alpha=0.2, lw=0.5)
    if ylim:
        overlay.set_ylim(*ylim)
    overlay.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                   frameon=False, ncol=1, fontsize=6.8,
                   handlelength=1.6, handletextpad=0.4, borderaxespad=0)
    fig.suptitle(suptitle, fontsize=10, fontweight='bold',
                 color=SN_INK, y=1.005)
    fig.text(0.5, -0.02, 'dotted line = head→fine-tune switch',
             ha='center', fontsize=7, color=CLR_MARK)
    fig.tight_layout(rect=(0, 0, 0.965, 1))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{LFG}/{fname}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('  saved', fname)


def fig_training_history():
    _training_figure('accuracy', 'val_accuracy', 'Accuracy (%)', (40, 101),
                     'fig_training_history',
                     'Learning curves: train vs validation accuracy '
                     '(audited partition)')


def fig_training_loss():
    _training_figure('loss', 'val_loss', 'Loss', None,
                     'fig_training_loss',
                     'Learning curves: train vs validation loss '
                     '(audited partition)')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run everything, skip gracefully on missing deps
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import traceback
    ALL = [
        ('fig_workflows',        fig_workflows),
        ('fig_confusion',        fig_confusion),
        ('fig_gradcam',          fig_gradcam),
        ('fig_gradcam_compact',  fig_gradcam_compact),
        ('fig_training_history', fig_training_history),
        ('fig_training_loss',    fig_training_loss),
    ]
    ok, skip = 0, 0
    for name, fn in ALL:
        try:
            print(f'[{name}]')
            fn()
            ok += 1
        except Exception:
            skip += 1
            print(f'  ERR {name}:')
            traceback.print_exc()
    print(f'\ndone: {ok} generated, {skip} skipped -> {LFG}')
