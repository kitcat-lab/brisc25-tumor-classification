# -*- coding: utf-8 -*-
"""Supplementary figures: learning curves (train vs validation accuracy and loss)
for every deep model that logged per-epoch history. Teal theme; PNG+PDF+SVG.

Only epoch-trained models have history: the scratch CNN and the six softmax
transfer/medical backbones (audited partition). Hybrids and classical ML have no
epoch history, and the OPTIMISED EfficientNet worker did not log history, so those
are omitted (no fabricated curves). A faint line marks the head->fine-tune switch.
"""
import json, os
import matplotlib as mpl, matplotlib.pyplot as plt

B = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'
OUTS = [f'{B}/figuras_artigo',
        '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/latex_artigo/figures']
for o in OUTS:
    os.makedirs(o, exist_ok=True)

# --- Palette matches fig_workflows (user's rose/sand/teal/plum scheme) --------
PAL_ROSE_D = '#B45E70'      # train curve (rose)
PAL_TEAL_D = '#2E7A75'      # val curve  (teal)
PAL_INK    = '#2E3149'      # slate ink for titles / labels
PAL_MARK   = '#B08B2E'      # head->fine-tune boundary marker (sand accent)
TEAL_D = PAL_INK            # legacy alias kept for the suptitle colour
TEAL_M = PAL_TEAL_D
TEAL_L = PAL_TEAL_D
# Qualitative palette for the "all models" overlay: cycle rose/teal/sand/plum
# variants so no two adjacent models share a colour.
CYC = ['#2E7A75', '#B45E70', '#6E4C8E', '#B08B2E', '#7A9CC4', '#8E5572',
       '#3D7068', '#9A6A3A', '#5B6E9F', '#C48DA5']
mpl.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

# (title, history.json, metrics.json for the head->ft boundary)
MODELS = [
    ('Custom CNN',             f'{B}/leakage_v3/audited/history.json',
                               f'{B}/leakage_v3/audited/metrics.json'),
    ('ResNet50 (ImageNet)',    f'{B}/leakage_backbones/W6_ResNet50_ImageNet/audited/history.json',
                               f'{B}/leakage_backbones/W6_ResNet50_ImageNet/audited/metrics.json'),
    ('VGG16 (ImageNet)',       f'{B}/leakage_backbones/W7_VGG16_ImageNet/audited/history.json',
                               f'{B}/leakage_backbones/W7_VGG16_ImageNet/audited/metrics.json'),
    ('EfficientNetB0 (ImageNet)', f'{B}/leakage_backbones/W14_EfficientNetB0_ImageNet/audited/history.json',
                               f'{B}/leakage_backbones/W14_EfficientNetB0_ImageNet/audited/metrics.json'),
    ('EfficientNetB2 (ImageNet)', f'{B}/leakage_backbones/W8_EfficientNetB2_ImageNet/audited/history.json',
                               f'{B}/leakage_backbones/W8_EfficientNetB2_ImageNet/audited/metrics.json'),
    ('ConvNeXt-Tiny (ImageNet)', f'{B}/leakage_backbones/W9_ConvNeXtTiny_ImageNet/audited/history.json',
                               f'{B}/leakage_backbones/W9_ConvNeXtTiny_ImageNet/audited/metrics.json'),
    ('ResNet50 (RadImageNet)', f'{B}/leakage_backbones/W10_RadImageNet_ResNet50/audited/history.json',
                               f'{B}/leakage_backbones/W10_RadImageNet_ResNet50/audited/metrics.json'),
    # optimised EfficientNet (appear once the worker is re-run with history logging)
    ('EfficientNetB0 (opt.)',  f'{B}/efficientnet_optimized/B0/audited/history.json',
                               f'{B}/efficientnet_optimized/B0/audited/metrics.json'),
    ('EfficientNetB1 (opt.)',  f'{B}/efficientnet_optimized/B1/audited/history.json',
                               f'{B}/efficientnet_optimized/B1/audited/metrics.json'),
    ('EfficientNetB2 (opt.)',  f'{B}/efficientnet_optimized/B2/audited/history.json',
                               f'{B}/efficientnet_optimized/B2/audited/metrics.json'),
    ('Custom CNN (opt.)',      f'{B}/scratch_cnn_optimized/audited/history.json',
                               f'{B}/scratch_cnn_optimized/audited/metrics.json'),
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def boundary(mpath):
    m = load(mpath)
    return m.get('epochs_head') if m else None


def make_figure(metric, val_metric, ylabel, ylim, fname, suptitle):
    # only models that actually logged this metric; grid adapts to the count
    present = [(t, h, m) for (t, h, m) in MODELS
               if (load(h) or {}).get(metric)]
    cols = 4
    rows = -(-(len(present) + 1) // cols)          # +1 for the overlay panel
    fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows))
    axs = axs.ravel()
    for k in range(len(present) + 1, len(axs)):
        axs[k].set_visible(False)
    overlay = axs[len(present)]
    for i, (title, hpath, mpath) in enumerate(present):
        ax = axs[i]
        h = load(hpath)
        tr = [v * 100 if metric == 'accuracy' else v for v in h[metric]]
        va = [v * 100 if metric == 'accuracy' else v for v in h.get(val_metric, [])]
        ep = range(1, len(tr) + 1)
        ax.plot(ep, tr, color=PAL_ROSE_D, lw=1.6, label='train')
        if va:
            ax.plot(ep, va, color=PAL_TEAL_D, lw=1.6, ls='--', label='val')
        b = boundary(mpath)
        if b and 0 < b < len(tr):
            ax.axvline(b + 0.5, color=PAL_MARK, lw=0.9, ls=':', alpha=0.8)
        ax.set_title(title, fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.2, lw=0.5)
        if i % 4 == 0:
            ax.set_ylabel(ylabel)
        # overlay validation curves for all models (distinct short labels)
        if va:
            lbl = ('RadImageNet' if 'RadImageNet' in title
                   else title.split(' (')[0] + (' opt' if 'opt.' in title else ''))
            overlay.plot(ep, va, color=CYC[i % len(CYC)], lw=1.4, label=lbl)
    axs[0].legend(loc='lower right', frameon=False)
    overlay.set_title('All models (validation)', fontsize=8)
    overlay.set_xlabel('Epoch'); overlay.grid(alpha=0.2, lw=0.5)
    if ylim:
        overlay.set_ylim(*ylim)
    # Legend outside the overlay axes (to the right); avoids covering the curves
    # even with 10+ models. `bbox_to_anchor` anchors it just past the right spine;
    # `fig.tight_layout` then reserves space for it so nothing is clipped.
    overlay.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                   frameon=False, ncol=1, fontsize=6.8,
                   handlelength=1.6, handletextpad=0.4, borderaxespad=0)
    fig.suptitle(suptitle, fontsize=10, fontweight='bold', color=TEAL_D, y=1.005)
    fig.text(0.5, -0.02, 'dotted line = head→fine-tune switch',
             ha='center', fontsize=7, color=PAL_MARK)
    # Reserve room on the right for the external overlay legend before tight_layout.
    fig.tight_layout(rect=(0, 0, 0.965, 1))
    for o in OUTS:
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(f'{o}/{fname}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


make_figure('accuracy', 'val_accuracy', 'Accuracy (%)', (40, 101),
            'fig_training_history',
            'Learning curves: train vs validation accuracy (audited partition)')
make_figure('loss', 'val_loss', 'Loss', None,
            'fig_training_loss',
            'Learning curves: train vs validation loss (audited partition)')
print('done -> figures in figuras_artigo/ and latex_artigo/figures/')
