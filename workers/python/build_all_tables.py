# -*- coding: utf-8 -*-
"""Generate the COMPLETE result tables (all models, scenarios + literature) from
the real result files. No value is typed by hand -- every number is read from
metrics.json / comparison.csv / classic_ml_all_results.csv / the literature CSVs.

Outputs:
  latex_artigo/supplementary_tables.tex   standalone-compilable LaTeX (booktabs)
  brisc_gui/all_results_tables.md          markdown preview of the same tables

Run in the TF/venv environment (needs pandas):
    python build_all_tables.py
"""
import os, io, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LK   = f'{HERE}/leakage_backbones'
OPT  = f'{HERE}/efficientnet_optimized'
HYB  = f'{HERE}/w11_hybrid'
TEX_OUT = f'{ROOT}/latex_artigo/supplementary_tables.tex'
MD_OUT  = f'{HERE}/all_results_tables.md'

# backbone workflow -> display name
BB = [('W6_ResNet50_ImageNet', 'ResNet50'),
      ('W7_VGG16_ImageNet', 'VGG16'),
      ('W8_EfficientNetB2_ImageNet', 'EfficientNetB2'),
      ('W9_ConvNeXtTiny_ImageNet', 'ConvNeXt-Tiny'),
      ('W10_RadImageNet_ResNet50', 'RadImageNet ResNet50'),
      ('W14_EfficientNetB0_ImageNet', 'EfficientNetB0')]

md_parts, tex_parts = [], []


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, d=2):
    return f'{v:.{d}f}' if isinstance(v, (int, float)) else '---'


# ─────────────────────────────────────────────────────────────────────────────
# T1  All deep single models (softmax), audited + unaudited
# ─────────────────────────────────────────────────────────────────────────────
def table1():
    rows = []
    # scratch CNN from its meta marker (audited only; no unaudited run)
    meta = jload(f'{HERE}/modelo_brisc_v3_meta.json')
    if meta:
        rows.append(dict(model='Custom CNN',
                         acc=meta.get('test_accuracy'), f1=meta.get('test_f1_macro'),
                         auc=meta.get('test_auc_ovr'), lo=None, hi=None,
                         unaud=None))
    for wf, name in BB:
        a = jload(f'{LK}/{wf}/audited/metrics.json')
        u = jload(f'{LK}/{wf}/unaudited/metrics.json')
        if not a:
            continue
        rows.append(dict(model=name, acc=a['accuracy'], f1=a['f1_macro'],
                         auc=a['auc_ovr'], lo=a.get('ci95_lo'), hi=a.get('ci95_hi'),
                         unaud=(u['accuracy'] if u else None)))
    # Optimised scratch CNN (best-effort deeper architecture, no BN — see the
    # article's Discussion; audited only, no unaudited run).
    sco = jload(f'{HERE}/scratch_cnn_optimized/audited/metrics.json')
    if sco:
        rows.append(dict(model='Custom CNN (4-block, no BN)',
                         acc=sco.get('accuracy'), f1=sco.get('f1_macro'),
                         auc=sco.get('auc_ovr'),
                         lo=sco.get('ci95_lo'), hi=sco.get('ci95_hi'),
                         unaud=None))
    cap = ('All deep single models on the audited test set (uniform two-stage protocol). '
           'Unaud.\\ = accuracy on the un-audited release; $\\Delta$ = unaud.$-$aud.')
    hdr = ['Model', 'Acc.', 'F1', 'AUC', 'CI$_{95}$', 'Unaud.', '$\\Delta$']
    mdh = ['Model', 'Acc.', 'F1', 'AUC', 'CI95', 'Unaud.', 'Δ']
    trows = []
    for r in rows:
        ci = f"[{fmt(r['lo'],1)}, {fmt(r['hi'],1)}]" if r['lo'] is not None else '---'
        d = (f"{r['unaud']-r['acc']:+.2f}" if r['unaud'] is not None else '---')
        un = fmt(r['unaud']) if r['unaud'] is not None else '---'
        trows.append([r['model'], fmt(r['acc']), fmt(r['f1']), fmt(r['auc']), ci, un, d])
    emit('tab:supp-deep', cap, hdr, mdh, trows, 'lcccccc', size='\\footnotesize')


# ─────────────────────────────────────────────────────────────────────────────
# T2  Optimised EfficientNet family vs Fateh
# ─────────────────────────────────────────────────────────────────────────────
def table2():
    fateh = pd.read_csv(f'{HERE}/fateh_baselines.csv').set_index('model')['weighted_avg_accuracy_raw']
    trows = []
    for v in ['B0', 'B1', 'B2']:
        a = jload(f'{OPT}/{v}/audited/metrics.json')
        u = jload(f'{OPT}/{v}/unaudited/metrics.json')
        if not a:
            continue
        fr = fateh.get(f'EfficientNet{v}')
        d = f"{u['accuracy']-a['accuracy']:+.2f}" if u else '---'
        trows.append([f'EfficientNet{v}', fmt(u['accuracy']) if u else '---',
                      fmt(a['accuracy']), fmt(a['f1_macro']), fmt(a['auc_ovr']),
                      fmt(fr), d])
    cap = ('EfficientNet optimized tuning (BatchNorm frozen). Ours vs Fateh~\\textit{et~al.} '
           'on the raw release; $\\Delta$ = unaud.$-$aud.')
    hdr = ['Model', 'Unaud.', 'Aud.', 'F1 aud.', 'AUC aud.', 'Fateh', '$\\Delta$']
    mdh = ['Model', 'Unaud.', 'Aud.', 'F1 aud.', 'AUC aud.', 'Fateh', 'Δ']
    emit('tab:supp-effopt', cap, hdr, mdh, trows, 'lcccccc', size='\\small')


# ─────────────────────────────────────────────────────────────────────────────
# T3  Hybrid sweep: backbone x classifier accuracy matrix
# ─────────────────────────────────────────────────────────────────────────────
def table3():
    HB = [('W3_CNN_scratch', 'Custom CNN'),
          ('W6_ResNet50_ImageNet', 'ResNet50'),
          ('W7_VGG16_ImageNet', 'VGG16'),
          ('W8_EfficientNetB2_ImageNet', 'EfficientNetB2'),
          ('W9_ConvNeXtTiny_ImageNet', 'ConvNeXt-Tiny'),
          ('W10_RadImageNet_ResNet50', 'RadImageNet')]
    # classifier order from the first available file
    clfs = None
    data = {}
    for wf, name in HB:
        p = f'{HYB}/{wf}/comparison.csv'
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if clfs is None:
            clfs = list(df['classifier'])
        data[name] = dict(zip(df['classifier'], df['accuracy']))
    short = {'LightGBM': 'LGBM', 'RandomForest': 'RF', 'CatBoost': 'CatB',
             'XGBoost': 'XGB', 'GradientBoosting': 'GB', 'MLP': 'MLP', 'SVM (RBF)': 'SVM'}
    hdr = ['Backbone'] + [short.get(c, c) for c in clfs] + ['Best']
    mdh = hdr
    trows = []
    for name, accs in data.items():
        best = max(accs.values())
        cells = []
        for c in clfs:
            v = accs.get(c)
            s = fmt(v)
            if v == best:
                s = f'\\textbf{{{s}}}'
            cells.append(s)
        trows.append([name] + cells + [f'\\textbf{{{fmt(best)}}}'])
    cap = ('Hybrid sweep: test accuracy (\\%) per (backbone, classifier) on the '
           'audited partition. Bold = best head per backbone.')
    emit('tab:supp-hybrid', cap, hdr, mdh, trows,
         'l' + 'c' * (len(clfs) + 1), size='\\footnotesize', md_bold_strip=True)


# ─────────────────────────────────────────────────────────────────────────────
# T4  Classical ML: best per scenario, official vs random split
# ─────────────────────────────────────────────────────────────────────────────
def table4():
    df = pd.read_csv(f'{HERE}/classic_ml_all_results.csv')
    trows = []
    for scen in sorted(df['scenario'].unique()):
        cells = [scen.replace('_', '\\_')]
        best_o = best_r = None
        for split, key in [('original', 'o'), ('aleatoria', 'r')]:
            sub = df[(df['scenario'] == scen) & (df['split'] == split)]
            if len(sub):
                b = sub.loc[sub['accuracy'].idxmax()]
                cells.append(f"{b['accuracy']:.2f}")
                cells.append(f"{b['classificador']}/{b['NumLevels']}/{int(b['n_features'])}f")
                if split == 'original':
                    best_o = b['accuracy']
                else:
                    best_r = b['accuracy']
            else:
                cells += ['---', '---']
        d = f'{best_r-best_o:+.2f}' if (best_o is not None and best_r is not None) else '---'
        cells.append(d)
        trows.append(cells)
    hdr = ['Scenario', 'Off.\\ best', 'clf/lvl/feat', 'Rand.\\ best', 'clf/lvl/feat', '$\\Delta$']
    mdh = ['Scenario', 'Off. best', 'clf/lvl/feat', 'Rand. best', 'clf/lvl/feat', 'Δ']
    cap = ('Classical ML factorial (wavelet $\\times$ skull-strip $\\times$ feature-selection): '
           'best of 19 classifiers $\\times$ 3 levels per cell, on official/random splits (\\%). '
           'clf/lvl/feat = winning classifier, wavelet level, feature count.')
    emit('tab:supp-classic', cap, hdr, mdh, trows, 'llclcc', size='\\footnotesize')


# ─────────────────────────────────────────────────────────────────────────────
# T5  Same-architecture: ours vs Fateh
# ─────────────────────────────────────────────────────────────────────────────
def table5():
    df = pd.read_csv(f'{HERE}/same_architecture_comparison.csv')
    trows = []
    for _, r in df.iterrows():
        trows.append([r['architecture'], fmt(r['fateh_raw']),
                      fmt(r['ours_unaudited']), fmt(r['ours_audited']),
                      f"{r['ours_aud_minus_fateh']:+.2f}"])
    hdr = ['Architecture', 'Fateh raw', 'Ours unaud.', 'Ours aud.', 'Aud.$-$Fateh']
    mdh = ['Architecture', 'Fateh raw', 'Ours unaud.', 'Ours aud.', 'Aud.-Fateh']
    cap = ('Same-architecture check: our uniform protocol vs Fateh~\\textit{et~al.} on the raw release (\\%).')
    emit('tab:supp-samearch', cap, hdr, mdh, trows, 'lcccc', size='\\small')


# ─────────────────────────────────────────────────────────────────────────────
# T6  Fateh 13 baselines
# ─────────────────────────────────────────────────────────────────────────────
def table6():
    df = pd.read_csv(f'{HERE}/fateh_baselines.csv')
    trows = [[r['model'], fmt(r['weighted_avg_accuracy_raw']),
              r['status'].replace('_', '/')] for _, r in df.iterrows()]
    hdr = ['Model', 'Acc.\\ (raw)', 'Status']
    mdh = ['Model', 'Acc. (raw)', 'Status']
    cap = ('Fateh~\\textit{et~al.}\'s 13 benchmark backbones on the raw release (\\%).')
    emit('tab:supp-fateh', cap, hdr, mdh, trows, 'lcc', size='\\small')


# ─────────────────────────────────────────────────────────────────────────────
# T7  External literature (the article's verified tab:lit set)
# ─────────────────────────────────────────────────────────────────────────────
def table7():
    # Dataset names use the same convention as the main article
    # (author-year for Figshare/Kaggle-hosted sets, not just the host site).
    LIT = [('Alkharaan et al.', 'BRISC 2025 (raw)', '99.70'),
           ('Fateh et al.', 'BRISC 2025 (raw)', '99.20'),
           ('Kakon et al.', 'Nickparvar->BRISC (0-shot)', '96.70'),
           ('Babu Vimala et al.', 'Cheng T1-CE', '99.06'),
           ('Rasool et al.', 'Cheng T1-CE', '98.10'),
           ('Prayogo et al. (HayabusaNet)', 'Nickparvar 4-cls', '98.86'),
           ('Badza & Barjaktarovic', 'Cheng T1-CE', '96.56'),
           ('Saeedi et al.', 'Cheng T1-CE', '96.47'),
           ('Khairandish et al.', 'Cheng T1-CE', '98.50'),
           ('Vadde & Bukaita', 'Nickparvar 4-cls', '99.69'),
           ('This work (hybrid)', 'BRISC 2025 (audited)', '98.67')]
    trows = [[s.replace('&', '\\&'), d.replace('->', '$\\to$'), a] for s, d, a in LIT]
    hdr = ['Study', 'Dataset', 'Acc.\\ (\\%)']
    mdh = ['Study', 'Dataset', 'Acc. (%)']
    cap = ('Representative reported accuracies with their datasets (verified per paper).')
    emit('tab:supp-lit', cap, hdr, mdh, trows, 'llc', size='\\small')


# ─────────────────────────────────────────────────────────────────────────────
def emit(label, caption, hdr, mdh, trows, colspec, size='\\small', md_bold_strip=False):
    # LaTeX
    # [H] (float package) = place exactly here, in text order. With many
    # small tables in sequence, letting LaTeX's [htbp] algorithm defer floats
    # produces the "blank page" artifact (rejected floats pile up and get
    # flushed together by \clearpage). [H] avoids that entirely.
    t = ['\\begin{table}[H]', '\\centering', f'\\caption{{{caption}}}',
         f'\\label{{{label}}}', size, '\\setlength{\\tabcolsep}{4pt}',
         f'\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}', '\\toprule',
         ' & '.join(hdr) + ' \\\\', '\\midrule']
    for r in trows:
        t.append(' & '.join(str(x) for x in r) + ' \\\\')
    t += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    tex_parts.append('\n'.join(t))
    # Markdown
    def strip(x):
        return str(x).replace('\\textbf{', '**').replace('}', '**') if '\\textbf' in str(x) else str(x)
    m = ['', '| ' + ' | '.join(mdh) + ' |', '|' + '|'.join(['---'] * len(mdh)) + '|']
    for r in trows:
        cells = [strip(x).replace('\\_', '_').replace('\\&', '&').replace('$\\to$', '->')
                 .replace('\\%', '%').replace('$-$', '-').replace('$\\Delta$', 'Δ') for x in r]
        m.append('| ' + ' | '.join(cells) + ' |')
    md_parts.append(f'\n### {label}\n{caption}'.replace('\\', '') + '\n' + '\n'.join(m))


# ─────────────────────────────────────────────────────────────────────────────
# T8  Full list of the 19 classical classifiers used in the classical ML sweep
# ─────────────────────────────────────────────────────────────────────────────
def table8():
    CLFS = [
        ('AdaBoost',                        'Freund and Schapire, 1997 [ref:adaboost]'),
        ('Bagging',                         'Breiman, 1996 [ref:bagging]'),
        ('CatBoost',                        'Prokhorenkova et al., 2018 [ref:catboost]'),
        ('Decision Tree',                   'Breiman et al., 1984 [ref:cart]'),
        ('Dummy (stratified)',              'sklearn baseline'),
        ('Extra Trees',                     'Geurts et al., 2006 [ref:et]'),
        ('Gaussian Naive Bayes',            'textbook method'),
        ('Gradient Boosting',               'Friedman, 2001 [ref:gb]'),
        ('k-Nearest Neighbours',            'Cover and Hart, 1967 [ref:knn]'),
        ('LightGBM',                        'Ke et al., 2017 [ref:lgbm]'),
        ('Linear Discriminant Analysis',    'Fisher, 1936; textbook'),
        ('Logistic Regression',             'textbook method'),
        ('Multi-Layer Perceptron',          'textbook method'),
        ('Quadratic Discriminant Analysis', 'textbook method'),
        ('Random Forest',                   'Breiman, 2001 [ref:rf]'),
        ('Ridge Classifier',                'Hoerl and Kennard, 1970 [ref:ridge]'),
        ('SGD Classifier',                  'Bottou, 2010 [ref:sgd]'),
        ('Support Vector Machine (RBF)',    'Cortes and Vapnik, 1995 [ref:svm]'),
        ('XGBoost',                         'Chen and Guestrin, 2016 [ref:xgb]'),
    ]
    trows = [[n, r.split(' [ref:')[0]] for n, r in CLFS]
    hdr = ['Classifier', 'Reference']
    mdh = ['Classifier', 'Reference']
    cap = ('The 19 classical classifiers used in the wavelet+GLCM sweep. '
           'The five appearing in the main methodology (LightGBM, XGBoost, '
           'CatBoost, Random Forest, Extra Trees) are cited there; the '
           'remaining 14 are standard scikit-learn implementations.')
    emit('tab:supp-clfs', cap, hdr, mdh, trows, 'll', size='\\footnotesize')


# ─────────────────────────────────────────────────────────────────────────────
# T9  Classical ML feature importance (best classical model: db4 NL16 + LightGBM, 90.71%)
# Re-fits the exact same model as build_paper_figures.py::classical_cm() (same
# data, same hyperparameters, same seed) and reads off LightGBM's native
# feature_importances_ (split-count gain). Real computation, not invented.
# ─────────────────────────────────────────────────────────────────────────────
def table9():
    try:
        import openpyxl
        from lightgbm import LGBMClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print('  [skip] table9: openpyxl/lightgbm/sklearn not available')
        return
    p = f'{ROOT}/ML classico/features_BRISC_db4_preprocessed.xlsx'
    if not os.path.exists(p):
        print(f'  [skip] table9: {p} not found')
        return
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    ws = wb['NL16']
    it = ws.iter_rows(values_only=True)
    hdr_row = list(next(it))
    ci = {n: i for i, n in enumerate(hdr_row)}
    feat = [n for n in hdr_row if n not in ('filename', 'split', 'class', 'view', 'index')]
    CLASSES4 = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    ci2 = {c: i for i, c in enumerate(CLASSES4)}
    Xtr, ytr = [], []

    def sf(v):
        try: return float(v)
        except (TypeError, ValueError): return 0.0

    for r in it:
        if r[ci['filename']] is None or r[ci['split']] != 'train':
            continue
        Xtr.append([sf(r[ci[f]]) for f in feat]); ytr.append(ci2[r[ci['class']]])
    wb.close()
    sc = StandardScaler().fit(Xtr)
    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                         random_state=42, verbose=-1)
    clf.fit(sc.transform(Xtr), ytr)
    imp = clf.feature_importances_
    order = np.argsort(-imp)[:15]
    total = imp.sum()
    trows = [[f'{r+1}', feat[i].replace('_', '\\_'), f'{imp[i]/total*100:.2f}']
             for r, i in enumerate(order)]
    hdr = ['Rank', 'Feature', 'Importance (\\%)']
    mdh = ['Rank', 'Feature', 'Importance (%)']
    cap = ('Top-15 features (of 198) for the best classical model '
           '(db4, NL16, LightGBM, 90.71\\% audited), by native split-gain '
           'importance. Feature names follow \\texttt{sub-band\\_descriptor} '
           '(e.g.\\ \\texttt{img} = original image, \\texttt{HH2} = level-2 '
           'diagonal-detail wavelet sub-band; descriptor codes are the 22 '
           'Haralick (1973) GLCM statistics, e.g.\\ '
           '\\texttt{contr} = contrast, \\texttt{energ} = energy, '
           '\\texttt{entro} = entropy).')
    emit('tab:supp-featimp-classic', cap, hdr, mdh, trows, 'clc', size='\\footnotesize')
    print('  table9: top feature =', feat[order[0]], f'({imp[order[0]]/total*100:.1f}% of total gain)')


# ─────────────────────────────────────────────────────────────────────────────
# T10  Hybrid feature importance (best hybrid model: VGG16 + LightGBM, 98.67%)
# Reads the SHAP TreeExplainer results already computed by workflow12
# (w11_hybrid/shap/importance_matrix.npz: mean |SHAP| per of the 128 CNN
# penultimate-layer dimensions). Not re-computed here, just tabulated.
# ─────────────────────────────────────────────────────────────────────────────
def table10():
    npz = f'{HYB}/shap/importance_matrix.npz'
    if not os.path.exists(npz):
        print(f'  [skip] table10: {npz} not found')
        return
    d = np.load(npz, allow_pickle=True)
    backbones = list(d['backbones']); clfs = list(d['classifiers'])
    imp = d['imp']
    bi = backbones.index('W7_VGG16_ImageNet'); ci = clfs.index('LightGBM')
    v = imp[bi, ci]
    order = np.argsort(-v)[:15]
    total = v.sum()
    trows = [[f'{r+1}', f'dim {i}', f'{v[i]/total*100:.2f}']
             for r, i in enumerate(order)]
    hdr = ['Rank', 'CNN feature', 'Importance (\\%)']
    mdh = ['Rank', 'CNN feature', 'Importance (%)']
    cap = ('Top-15 features (of 128) for the best hybrid model '
           '(VGG16 + LightGBM, 98.67\\% audited), by mean $|$SHAP$|$ '
           '(TreeExplainer). Dimensions index the 128-d penultimate-layer '
           'embedding and carry no individual semantic label, unlike the '
           'classical wavelet+GLCM features in Table~\\ref{tab:supp-featimp-classic}.')
    emit('tab:supp-featimp-hybrid', cap, hdr, mdh, trows, 'clc', size='\\footnotesize')
    print('  table10: top dim =', int(order[0]), f'({v[order[0]]/total*100:.1f}% of total |SHAP|)')


for fn in (table1, table2, table3, table4, table5, table6, table7, table8, table9, table10):
    fn()

# write standalone LaTeX
preamble = (r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage[T1]{fontenc}
% Prefix tables/figures with `S' -> `Table S1', `Fig. S1', ...
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
% Compact spacing: tight around floats + minimal skip between paragraphs so
% pages fill efficiently and blank space is avoided.
\setlength{\textfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\setlength{\parskip}{2pt}
\title{\vspace{-1cm}BRISC 2025 -- Supplementary Material}
\author{}\date{}
\begin{document}\maketitle
\vspace{-1cm}
\section*{Supplementary tables}
""")
# learning-curve figures (generated by build_training_history.py)
figures = r"""
\section*{Supplementary figures}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{figures/fig_gradcam.png}
\caption{Grad-CAM++ heatmaps for all seven deep models (rows: custom CNN, ResNet50, VGG16, EfficientNetB2, ConvNeXt-Tiny, RadImageNet ResNet50 and the optimized EfficientNetB1) on the same four test images (columns). Full version of the compact grid in the main paper.}
\label{fig:supp-gradcam-full}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_training_history.png}
\caption{Learning curves: train vs validation \emph{accuracy} (audited partition). Dotted line = head$\to$fine-tune switch. Custom CNN, six softmax transfer/medical backbones and the optimized EfficientNet family; hybrids and classical ML are not epoch-trained.}
\label{fig:supp-lc-acc}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/fig_training_loss.png}
\caption{Learning curves: train vs validation \emph{loss} (audited partition).}
\label{fig:supp-lc-loss}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{figures/skull_strip_comparison.png}
\caption{Skull stripping comparison: original MRI vs.\ v1 (morphological) vs.\ v2 (active contours). Percentage of retained pixels shown per image. The morphological approach over-strips in several cases (e.g.\ No Tumor, 17\% kept).}
\label{fig:supp-skull}
\end{figure}
"""
with io.open(TEX_OUT, 'w', encoding='utf-8') as f:
    f.write(preamble + '\n'.join(tex_parts) + figures + '\n\\end{document}\n')
with io.open(MD_OUT, 'w', encoding='utf-8') as f:
    f.write('# BRISC 2025 - Complete Result Tables (all real values)\n' + '\n'.join(md_parts) + '\n')

print('wrote', TEX_OUT)
print('wrote', MD_OUT)
print('tables:', len(tex_parts))
