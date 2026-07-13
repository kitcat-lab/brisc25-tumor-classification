# -*- coding: utf-8 -*-
"""Build report_comparacao_global.md: the paper's central comparison of ALL
approaches on BRISC 2025 audited (original split). Read-only; trains nothing.
Classical numbers come from classic_ml_all_results.csv (MATLAB reports);
deep/hybrid numbers are the values already in our result files."""
import csv, io, collections
OUT = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'

rows = list(csv.DictReader(io.open(f'{OUT}/classic_ml_all_results.csv', encoding='utf-8')))
def f(x):
    try: return float(x)
    except: return -1

orig = [r for r in rows if r['split'] == 'original']

# best per scenario (original split)
best_scen = {}
for r in orig:
    s = r['scenario']
    if s not in best_scen or f(r['accuracy']) > f(best_scen[s]['accuracy']):
        best_scen[s] = r

# factorial main effects on the BEST-per-scenario accuracy (original split).
def is_ss(s):    return 'no_ss' not in s          # ss scenarios never contain 'no_ss'
def is_fs(s):    return s.endswith('_fs') and 'no_fs' not in s
def group_mean(pred):
    vals = [f(best_scen[s]['accuracy']) for s in best_scen if pred(s)]
    return sum(vals)/len(vals) if vals else float('nan')

eff = {
    'skull_strip_yes': group_mean(lambda s: is_ss(s)),
    'skull_strip_no':  group_mean(lambda s: not is_ss(s)),
    'feat_sel_yes':    group_mean(lambda s: is_fs(s)),
    'feat_sel_no':     group_mean(lambda s: not is_fs(s)),
    'wavelet_bior11':  group_mean(lambda s: s.startswith('bior11')),
    'wavelet_db4':     group_mean(lambda s: s.startswith('db4')),
    'wavelet_local':   group_mean(lambda s: s.startswith('local')),
}

# overall best classical original-split
best_orig = max(orig, key=lambda r: f(r['accuracy']))

L = []
L.append('# Grand comparison: all approaches on BRISC 2025 (audited)\n')
L.append('> **Paper thesis.** Prior brain-tumour MRI studies compare approaches across '
         '*different* datasets with heterogeneous (often un-audited) splits, confounding '
         'the comparison with dataset difficulty and possible train-to-test leakage. Here '
         'every approach - classical ML with hand-crafted features, a CNN trained from '
         'scratch, transfer-learning backbones, and deep+classical hybrids - is evaluated '
         'on the **same forensically audited BRISC 2025 partition and the same official '
         '(original) train/test split**, so the numbers are directly comparable and the '
         'question "which approach really wins" can be answered cleanly.\n')
L.append('> All values are test-set metrics on the audited partition (train 4,363 / test '
         '678). Classical-ML numbers use the **original** split (not the random split) to '
         'match the deep-learning evaluation. Sources: `classic_ml_all_results.csv` '
         '(MATLAB reports), `leakage_backbones/`, `w11_hybrid/`.\n')

L.append('\n## 1. Headline: best of each approach family (audited, original split)\n')
L.append('| Approach family | Best configuration | Acc | F1 | AUC |')
L.append('|-----------------|--------------------|----:|---:|----:|')
L.append(f"| Classical ML (hand-crafted features) | {best_orig['scenario']} / {best_orig['NumLevels']} / {best_orig['classificador']} (198 wavelet+GLCM) | {best_orig['accuracy']} | {best_orig['f1_score']} | {best_orig['auc']} |")
L.append('| CNN from scratch | W2 3-block CNN (v3 protocol) | 93.81 | 93.94 | 99.55 |')
L.append('| Transfer learning (softmax) | W7 VGG16 (ImageNet) | 97.94 | 97.96 | 99.86 |')
L.append('| Medical pretraining (softmax) | W10 ResNet50 (RadImageNet) | 72.12 | 72.78 | 90.81 |')
L.append('| **Hybrid (deep features + classical)** | **W7 VGG16 + LightGBM** | **98.67** | **98.69** | **99.83** |')

L.append('\n**Answer to the paper question.** On a single validated, de-duplicated dataset '
         'with a fixed split, the ranking is unambiguous:\n')
L.append('classical ML (~90.7%) < CNN from scratch (~93.8%) < transfer-learning softmax '
         '(~97.9%) < **deep+classical hybrid (98.7%)**. The best approach is a '
         'transfer-learning backbone used as a feature extractor with a gradient-boosted '
         'classifier on top - i.e. the hybrid, led by VGG16 + LightGBM.\n')

L.append('\n## 2. Full progression (audited, best per row)\n')
L.append('| Rank | Approach | Config | Acc | F1 | AUC |')
L.append('|-----:|----------|--------|----:|---:|----:|')
prog = [
    ('Hybrid', 'W7 VGG16 + LightGBM', 98.67, 98.69, 99.83),
    ('Hybrid', 'W6 ResNet50 + SVM (RBF)', 97.49, 97.57, 99.84),
    ('Transfer (softmax)', 'W7 VGG16 (ImageNet)', 97.94, 97.96, 99.86),
    ('Transfer (softmax)', 'W6 ResNet50 (ImageNet)', 97.49, 97.56, 99.80),
    ('Hybrid', 'W3 scratch CNN + LightGBM', 96.76, 96.77, 99.77),
    ('Hybrid', 'W9 ConvNeXt-Tiny + SVM', 96.46, 96.55, 99.64),
    ('Transfer (softmax)', 'W9 ConvNeXt-Tiny', 94.99, 95.16, 99.63),
    ('Hybrid', 'W8 EfficientNetB2 + SVM', 94.84, 94.99, 99.31),
    ('Transfer (softmax)', 'W8 EfficientNetB2', 93.95, 94.18, 99.30),
    ('CNN from scratch', 'W2 3-block CNN (v3)', 93.81, 93.94, 99.55),
    ('Classical ML', f"{best_orig['scenario']} {best_orig['NumLevels']} {best_orig['classificador']}",
        f(best_orig['accuracy']), f(best_orig['f1_score']), f(best_orig['auc'])),
    ('Medical pretrain (softmax)', 'W10 RadImageNet (12 ep)', 72.12, 72.78, 90.81),
]
prog.sort(key=lambda t: -t[3])
for i, (fam, cfg, a, f1, au) in enumerate(prog, 1):
    L.append(f'| {i} | {fam} | {cfg} | {a} | {f1} | {au} |')

L.append('\n## 3. Classical ML detail (best per scenario, audited/original split)\n')
L.append('Scenario = wavelet family x skull-strip x feature-selection. Best NL/classifier shown.\n')
L.append('| Scenario | NL | Classifier | Acc | F1 | AUC |')
L.append('|----------|----|-----------|----:|---:|----:|')
for s in sorted(best_scen, key=lambda s: -f(best_scen[s]['accuracy'])):
    r = best_scen[s]
    L.append(f"| {s} | {r['NumLevels']} | {r['classificador']} | {r['accuracy']} | {r['f1_score']} | {r['auc']} |")

L.append('\n## 4. Classical ML factorial main effects (audited/original split)\n')
L.append('Mean of the best per-scenario accuracy, grouped by each factor.\n')
L.append('| Factor | Level | Mean acc | Effect |')
L.append('|--------|-------|---------:|--------|')
L.append(f"| Skull stripping | no  | {eff['skull_strip_no']:.2f} | reference |")
L.append(f"| Skull stripping | yes | {eff['skull_strip_yes']:.2f} | {eff['skull_strip_yes']-eff['skull_strip_no']:+.2f} pp |")
L.append(f"| Feature selection | no  | {eff['feat_sel_no']:.2f} | reference |")
L.append(f"| Feature selection | yes | {eff['feat_sel_yes']:.2f} | {eff['feat_sel_yes']-eff['feat_sel_no']:+.2f} pp |")
L.append(f"| Wavelet | bior1.1 | {eff['wavelet_bior11']:.2f} | - |")
L.append(f"| Wavelet | db4 | {eff['wavelet_db4']:.2f} | - |")
import math
if not math.isnan(eff['wavelet_local']):
    L.append(f"| Wavelet | local | {eff['wavelet_local']:.2f} | - |")

L.append('\n## 5. Notes on comparability\n')
L.append('- **Split matters.** The best classical accuracy on the *random* split is 91.38% '
         '(bior1.1/NL16/LGBM) but on the *original* split it is '
         f"{best_orig['accuracy']}% - only the original-split numbers are comparable to the "
         'deep-learning results, which all use the official audited split.\n')
L.append('- **Same 678-image audited test set** underlies every row of Section 1-2, so '
         'differences reflect the method, not the dataset or the split.\n')
L.append('- Hybrid, transfer and scratch numbers are reproducible from the notebooks; '
         'classical numbers are read from the MATLAB `Relatorio_*.xlsx` reports and '
         'aggregated in `classic_ml_all_results.csv`.\n')

io.open(f'{OUT}/report_comparacao_global.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('wrote report_comparacao_global.md')
print(f"best classical (original): {best_orig['scenario']} {best_orig['NumLevels']} "
      f"{best_orig['classificador']} = {best_orig['accuracy']}%")
print('factorial effects:', {k: round(v,2) for k,v in eff.items()})
