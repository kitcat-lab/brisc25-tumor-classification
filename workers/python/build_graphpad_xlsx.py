# -*- coding: utf-8 -*-
"""Export article data as GraphPad Prism-ready .xlsx files.

Conventions:
  Column table (one group, one measure with CI): columns [Value, Lower, Upper]
     -> Prism: "Column" table; use "Enter mean, N, error" or "Bar graph" with
        low/high asymmetric error bars.
  Grouped table (categories x groups, one measure w/ CI):
     -> Prism: "Grouped" table; each subcolumn = one group; rows = categories.
  XY table (matrix / heatmap): first column = row labels, then values.
     -> Prism: "Column" or "XY" depending on plot; for heatmaps use
        the Prism Heat Map analysis on this grid.

One xlsx per figure, saved to xlsx_graphpad/.
"""
import csv, io, os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

B  = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'
OUT = f'{B}/xlsx_graphpad'
os.makedirs(OUT, exist_ok=True)

HEADER = Font(bold=True, color='FFFFFF')
HEADER_FILL = PatternFill('solid', fgColor='1F4E4A')
CTR = Alignment(horizontal='center', vertical='center', wrap_text=True)

def load_csv(p): return list(csv.DictReader(io.open(p, encoding='utf-8')))
def f(x, default=None):
    try: return float(x)
    except: return default

def new_wb(sheet_name):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = sheet_name; return wb, ws

def write_row(ws, r, values, is_header=False):
    for j, v in enumerate(values, 1):
        c = ws.cell(row=r, column=j, value=v)
        c.alignment = CTR
        if is_header:
            c.font = HEADER; c.fill = HEADER_FILL

def autosize(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

def add_notes_sheet(wb, notes):
    ws = wb.create_sheet('_README')
    for i, line in enumerate(notes, 1):
        c = ws.cell(row=i, column=1, value=line)
        c.alignment = Alignment(wrap_text=True, vertical='top')
    ws.column_dimensions['A'].width = 100

# ================= FIG 2  Grand comparison =================
def fig2():
    lk = {(r['workflow'],r['dataset']):r for r in load_csv(f'{B}/leakage_backbones/all_results.csv')}
    hy = load_csv(f'{B}/w11_hybrid/all_hybrid_results.csv')
    def hyb(bb,c):
        for r in hy:
            if r['backbone']==bb and r['classifier']==c: return r
    w7h = hyb('W7_VGG16_ImageNet','LightGBM')
    rows = [
        ('Classical ML (db4+GLCM+LGBM)',      90.71, None,  None),
        ('Scratch CNN (W2 v3)',                93.81, 91.15, 94.69),
        ('Transfer learning (VGG16 softmax)',  97.94, 96.61, 98.97),
        ('Medical pretraining (RadImageNet)',  72.12, 69.17, 75.37),
        ('Hybrid (VGG16 + LightGBM)',          float(w7h['accuracy']),
                                               float(w7h['ci95_lo']),
                                               float(w7h['ci95_hi'])),
    ]
    wb, ws = new_wb('Fig2_ApproachFamilies')
    write_row(ws, 1, ['Approach', 'Accuracy (%)', 'CI95 lower', 'CI95 upper'], is_header=True)
    for i, r in enumerate(rows, 2): write_row(ws, i, list(r))
    autosize(ws, [42, 14, 12, 12])
    add_notes_sheet(wb, [
        'FIG 2 -- Grand comparison of approach families on the audited BRISC 2025 test set (n=678).',
        'Prism table type: Column table (one grouping variable).',
        'Import: paste Approach as row labels; Accuracy as the value column; CI95 lower/upper',
        'as the "asymmetric error" pair (Prism: Format Graph > Error bars > Asymmetric,',
        'or use "Enter replicate values" with Lower/Upper).',
        'Suggested colours: colour-blind ColorBrewer Set2 (66C2A5 8DA0CB FC8D62 E78AC3 A6D854).',
        'Classical ML has no CI (single-run MATLAB report) -- leave error blank in Prism.',
    ])
    wb.save(f'{OUT}/fig2_grand_comparison.xlsx')

# ================= FIG 3  Softmax vs Hybrid per backbone =================
def fig3():
    lk = {(r['workflow'],r['dataset']):r for r in load_csv(f'{B}/leakage_backbones/all_results.csv')}
    hy = load_csv(f'{B}/w11_hybrid/all_hybrid_results.csv')
    order = ['W7_VGG16_ImageNet','W6_ResNet50_ImageNet','W9_ConvNeXtTiny_ImageNet',
             'W8_EfficientNetB2_ImageNet','W3_CNN_scratch','W10_RadImageNet_ResNet50']
    soft_map = {'W3_CNN_scratch':93.81}
    for wf in order:
        if wf in lk_keys_audited(lk):
            soft_map[wf] = float(lk[(wf,'audited')]['accuracy'])
    best_h = {}
    for r in hy:
        b = r['backbone']
        if b not in best_h or f(r['accuracy'])>f(best_h[b]['accuracy']): best_h[b] = r
    label = lambda b: b.replace('_ImageNet','').replace('_',' ')

    wb, ws = new_wb('Fig3_SoftmaxVsHybrid')
    write_row(ws, 1, ['Backbone', 'Softmax head', 'Best hybrid', 'Delta (pp)',
                       'Best hybrid classifier'], is_header=True)
    for i, b in enumerate(order, 2):
        s = soft_map[b]; h = float(best_h[b]['accuracy'])
        write_row(ws, i, [label(b), s, h, round(h-s,2), best_h[b]['classifier']])
    autosize(ws, [30, 14, 14, 12, 22])
    add_notes_sheet(wb, [
        'FIG 3 -- Softmax head vs best hybrid classifier on top of the same backbone (audited).',
        'Prism table type: Grouped table (2 groups: Softmax, Hybrid).',
        'Import: paste Backbone as row labels; Softmax head and Best hybrid as the two subcolumns.',
        'Add Delta (pp) as annotations, not as a data column.',
        'Suggested plot: interleaved bars, no CI (single-run values).',
    ])
    wb.save(f'{OUT}/fig3_softmax_vs_hybrid.xlsx')

def lk_keys_audited(lk):
    return {k[0] for k in lk if k[1]=='audited'}

# ================= FIG 4  Leakage per backbone =================
def fig4():
    lk = load_csv(f'{B}/leakage_backbones/all_results.csv')
    bbs = ['W6_ResNet50_ImageNet','W7_VGG16_ImageNet','W8_EfficientNetB2_ImageNet',
           'W9_ConvNeXtTiny_ImageNet','W10_RadImageNet_ResNet50']
    lkm = {(r['workflow'], r['dataset']): r for r in lk}
    label = lambda b: b.replace('_ImageNet','').replace('_',' ')

    wb, ws = new_wb('Fig4_LeakageDelta')
    write_row(ws, 1, ['Backbone',
                       'Audited acc', 'Audited CI lo', 'Audited CI hi',
                       'Unaudited acc', 'Unaudited CI lo', 'Unaudited CI hi',
                       'Delta (unaud - aud)'], is_header=True)
    for i, b in enumerate(bbs, 2):
        a = lkm[(b,'audited')]; u = lkm[(b,'unaudited')]
        write_row(ws, i, [label(b),
            float(a['accuracy']), float(a['ci95_lo']), float(a['ci95_hi']),
            float(u['accuracy']), float(u['ci95_lo']), float(u['ci95_hi']),
            round(float(u['accuracy'])-float(a['accuracy']),2)])
    autosize(ws, [26,12,12,12,12,12,12,14])
    add_notes_sheet(wb, [
        'FIG 4 -- Audited vs unaudited accuracy per backbone (leakage check).',
        'Prism table type: Grouped table (2 groups: Audited, Unaudited).',
        'Import: paste Backbone as row labels; Audited acc and Unaudited acc as subcolumns.',
        'For error bars, use the "asymmetric error" pair for each group:',
        '  Audited whiskers  = (Audited CI lo, Audited CI hi)',
        '  Unaudited whiskers = (Unaudited CI lo, Unaudited CI hi)',
        'All CIs overlap -> no significant leakage effect (already computed).',
    ])
    wb.save(f'{OUT}/fig4_leakage.xlsx')

# ================= FIG 5  Classical ML factorial + split (2 panels) =================
def fig5():
    rows = load_csv(f'{B}/classic_ml_all_results.csv')
    import math
    def best_per(split):
        d={}
        for r in rows:
            if r.get('split')!=split: continue
            s = r['scenario']
            if s not in d or f(r['accuracy'])>f(d[s]['accuracy']): d[s]=r
        return d
    bo, br = best_per('original'), best_per('aleatoria')

    wb = openpyxl.Workbook(); wb.remove(wb.active)
    # 5A -- factorial effects
    ws = wb.create_sheet('5A_FactorialEffects')
    def is_ss(s): return 'no_ss' not in s
    def is_fs(s): return s.endswith('_fs') and 'no_fs' not in s
    def gmean(pred):
        v = [f(bo[s]['accuracy']) for s in bo if pred(s)]
        return sum(v)/len(v) if v else None
    factors = [
        ('Skull stripping', 'No',  gmean(lambda s: not is_ss(s))),
        ('Skull stripping', 'Yes', gmean(is_ss)),
        ('Feature selection','No', gmean(lambda s: not is_fs(s))),
        ('Feature selection','Yes', gmean(is_fs)),
        ('Wavelet', 'bior1.1', gmean(lambda s: s.startswith('bior11'))),
        ('Wavelet', 'db4',     gmean(lambda s: s.startswith('db4'))),
    ]
    write_row(ws, 1, ['Factor','Level','Mean best accuracy (%)'], is_header=True)
    for i, r in enumerate(factors, 2): write_row(ws, i, list(r))
    autosize(ws, [22,14,26])
    # 5B -- official vs random per scenario
    ws = wb.create_sheet('5B_SplitComparison')
    scen = sorted(set(bo)&set(br), key=lambda s: -f(bo[s]['accuracy']))
    write_row(ws, 1, ['Scenario','Official split acc','Random split acc','Delta'], is_header=True)
    for i, s in enumerate(scen, 2):
        o = f(bo[s]['accuracy']); r = f(br[s]['accuracy'])
        write_row(ws, i, [s.replace('_',' '), o, r, round(r-o,2)])
    autosize(ws, [32,20,20,10])
    add_notes_sheet(wb, [
        'FIG 5 -- Classical ML: (A) factorial main effects; (B) official vs random split.',
        '',
        '5A -- Prism table type: Grouped table.',
        '  Row = Level; group by Factor. Values are mean-of-best-per-scenario accuracy.',
        '  Suggest paired bars per Factor (e.g. Skull No / Skull Yes side by side).',
        '',
        '5B -- Prism table type: Grouped table (2 groups: Official, Random).',
        '  Row labels = Scenario. Two subcolumns for the two splits.',
        '  Delta is annotation, not a data column in Prism.',
    ])
    wb.save(f'{OUT}/fig5_classical_factorial_split.xlsx')

# ================= FIG 6  SHAP (2 panels) =================
def fig6():
    wa = load_csv(f'{B}/w11_hybrid/shap/within_backbone_agreement.csv')
    import pandas as pd
    conc = pd.read_csv(f'{B}/w11_hybrid/shap/top10_concentration.csv', index_col=0)

    wb = openpyxl.Workbook(); wb.remove(wb.active)
    # 6A -- within-backbone agreement
    ws = wb.create_sheet('6A_WithinBackboneAgreement')
    write_row(ws, 1, ['Backbone','Mean rho','Min rho','Max rho'], is_header=True)
    wa_s = sorted(wa, key=lambda r: float(r['mean_rho_across_classifiers']))
    for i, r in enumerate(wa_s, 2):
        lab = r['backbone'].replace('_ImageNet','').replace('_',' ')
        write_row(ws, i, [lab, float(r['mean_rho_across_classifiers']),
                                float(r['min_rho']), float(r['max_rho'])])
    autosize(ws, [26,12,10,10])
    # 6B -- concentration matrix (rows = backbones, cols = classifiers)
    ws = wb.create_sheet('6B_ConcentrationMatrix')
    cols = list(conc.columns)
    write_row(ws, 1, ['Backbone'] + cols, is_header=True)
    for i, bb in enumerate(conc.index, 2):
        lab = bb.replace('_ImageNet','').replace('_',' ')
        write_row(ws, i, [lab] + [round(float(conc.loc[bb,c]),3) for c in cols])
    autosize(ws, [26] + [12]*len(cols))
    add_notes_sheet(wb, [
        'FIG 6 -- SHAP explainability (two panels).',
        '',
        '6A Within-backbone agreement -- Prism table type: Column table.',
        '  Rows = Backbone. Value = Mean rho. Error = (Mean rho - Min rho, Max rho - Mean rho)',
        '  for asymmetric range whiskers.',
        '',
        '6B Concentration matrix -- Prism heat map.',
        '  Import as XY grid; run Analyse -> Heat map. Suggested colormap: viridis.',
    ])
    wb.save(f'{OUT}/fig6_shap.xlsx')

# ================= FIG 7  McNemar p-values =================
def fig7():
    import pandas as pd
    P = pd.read_csv(f'{B}/figuras_artigo/mcnemar_pvalues.csv', index_col=0)
    wb, ws = new_wb('Fig7_McNemar_pvalues')
    cols = list(P.columns)
    # header row: first cell blank
    write_row(ws, 1, ['Method'] + cols, is_header=True)
    for i, m in enumerate(P.index, 2):
        row = [m] + [float(P.loc[m,c]) for c in cols]
        write_row(ws, i, row)
    autosize(ws, [30] + [24]*len(cols))
    # add log10 sheet for heatmap colour
    ws2 = wb.create_sheet('Fig7_log10p')
    import math
    write_row(ws2, 1, ['Method'] + cols, is_header=True)
    for i, m in enumerate(P.index, 2):
        row = [m] + [math.log10(max(float(P.loc[m,c]), 1e-300)) for c in cols]
        write_row(ws2, i, row)
    autosize(ws2, [30] + [24]*len(cols))
    add_notes_sheet(wb, [
        'FIG 7 -- McNemar paired tests on the identical 678-image test set.',
        'Prism table type: XY grid / Heat map.',
        'Two sheets provided: raw p-values and log10(p) (for a linear colour scale).',
        'Suggested colormap: RdYlGn reversed (green = significant, red = non-significant).',
        'Diagonal is 1.0 (self-comparison).',
    ])
    wb.save(f'{OUT}/fig7_mcnemar.xlsx')

# ================= FIG 8  Literature positioning =================
def fig8():
    import re
    src = f'/mnt/c/Users/cbot/Desktop/BRISC pos graduação/literature/_literature_comparison.xlsx'
    wb_lit = openpyxl.load_workbook(src, data_only=True); ws_lit = wb_lit['Comparison']
    lit_rows = list(ws_lit.iter_rows(values_only=True))
    hdr = next(list(r) for r in lit_rows if r and r[0]=='Study'); ix = {h:i for i,h in enumerate(hdr)}
    def parse_acc(s):
        if s is None: return None
        vals = [float(x) for x in re.findall(r'(\d{2,3}\.?\d?)', str(s)) if 40<=float(x)<=100]
        return max(vals) if vals else None
    def ds_family(d):
        d = (d or '').lower()
        if 'brisc' in d: return 'BRISC 2025'
        if 'cheng' in d or 'figshare' in d: return 'Figshare/Cheng'
        if 'br35h' in d or 'navoneel' in d: return 'BR35H'
        if 'brats' in d: return 'BraTS'
        if 'kaggle' in d: return 'Kaggle'
        return 'Other/multi-source'
    entries = []
    for r in lit_rows:
        if not r or not r[0]: continue
        st = str(r[0]).strip()
        if st in ('Study','Reference') or (st.isupper() and (len(r)<2 or not r[1])): continue
        if 'THIS WORK' in st.upper(): continue
        acc = parse_acc(r[ix['Accuracy']]) if 'Accuracy' in ix else None
        if acc is None: continue
        ds = str(r[ix['Dataset']]) if 'Dataset' in ix and r[ix['Dataset']] else ''
        bb = str(r[ix['Backbone']]) if 'Backbone' in ix and r[ix['Backbone']] else ''
        ch = str(r[ix['Classifier head']]) if 'Classifier head' in ix and r[ix['Classifier head']] else ''
        yr = str(r[ix['Year']]) if 'Year' in ix and r[ix['Year']] else ''
        author = re.split(r'\d', st)[0].strip().rstrip('(').strip()
        method = (bb + (' + '+ch if ch and ch.lower() not in ('softmax','none','') else '')).strip(' +')
        entries.append([f'{author} et al. ({yr})', method, ds, ds_family(ds), acc, False])
    entries += [
        ['This work - Classical',       'db4+GLCM+LGBM',      'BRISC 2025 (audited)','BRISC 2025',90.71,True],
        ['This work - CNN scratch',     '3-block CNN (W2)',   'BRISC 2025 (audited)','BRISC 2025',93.81,True],
        ['This work - Transfer',        'VGG16 (softmax)',    'BRISC 2025 (audited)','BRISC 2025',97.94,True],
        ['This work - RadImageNet',     'ResNet50',           'BRISC 2025 (audited)','BRISC 2025',72.12,True],
        ['This work - Hybrid',          'VGG16 + LightGBM',   'BRISC 2025 (audited)','BRISC 2025',98.67,True],
    ]
    entries.sort(key=lambda e: e[4])
    wb, ws = new_wb('Fig8_LiteraturePositioning')
    write_row(ws, 1, ['Study','Method','Dataset','Dataset family','Accuracy (%)','This work?'], is_header=True)
    for i, r in enumerate(entries, 2):
        write_row(ws, i, [r[0], r[1], r[2], r[3], r[4], 'yes' if r[5] else 'no'])
    autosize(ws, [40, 40, 34, 22, 14, 12])
    add_notes_sheet(wb, [
        'FIG 8 -- Positioning against the literature, coloured by dataset.',
        'Prism table type: Column table with grouping.',
        'Import: paste Study as row labels; Accuracy as the single value column.',
        'In Prism: colour bars by "Dataset family" (Group by row category).',
        'Highlight "This work?" == yes with hatched pattern or bold outline.',
        'Suggested horizontal bar orientation, sorted ascending by accuracy.',
    ])
    wb.save(f'{OUT}/fig8_literature_positioning.xlsx')

# ================= APPENDIX A1 (same as 5B but standalone) =================
def figA1():
    import os as _os
    # copy 5B sheet as a standalone
    rows = load_csv(f'{B}/classic_ml_all_results.csv')
    def best_per(split):
        d={}
        for r in rows:
            if r.get('split')!=split: continue
            s = r['scenario']
            if s not in d or f(r['accuracy'])>f(d[s]['accuracy']): d[s]=r
        return d
    bo, br = best_per('original'), best_per('aleatoria')
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    ws = wb.create_sheet('A1_BestPerScenario_bothsplits')
    write_row(ws, 1, ['Scenario','Wavelet','Skull','FeatSel','Split',
                       'Classifier','NL','Accuracy (%)','F1 (%)','AUC (%)'], is_header=True)
    def key_of(s):
        p=s.split('_'); wl='bior1.1' if p[0]=='bior11' else p[0]
        ss='Yes' if 'no_ss' not in s else 'No'
        fs='Yes' if s.endswith('_fs') and 'no_fs' not in s else 'No'
        return wl,ss,fs
    scen = sorted(set(bo)&set(br), key=lambda s: -f(bo[s]['accuracy']))
    r = 2
    for s in scen:
        wl,ss,fs = key_of(s)
        for split_lab, dct in [('Official', bo),('Random', br)]:
            rr = dct[s]
            write_row(ws, r, [s, wl, ss, fs, split_lab,
                              rr['classificador'], rr['NumLevels'],
                              f(rr['accuracy']), f(rr['f1_score']), f(rr['auc'])])
            r += 1
    autosize(ws, [26,10,8,10,10,14,8,14,10,10])
    ws2 = wb.create_sheet('A1_GraphView_ForPrism')
    write_row(ws2, 1, ['Scenario','Official split acc','Random split acc','Delta (pp)'], is_header=True)
    for i, s in enumerate(scen, 2):
        o = f(bo[s]['accuracy']); rr = f(br[s]['accuracy'])
        write_row(ws2, i, [s.replace('_',' '), o, rr, round(rr-o,2)])
    autosize(ws2, [32,20,20,12])
    add_notes_sheet(wb, [
        'APPENDIX A.1 -- Classical ML, official vs random split.',
        'Two sheets:',
        '  A1_BestPerScenario_bothsplits -- long-format detail (16 rows).',
        '  A1_GraphView_ForPrism         -- wide format ready for a grouped bar.',
        'For the figure, use the second sheet as a Grouped table (Official / Random subcolumns).',
    ])
    wb.save(f'{OUT}/figA1_split_comparison.xlsx')

# ================= Per-class metrics =================
def perclass():
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    for wf, path in [
        ('W6_ResNet50',  f'{B}/leakage_backbones/W6_ResNet50_ImageNet/audited/perclass.csv'),
        ('W7_VGG16',     f'{B}/leakage_backbones/W7_VGG16_ImageNet/audited/perclass.csv'),
        ('W10_RadImg_20',f'{B}/leakage_backbones_deep/W10_RadImageNet_ResNet50/audited/perclass.csv'),
    ]:
        if not os.path.exists(path): continue
        pc = load_csv(path)
        ws = wb.create_sheet(wf)
        write_row(ws, 1, ['Class','Precision (%)','Recall (%)','F1 (%)','Support'], is_header=True)
        for i, r in enumerate(pc, 2):
            write_row(ws, i, [r['class'], f(r['precision']), f(r['recall']), f(r['f1']), f(r['support'])])
        autosize(ws, [16,14,12,12,10])
    add_notes_sheet(wb, [
        'Per-class metrics on the audited test set for selected models.',
        'Prism: Grouped table (Precision / Recall / F1 as 3 subcolumns; rows = Class).',
    ])
    wb.save(f'{OUT}/perclass_metrics.xlsx')

# ================= Master workbook (index) =================
def master_index():
    files = sorted(os.listdir(OUT))
    wb, ws = new_wb('Index')
    write_row(ws, 1, ['File', 'Figure', 'Prism table type', 'One-line description'], is_header=True)
    idx = 2
    guide = {
      'fig2_grand_comparison.xlsx':      ('Fig 2', 'Column',   'Best accuracy per approach family + CI'),
      'fig3_softmax_vs_hybrid.xlsx':     ('Fig 3', 'Grouped',  'Softmax vs hybrid per backbone'),
      'fig4_leakage.xlsx':               ('Fig 4', 'Grouped',  'Audited vs unaudited per backbone (with CI)'),
      'fig5_classical_factorial_split.xlsx':('Fig 5','Grouped','Factorial effects + official vs random split'),
      'fig6_shap.xlsx':                  ('Fig 6', 'Column + Heat map', 'Within-backbone agreement + concentration matrix'),
      'fig7_mcnemar.xlsx':               ('Fig 7', 'Heat map', 'McNemar p-value matrix (raw + log10)'),
      'fig8_literature_positioning.xlsx':('Fig 8', 'Column w/ grouping', 'Literature accuracies coloured by dataset'),
      'figA1_split_comparison.xlsx':     ('Fig A1','Grouped',  'Appendix A: official vs random split per scenario'),
      'perclass_metrics.xlsx':           ('Extra', 'Grouped',  'Per-class precision/recall/F1 for W6/W7/W10'),
    }
    for f_ in files:
        if f_ == '_INDEX.xlsx': continue
        g = guide.get(f_, ('','',''))
        write_row(ws, idx, [f_, g[0], g[1], g[2]]); idx += 1
    autosize(ws, [42, 12, 22, 60])
    add_notes_sheet(wb, [
        'GraphPad Prism-ready data for the BRISC 2025 article.',
        'Each xlsx corresponds to ONE figure. Every file has a _README sheet with',
        'the exact Prism table type to select on import and any special notes',
        '(colour palette, whisker convention, orientation).',
        '',
        'Colour palette used in the paper (ColorBrewer Set2 -- colour-blind safe):',
        '  Classical ML         #66C2A5',
        '  CNN scratch          #8DA0CB',
        '  Transfer (softmax)   #FC8D62',
        '  Medical pretraining  #E78AC3',
        '  Hybrid               #A6D854',
        '',
        'Datasets in Fig 8 -- suggested colours:',
        '  BRISC 2025           #1F4E4A',
        '  Figshare/Cheng       #1F77B4',
        '  Kaggle               #E08A1E',
        '  BraTS                #8E44AD',
        '  BR35H                #C0392B',
        '  Other/multi-source   #7F7F7F',
        '',
        'All numbers were computed by the notebooks in this repository. Regenerate any file',
        'by re-running build_graphpad_xlsx.py after new results are available.',
    ])
    wb.save(f'{OUT}/_INDEX.xlsx')

for fn in (fig2, fig3, fig4, fig5, fig6, fig7, fig8, figA1, perclass):
    fn(); print(' -', fn.__name__)
master_index(); print(' - master_index')
print(f'\nAll files -> {OUT}')
