# -*- coding: utf-8 -*-
"""Aggregate the classical-ML MATLAB reports (Relatorio_*.xlsx) into one tidy
table. Keeps the LATEST timestamp per scenario. Read-only; trains nothing."""
import openpyxl, os, re, csv, io
RES = ('/mnt/c/Users/cbot/Desktop/BRISC pos graduação/ML classico')
OUT = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'

# Group Relatorio files by scenario, keep latest timestamp.
files = [f for f in os.listdir(RES) if f.startswith('Relatorio_') and f.endswith('.xlsx')]
by_scen = {}
for f in files:
    m = re.match(r'Relatorio_(.+)_(\d{8}_\d{4})\.xlsx', f)
    if not m:
        continue
    scen, ts = m.group(1), m.group(2)
    if scen not in by_scen or ts > by_scen[scen][0]:
        by_scen[scen] = (ts, f)

print('scenarios (latest file each):')
for scen, (ts, f) in sorted(by_scen.items()):
    print(f'  {scen:22s} <- {f}')

rows = []
for scen, (ts, f) in by_scen.items():
    wb = openpyxl.load_workbook(f'{RES}/{f}', read_only=True, data_only=True)
    ws = wb['Main_Results']
    it = ws.iter_rows(values_only=True)
    hdr = [str(c) for c in next(it)]
    for r in it:
        if r[0] is None:
            continue
        d = dict(zip(hdr, r))
        d['_file'] = f
        rows.append(d)
    wb.close()

# Write tidy CSV
keys = ['scenario', 'NumLevels', 'split', 'classificador', 'n_features',
        'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'gmean', 'auc', '_file']
with io.open(f'{OUT}/classic_ml_all_results.csv', 'w', encoding='utf-8', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=keys, extrasaction='ignore')
    w.writeheader()
    for d in rows:
        w.writerow(d)
print(f'\nTotal rows aggregated: {len(rows)}  -> classic_ml_all_results.csv')

def fnum(x):
    try: return float(x)
    except: return -1

for split in ('original', 'aleatoria'):
    sub = [d for d in rows if str(d.get('split')) == split]
    sub.sort(key=lambda d: fnum(d.get('accuracy')), reverse=True)
    print(f'\n=== TOP 8 classical (split={split}) ===')
    for d in sub[:8]:
        print(f"  {d['scenario']:22s} {d['NumLevels']:4} {d['classificador']:5} "
              f"acc={d['accuracy']:>6} f1={d['f1_score']:>6} auc={d['auc']:>6}")
