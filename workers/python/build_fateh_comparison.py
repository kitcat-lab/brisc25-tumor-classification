# -*- coding: utf-8 -*-
"""Same-architecture comparison: BRISC paper baselines (raw release, weighted-avg
accuracy, mean of 3 runs) vs our results (audited + unaudited). Fateh numbers are
transcribed verbatim from the dataset paper's Table 4 (Sci Data)."""
import csv, io, json, os
B = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'

# --- Fateh et al. baselines: weighted-average accuracy (%) on the raw release ---
FATEH = {
    'ResNet50': 98.20, 'ResNet101': 98.09, 'DenseNet121': 45.31, 'DenseNet169': 50.93,
    'MobileNetV2': 21.15, 'MobileNetV3': 94.42, 'EfficientNetB0': 99.20,
    'EfficientNetB1': 99.03, 'EfficientNetB2': 98.37, 'Xception': 5.72,
    'VGG16': 96.92, 'VGG19': 96.29, 'InceptionV3': 61.98,
}
# save the full baseline list
with io.open(f'{B}/fateh_baselines.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(['model', 'weighted_avg_accuracy_raw', 'status'])
    for m, a in sorted(FATEH.items(), key=lambda kv: -kv[1]):
        status = 'ok' if a >= 90 else ('unstable/failed' if a < 70 else 'moderate')
        w.writerow([m, a, status])

# --- our results (uniform protocol) for the architectures we both ran ---
def our(path):
    return json.load(open(path))['accuracy'] if os.path.exists(path) else None
LK = f'{B}/leakage_backbones'
ours = {
    'ResNet50':       (our(f'{LK}/W6_ResNet50_ImageNet/audited/metrics.json'),
                       our(f'{LK}/W6_ResNet50_ImageNet/unaudited/metrics.json')),
    'VGG16':          (our(f'{LK}/W7_VGG16_ImageNet/audited/metrics.json'),
                       our(f'{LK}/W7_VGG16_ImageNet/unaudited/metrics.json')),
    'EfficientNetB2': (our(f'{LK}/W8_EfficientNetB2_ImageNet/audited/metrics.json'),
                       our(f'{LK}/W8_EfficientNetB2_ImageNet/unaudited/metrics.json')),
    'EfficientNetB0': (our(f'{LK}/W14_EfficientNetB0_ImageNet/audited/metrics.json'),
                       our(f'{LK}/W14_EfficientNetB0_ImageNet/unaudited/metrics.json')),
}

rows = []
for m, (a, u) in ours.items():
    fr = FATEH[m]
    rows.append({'architecture': m, 'fateh_raw': fr,
                 'ours_unaudited': u, 'ours_audited': a,
                 'ours_unaud_minus_fateh': (round(u-fr,2) if u is not None else None),
                 'ours_aud_minus_fateh':   (round(a-fr,2) if a is not None else None)})
with io.open(f'{B}/same_architecture_comparison.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print('=== Fateh baselines (weighted-avg acc, raw release) ===')
for m, a in sorted(FATEH.items(), key=lambda kv: -kv[1]):
    flag = '' if a >= 90 else '   <-- unstable/failed baseline'
    print(f'  {m:16s} {a:6.2f}{flag}')
print('\n=== Same-architecture: Fateh (raw) vs ours (uniform protocol) ===')
print(f'  {"arch":16s} {"Fateh raw":>9s} {"ours unaud":>10s} {"ours aud":>9s} '
      f'{"u-F":>7s} {"a-F":>7s}')
for r in rows:
    print(f"  {r['architecture']:16s} {r['fateh_raw']:9.2f} "
          f"{(r['ours_unaudited'] or 0):10.2f} {(r['ours_audited'] or 0):9.2f} "
          f"{(r['ours_unaud_minus_fateh'] or 0):+7.2f} {(r['ours_aud_minus_fateh'] or 0):+7.2f}")
print('\nsaved fateh_baselines.csv + same_architecture_comparison.csv')
