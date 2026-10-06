"""Recompute corrected tables from predictions, without models, images or training."""
import argparse, itertools, json
from pathlib import Path
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from common import ROOT, metrics, align_frames

FAMILY = ['cnn_canonical', 'vgg16', 'vgg16_lgbm', 'radimagenet']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT/'results/verified')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/recomputed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resamples', type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; use a fresh directory')
    frames = {p.name.removesuffix('_predictions.csv'): pd.read_csv(p)
              for p in sorted(args.predictions.glob('*_predictions.csv'))}
    if not set(FAMILY) <= set(frames):
        parser.error('Missing primary-family prediction tables')
    rows = []
    for name, frame in frames.items():
        values = metrics(frame, args.seed, args.resamples)
        rows.append({'model': name, 'evidence': 'local_model_inference' if name!='classical_refit' else 'new_python_refit',
                     'exploratory_test_selection': name.endswith('_lgbm'), **values})
    contrasts = []
    for x, y in itertools.combinations(FAMILY, 2):
        a, b = align_frames(frames[x], frames[y])
        ca, cb = a.y_true == a.y_pred, b.y_true == b.y_pred
        tab = [[int((ca&cb).sum()), int((ca&~cb).sum())],
               [int((~ca&cb).sum()), int((~ca&~cb).sum())]]
        contrasts.append({'a': x, 'b': y, 'n': len(a), 'a_only_correct': tab[0][1],
                          'b_only_correct': tab[1][0],
                          'p_nominal': mcnemar(tab, exact=False, correction=True).pvalue,
                          'p_exact': mcnemar(tab, exact=True).pvalue})
    for r, p in zip(contrasts, multipletests([r['p_nominal'] for r in contrasts], method='holm')[1]):
        r['p_holm_six_contrasts'] = p
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output/'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(args.output/'mcnemar.csv', index=False)
    (args.output/'methods.json').write_text(json.dumps({'family': FAMILY, 'n_contrasts': 6,
        'bootstrap': 'class-stratified', 'resamples': args.resamples, 'seed': args.seed,
        'row_order': 'class label then filename', 'test_selection_bias_corrected': False}, indent=2))
    print(pd.DataFrame(rows)[['model','accuracy','ci95_lo','ci95_hi']].to_string(index=False))
    print('Saved:', args.output)

if __name__ == '__main__': main()
