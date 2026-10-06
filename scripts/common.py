"""Shared statistics and prediction validation; percentages are on a 0–100 scale."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
ROOT = Path(__file__).resolve().parents[1]

def bootstrap_acc_ci(y_true, y_pred, n_iter=1000, alpha=.05, seed=42):
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError('Labels and predictions must have equal, nonzero length')
    groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_iter):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(100 * np.mean(y_true[idx] == y_pred[idx]))
    return tuple(float(v) for v in np.percentile(values, [100*alpha/2, 100*(1-alpha/2)]))

def save_predictions(path, filenames, y_true, y_pred, probabilities):
    frame = pd.DataFrame({'filename': [Path(p).name for p in filenames],
                          'y_true': y_true, 'y_pred': y_pred,
                          **{f'p_{c}': probabilities[:, i] for i, c in enumerate(CLASSES)}})
    validate_predictions(frame)
    frame.to_csv(path, index=False)

def validate_predictions(frame):
    required = {'filename', 'y_true', 'y_pred'}
    if not required <= set(frame) or frame.empty:
        raise ValueError('Prediction table must contain filename, y_true, y_pred')
    if frame.filename.isna().any() or frame.filename.duplicated().any():
        raise ValueError('Missing or duplicate image identifiers')
    for col in ['y_true', 'y_pred']:
        if frame[col].isna().any() or not frame[col].isin(range(len(CLASSES))).all():
            raise ValueError(f'Invalid class labels in {col}')
    probability_columns = [f'p_{c}' for c in CLASSES]
    if any(c in frame for c in probability_columns):
        if not all(c in frame for c in probability_columns):
            raise ValueError('Incomplete probability columns')
        probs = frame[probability_columns].to_numpy()
        if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
            raise ValueError('Invalid probabilities')
        if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
            raise ValueError('Probabilities do not sum to one')
        if not np.array_equal(probs.argmax(axis=1), frame.y_pred.to_numpy()):
            raise ValueError('Prediction disagrees with probability argmax')

def metrics(frame, seed=42, n_iter=1000):
    validate_predictions(frame)
    # Fix row order for finite Monte Carlo reproducibility across input permutations.
    frame = frame.sort_values(['y_true', 'filename']).reset_index(drop=True)
    y, p = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    low, high = bootstrap_acc_ci(y, p, seed=seed, n_iter=n_iter)
    result = {'n': len(y), 'correct': int(np.sum(y == p)),
              'accuracy': 100*accuracy_score(y, p),
              'f1_macro': 100*f1_score(y, p, average='macro'),
              'ci95_lo': low, 'ci95_hi': high, 'ci_method': 'class_stratified_bootstrap',
              'bootstrap_resamples': n_iter, 'seed': seed}
    columns = [f'p_{c}' for c in CLASSES]
    if all(c in frame for c in columns):
        result['auc_macro_ovr'] = 100*roc_auc_score(y, frame[columns], multi_class='ovr', average='macro')
    return result

def align_frames(a, b):
    validate_predictions(a); validate_predictions(b)
    a, b = a.set_index('filename').sort_index(), b.set_index('filename').sort_index()
    if not a.index.equals(b.index) or not a.y_true.equals(b.y_true):
        raise ValueError('Paired tests require identical image IDs and labels')
    return a, b
