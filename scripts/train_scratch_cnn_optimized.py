#!/usr/bin/env python
"""Historical four-block scratch CNN protocol: no BatchNormalization, seven
convolutions across four blocks, global average pooling, Dense(256), dropout
and a learning-rate schedule. The configured L2 coefficient is zero.

Usage: python train_scratch_cnn_optimized.py <audited|unaudited> <base_dir> <out_root>
Fresh executions use stratified bootstrap and identified prediction outputs.
"""
import os, sys, json, time, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import pandas as pd

SEED       = 42
IMG_SIZE   = 128
BATCH      = 32
EPOCHS     = 80
LR         = 1e-3
VAL_SPLIT  = 0.20
L2         = 0.0          # v1 over-regularised (collapsed to pituitary); reg now light
CLASSES    = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
N_CLASSES  = len(CLASSES)


def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def build_paths_df(base_dir):
    out = {}
    for split in ('train', 'test'):
        rows = []
        for cls in CLASSES:
            d = f'{base_dir}/{split}/{cls}'
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rows.append({'path': f'{d}/{fn}', 'label': cls})
        out[split] = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return out['train'], out['test']


def make_generators(train_df, test_df):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # Two generators for train_df: an augmented one for the train split, and a
    # CLEAN one for the val split. This isolates the val signal from augmentation
    # noise, which had been inflating val accuracy to ~96% while test collapsed.
    tr_aug = ImageDataGenerator(rescale=1. / 255,
        rotation_range=10, zoom_range=0.10, width_shift_range=0.05,
        height_shift_range=0.05, horizontal_flip=True,
        validation_split=VAL_SPLIT)
    tr_clean = ImageDataGenerator(rescale=1. / 255, validation_split=VAL_SPLIT)
    te_pre = ImageDataGenerator(rescale=1. / 255)
    common = dict(target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
                  class_mode='categorical', classes=CLASSES)
    tr = tr_aug.flow_from_dataframe(train_df, x_col='path', y_col='label',
                                    subset='training', shuffle=True, seed=SEED, **common)
    va = tr_clean.flow_from_dataframe(train_df, x_col='path', y_col='label',
                                      subset='validation', shuffle=False, seed=SEED, **common)
    te = te_pre.flow_from_dataframe(test_df, x_col='path', y_col='label',
                                    shuffle=False, **common)
    return tr, va, te


def build_model():
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.models import Sequential

    # NO BatchNorm anywhere: v3 with BN gave 86.43% due to unstable BN running
    # statistics on this small single-source dataset (~4.4k train imgs), causing
    # a huge train-vs-inference-mode gap (val oscillating 0.28-0.89). This v4
    # tests whether the deeper architecture (4 blocks + GAP + LR schedule) alone
    # can beat the canonical 3-block CNN when the BN confound is removed.
    def conv_block(filters, n=2):
        blk = []
        for _ in range(n):
            blk += [layers.Conv2D(filters, 3, padding='same', activation='relu')]
        blk += [layers.MaxPooling2D(2)]
        return blk

    reg = regularizers.l2(L2) if L2 else None
    model = Sequential([layers.Input((IMG_SIZE, IMG_SIZE, 3))]
        + conv_block(32, 2)
        + conv_block(64, 2)
        + conv_block(128, 2)
        + [layers.Conv2D(256, 3, padding='same', activation='relu'),
           layers.GlobalAveragePooling2D(),
           layers.Dense(256, activation='relu',
                        kernel_regularizer=reg, name='feat_dense'),
           layers.Dropout(0.4, seed=SEED),
           layers.Dense(N_CLASSES, activation='softmax')])
    return model


from common import bootstrap_acc_ci


def main():
    if len(sys.argv) != 4:
        print('Usage: python train_scratch_cnn_optimized.py <audited|unaudited> '
              '<base_dir> <out_root>', file=sys.stderr)
        sys.exit(2)
    partition, base_dir, out_root = sys.argv[1:4]
    out_dir = f'{out_root}/{partition}'

    if Path(out_dir, 'metrics.json').exists() or Path(out_dir, 'comparison.csv').exists() or Path(out_dir, 'model.h5').exists() or Path(out_dir, 'model.keras').exists():
        raise FileExistsError('Existing execution; choose a fresh output root')
    os.makedirs(out_dir, exist_ok=True)

    import tensorflow as tf
    from tensorflow.keras import optimizers, callbacks
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix, precision_recall_fscore_support)
    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError: pass

    set_global_seed(SEED)
    print('=' * 64); print(f'  OPTIMISED scratch CNN  ({partition})  '
                            f'size={IMG_SIZE} batch={BATCH}'); print('=' * 64)

    train_df, test_df = build_paths_df(base_dir)
    tr, va, te = make_generators(train_df, test_df)
    model = build_model()
    model.compile(optimizer=optimizers.Adam(LR),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print(f'  params: {model.count_params():,}')

    cb = [callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=12,
                                  restore_best_weights=True, verbose=2),
          callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5,
                                      patience=4, min_lr=1e-6, verbose=2)]
    t0 = time.time()
    h = model.fit(tr, validation_data=va, epochs=EPOCHS, callbacks=cb, verbose=2)
    train_time = time.time() - t0

    te.reset()
    y_true = np.asarray(te.classes)
    y_proba = model.predict(te, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100
    ci_lo, ci_hi = bootstrap_acc_ci(y_true, y_pred)
    print(f'\n  Test Accuracy : {acc:.2f}%  CI95 [{ci_lo:.2f}, {ci_hi:.2f}]')
    print(f'  F1 macro {f1:.2f}%  AUC {auc:.2f}%  ({len(h.history["loss"])} epochs)')

    model.save(f'{out_dir}/model.keras')
    with open(f'{out_dir}/history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in h.history.items()}, f, indent=2)
    pd.DataFrame({'filename': [Path(p).name for p in te.filenames], 'y_true': y_true, 'y_pred': y_pred, **{f'p_{c}': y_proba[:, i] for i, c in enumerate(CLASSES)}}).to_csv(f'{out_dir}/predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred), index=CLASSES, columns=CLASSES) \
      .to_csv(f'{out_dir}/confusion_matrix.csv')
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, labels=range(N_CLASSES))
    pd.DataFrame({'class': CLASSES, 'precision': (prec_c * 100).round(2),
                  'recall': (rec_c * 100).round(2), 'f1': (f1_c * 100).round(2),
                  'support': sup_c.astype(int)}).to_csv(f'{out_dir}/perclass.csv', index=False)
    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump({'model': 'scratch_cnn_optimised', 'dataset': partition,
                   'img_size': IMG_SIZE, 'batch_size': BATCH, 'params': int(model.count_params()),
                   'train_imgs': tr.samples, 'val_imgs': va.samples, 'test_imgs': te.samples,
                   'accuracy': round(acc, 2), 'f1_macro': round(f1, 2), 'auc_ovr': round(auc, 2),
                   'ci_method': 'class_stratified_bootstrap', 'ci95_lo': round(ci_lo, 2), 'ci95_hi': round(ci_hi, 2),
                   'epochs': len(h.history['loss']), 'train_time_s': round(train_time, 1)}, f, indent=2)
    print(f'  saved -> {out_dir}')


if __name__ == '__main__':
    main()
