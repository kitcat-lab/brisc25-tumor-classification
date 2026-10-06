#!/usr/bin/env python
"""Optimised fine-tuning for the EfficientNet family (B0-B4).

Why a separate script? Our uniform two-stage protocol (freeze base, then unfreeze
only the top-60 layers at lr 1e-5) systematically under-fits EfficientNet: on the
audited partition B0 reached 89.1% and B2 93.9%, far below ResNet50/VGG16 (~98%)
and below the BRISC dataset paper's own B0 baseline (99.20% on the raw release).
Several factors change together: backbone unfreezing, learning rate, duration,
and BatchNormalization trainability. This is not a BatchNorm-specific ablation.

Optimised protocol used here:
  Stage A: backbone frozen, head trained 5 epochs at lr 1e-3.
  Stage B: unfreeze ALL layers EXCEPT BatchNormalization (kept frozen),
           fine-tune up to 30 epochs at lr 1e-4 with EarlyStopping + ReduceLROnPlateau.

Uniform-protocol results (W8/W14) are NOT overwritten: outputs go to
`efficientnet_optimized/<variant>/<partition>/`.

Usage:
    python train_efficientnet_optimized.py <variant> <partition> <base_dir> <out_root>
      variant   in {B0,B1,B2,B3,B4}
      partition in {audited,unaudited}
"""
import os, sys, json, time, timeit, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import pandas as pd

SEED         = 42
EPOCHS_HEAD  = 5
EPOCHS_FT    = 30
LR_HEAD      = 1e-3
LR_FT        = 1e-4          # higher than the uniform 1e-5 because we unfreeze the whole net
VAL_SPLIT    = 0.20
CLASSES      = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
N_CLASSES    = len(CLASSES)

# native input size + batch per variant (batch chosen for 8 GB VRAM)
VARIANT = {
    'B0': dict(size=224, batch=16),
    'B1': dict(size=240, batch=16),
    'B2': dict(size=260, batch=8),
    'B3': dict(size=300, batch=8),
    'B4': dict(size=380, batch=4),
}


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


def make_generators(train_df, test_df, img_size, batch, preprocessing_fn):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    tr_aug = ImageDataGenerator(preprocessing_function=preprocessing_fn,
        rotation_range=10, zoom_range=0.1, width_shift_range=0.05,
        height_shift_range=0.05, horizontal_flip=True, validation_split=VAL_SPLIT)
    te_pre = ImageDataGenerator(preprocessing_function=preprocessing_fn)
    common = dict(target_size=(img_size, img_size), batch_size=batch,
                  class_mode='categorical', classes=CLASSES)
    tr = tr_aug.flow_from_dataframe(train_df, x_col='path', y_col='label',
                                    subset='training', shuffle=True, seed=SEED, **common)
    va = tr_aug.flow_from_dataframe(train_df, x_col='path', y_col='label',
                                    subset='validation', shuffle=False, seed=SEED, **common)
    te = te_pre.flow_from_dataframe(test_df, x_col='path', y_col='label',
                                    shuffle=False, **common)
    return tr, va, te


def make_backbone(variant, img_size):
    from tensorflow.keras.applications import (EfficientNetB0, EfficientNetB1,
        EfficientNetB2, EfficientNetB3, EfficientNetB4)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    cls = {'B0': EfficientNetB0, 'B1': EfficientNetB1, 'B2': EfficientNetB2,
           'B3': EfficientNetB3, 'B4': EfficientNetB4}[variant]
    base = cls(weights='imagenet', include_top=False,
               input_shape=(img_size, img_size, 3))
    return base, preprocess_input


from common import bootstrap_acc_ci


def main():
    if len(sys.argv) != 5:
        print('Usage: python train_efficientnet_optimized.py <B0|B1|B2|B3|B4> '
              '<audited|unaudited> <base_dir> <out_root>', file=sys.stderr)
        sys.exit(2)
    variant, partition, base_dir, out_root = sys.argv[1:5]
    if variant not in VARIANT:
        print(f'Unknown variant {variant!r}', file=sys.stderr); sys.exit(2)
    cfg = VARIANT[variant]; img_size, batch = cfg['size'], cfg['batch']

    out_dir = f'{out_root}/{variant}/{partition}'

    if Path(out_dir, 'metrics.json').exists() or Path(out_dir, 'comparison.csv').exists() or Path(out_dir, 'model.h5').exists() or Path(out_dir, 'model.keras').exists():
        raise FileExistsError('Existing execution; choose a fresh output root')
    os.makedirs(out_dir, exist_ok=True)

    import tensorflow as tf
    from tensorflow.keras import layers, optimizers, callbacks
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import BatchNormalization
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix, precision_recall_fscore_support)
    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError: pass

    set_global_seed(SEED)
    print('=' * 64); print(f'  EfficientNet{variant}  ({partition})  '
                            f'size={img_size} batch={batch}  OPTIMISED'); print('=' * 64)

    backbone, pi = make_backbone(variant, img_size)
    train_df, test_df = build_paths_df(base_dir)
    tr, va, te = make_generators(train_df, test_df, img_size, batch, pi)

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    x = layers.Dense(128, activation='relu', name='feat_dense')(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    out = layers.Dense(N_CLASSES, activation='softmax')(x)
    model = Model(inputs, out)

    # Stage A -- head only
    backbone.trainable = False
    model.compile(optimizer=optimizers.Adam(LR_HEAD),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    es_a = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3,
                                   restore_best_weights=True, verbose=2)
    t0 = time.time()
    h_a = model.fit(tr, validation_data=va, epochs=EPOCHS_HEAD, callbacks=[es_a], verbose=2)
    t_a = time.time() - t0

    # Stage B -- unfreeze ALL layers EXCEPT BatchNormalization (the EfficientNet fix)
    backbone.trainable = True
    n_bn = 0
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False; n_bn += 1
    print(f'  unfroze backbone, kept {n_bn} BatchNorm layers frozen')
    model.compile(optimizer=optimizers.Adam(LR_FT),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    es_b = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=6,
                                   restore_best_weights=True, verbose=2)
    rl = callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max',
                                     factor=0.5, patience=3, min_lr=1e-7, verbose=2)
    t0 = time.time()
    h_b = model.fit(tr, validation_data=va, epochs=EPOCHS_FT, callbacks=[es_b, rl], verbose=2)
    t_b = time.time() - t0

    te.reset()
    y_true = np.asarray(te.classes)
    y_proba = model.predict(te, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100
    ci_lo, ci_hi = bootstrap_acc_ci(y_true, y_pred)

    print(f'\n  Test Accuracy : {acc:.2f}%  CI95 [{ci_lo:.2f}, {ci_hi:.2f}]')
    print(f'  F1 macro {f1:.2f}%  AUC {auc:.2f}%  (stageA {len(h_a.history["loss"])}ep, '
          f'stageB {len(h_b.history["loss"])}ep)')

    model.save(f'{out_dir}/model.keras')
    # concatenate the two-stage training history (head warmup + fine-tune) into one
    # per-epoch record, same schema as the other deep models -> enables learning
    # curves. Boundary head->fine-tune is at epoch = len(stage A) = epochs_head.
    history = {}
    for k in set(h_a.history) | set(h_b.history):
        history[k] = ([float(x) for x in h_a.history.get(k, [])]
                      + [float(x) for x in h_b.history.get(k, [])])
    with open(f'{out_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    pd.DataFrame({'filename': [Path(p).name for p in te.filenames], 'y_true': y_true, 'y_pred': y_pred, **{f'p_{c}': y_proba[:, i] for i, c in enumerate(CLASSES)}}).to_csv(f'{out_dir}/predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred), index=CLASSES, columns=CLASSES) \
      .to_csv(f'{out_dir}/confusion_matrix.csv')
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, labels=range(N_CLASSES))
    pd.DataFrame({'class': CLASSES, 'precision': (prec_c*100).round(2),
                  'recall': (rec_c*100).round(2), 'f1': (f1_c*100).round(2),
                  'support': sup_c.astype(int)}).to_csv(f'{out_dir}/perclass.csv', index=False)
    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump({'variant': f'EfficientNet{variant}', 'protocol': 'optimised (full FT, BN frozen)',
                   'dataset': partition, 'img_size': img_size, 'batch_size': batch,
                   'train_imgs': tr.samples, 'val_imgs': va.samples, 'test_imgs': te.samples,
                   'accuracy': round(acc, 2), 'f1_macro': round(f1, 2), 'auc_ovr': round(auc, 2),
                   'ci_method': 'class_stratified_bootstrap', 'ci95_lo': round(ci_lo, 2), 'ci95_hi': round(ci_hi, 2),
                   'epochs_head': len(h_a.history['loss']), 'epochs_ft': len(h_b.history['loss']),
                   'train_time_s': round(t_a + t_b, 1)}, f, indent=2)
    print(f'  saved -> {out_dir}')


if __name__ == '__main__':
    main()
