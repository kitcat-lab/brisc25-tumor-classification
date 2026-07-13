#!/usr/bin/env python
"""Retrain the W9 ConvNeXt-Tiny backbone using the identical protocol of
`train_one_backbone.py` (seed=42, augmentation, 2-stage FT with 5+12 epochs),
but persist the trained model in the newer Keras native `.keras` format
instead of the legacy `.h5`.

The Keras 3 legacy H5 loader has a documented incompatibility with the
ConvNeXt family: LayerScale is not auto-registered, and the H5 weight
serialisation packs LayerScale factors alongside the layer kernels in an
order that cannot be reconstructed by a fresh ConvNeXtTiny build. The
`.keras` zipped native format round-trips correctly.

Only the audited partition is retrained -- the unaudited artefact is not
needed by W11 (which is on the audited partition only).

Usage:
    python retrain_w9_keras.py
"""
import os, json, pickle, time, timeit, random, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

SEED         = 42
BATCH_SIZE   = 8      # ConvNeXt-Tiny at 224 needs the smaller batch, matching W6-W10
EPOCHS_HEAD  = 5
EPOCHS_FT    = 12
LR_HEAD      = 1e-3
LR_FT        = 1e-5
N_UNFREEZE   = 60
VAL_SPLIT    = 0.20
CLASSES      = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
N_CLASSES    = len(CLASSES)
IMG_SIZE     = 224

AUDITED_BASE = '/root/brisc/data/brisc2025_clean/classification_task'
OUT_DIR      = ('/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui/'
                'leakage_backbones/W9_ConvNeXtTiny_ImageNet/audited')
os.makedirs(OUT_DIR, exist_ok=True)


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
        df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
        out[split] = df
    return out['train'], out['test']


def make_generators(train_df, test_df, preprocessing_fn):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocessing_fn,
        rotation_range=10, zoom_range=0.1,
        width_shift_range=0.05, height_shift_range=0.05,
        horizontal_flip=True, validation_split=VAL_SPLIT)
    test_pre = ImageDataGenerator(preprocessing_function=preprocessing_fn)
    tr = train_aug.flow_from_dataframe(train_df, x_col='path', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES,
        subset='training', shuffle=True, seed=SEED)
    va = train_aug.flow_from_dataframe(train_df, x_col='path', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES,
        subset='validation', shuffle=False, seed=SEED)
    te = test_pre.flow_from_dataframe(test_df, x_col='path', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES, shuffle=False)
    return tr, va, te


def main():
    import tensorflow as tf
    from tensorflow.keras import layers, optimizers, callbacks
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import ConvNeXtTiny
    from tensorflow.keras.applications.convnext import preprocess_input as pi_convnext
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix, precision_recall_fscore_support)

    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError: pass

    set_global_seed(SEED)
    print('=' * 60); print('  W9 ConvNeXt-Tiny -- retrain saving .keras'); print('=' * 60)

    train_df, test_df = build_paths_df(AUDITED_BASE)
    tr, va, te = make_generators(train_df, test_df, pi_convnext)

    backbone = ConvNeXtTiny(weights='imagenet', include_top=False,
                             input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    x = layers.Dense(128, activation='relu', name='feat_dense')(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    out = layers.Dense(N_CLASSES, activation='softmax')(x)
    top = Model(inputs, out)
    top.compile(optimizer=optimizers.Adam(learning_rate=LR_HEAD),
                loss='categorical_crossentropy', metrics=['accuracy'])

    es_a = callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                    patience=3, restore_best_weights=True, verbose=1)
    t0 = time.time()
    h_a = top.fit(tr, validation_data=va, epochs=EPOCHS_HEAD,
                   callbacks=[es_a], verbose=2)
    t_a = time.time() - t0

    # Stage B
    backbone.trainable = True
    for L in backbone.layers[:-N_UNFREEZE]:
        L.trainable = False
    top.compile(optimizer=optimizers.Adam(learning_rate=LR_FT),
                loss='categorical_crossentropy', metrics=['accuracy'])
    es_b = callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                    patience=5, restore_best_weights=True, verbose=1)
    rl   = callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max',
                                        factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    t0 = time.time()
    h_b = top.fit(tr, validation_data=va, epochs=EPOCHS_FT,
                   callbacks=[es_b, rl], verbose=2)
    t_b = time.time() - t0

    te.reset()
    y_true  = np.asarray(te.classes)
    y_proba = top.predict(te, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_true, y_pred) * 100
    f1  = f1_score(y_true, y_pred, average='macro') * 100
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100
    print(f'\n  Test Accuracy : {acc:.2f}%')
    print(f'  Test F1 macro : {f1:.2f}%   AUC OvR: {auc:.2f}%')
    print(f'  Stage A : {len(h_a.history["loss"])} epochs, {t_a:.1f}s')
    print(f'  Stage B : {len(h_b.history["loss"])} epochs, {t_b:.1f}s')

    # SAVE IN .keras FORMAT -- the whole point of this script.
    kpath = f'{OUT_DIR}/model.keras'
    top.save(kpath)
    print(f'  saved {kpath}')

    # Overwrite predictions and companion CSVs so all downstream summary code
    # (leakage_backbones/all_results.csv aggregation etc.) picks up the new run.
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}) \
      .to_csv(f'{OUT_DIR}/predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred),
                 index=CLASSES, columns=CLASSES) \
      .to_csv(f'{OUT_DIR}/confusion_matrix.csv')
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=range(N_CLASSES))
    pd.DataFrame({'class': CLASSES,
                  'precision': (prec_c*100).round(2),
                  'recall':    (rec_c*100).round(2),
                  'f1':        (f1_c*100).round(2),
                  'support':   sup_c.astype(int)}) \
      .to_csv(f'{OUT_DIR}/perclass.csv', index=False)
    combined = {k: h_a.history.get(k, []) + h_b.history.get(k, [])
                for k in set(list(h_a.history) + list(h_b.history))}
    with open(f'{OUT_DIR}/history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in combined.items()}, f, indent=2)
    with open(f'{OUT_DIR}/metrics.json', 'w') as f:
        json.dump({
            'workflow': 'W9_ConvNeXtTiny_ImageNet', 'dataset': 'audited',
            'img_size': IMG_SIZE, 'batch_size': BATCH_SIZE,
            'train_imgs': tr.samples, 'val_imgs': va.samples, 'test_imgs': te.samples,
            'accuracy': round(acc, 2), 'f1_macro': round(f1, 2), 'auc_ovr': round(auc, 2),
            'epochs_head': len(h_a.history['loss']),
            'epochs_ft':   len(h_b.history['loss']),
            'train_time_s': round(t_a + t_b, 1),
            'note': 'retrained; saved in .keras format (H5 unusable for ConvNeXt in Keras 3)',
        }, f, indent=2)
    print('Done.')


if __name__ == '__main__':
    main()
