#!/usr/bin/env python
"""Retrain the W2 scratch CNN using the leakage_v3 protocol exactly, and
persist it by OVERWRITING `modelo_brisc.h5` (the file used by every
downstream workflow) so all analyses are consistent on a single artefact.

The previous `modelo_brisc.h5` in the project root was produced by an earlier
run of `workflow2_cnn_train.ipynb` that converged to only ~88% audited
accuracy -- below the 93.07% achieved by the corrected `workflow2_leakage_v3`
protocol (deterministic dataframe shuffle, `val_gen shuffle=False`, seed=42).
Overwriting is intentional: the paper cannot report two conflicting numbers
for the "W2 scratch CNN" -- there is one canonical trained model, and it is
this v3-protocol one. The PyQt6 GUI (`app.py`) may need a re-run of
`criar_pickle.py` to regenerate its pickle after this retrain, but the GUI
can be updated as a separate concern.

Config copied verbatim from `workflow2_leakage_v3.ipynb`:
  IMG_SIZE=128, BATCH_SIZE=32, EPOCHS=30, SEED=42
  Adam (default LR=1e-3), categorical_crossentropy, EarlyStopping(patience=5, val_loss)
  Augmentation: rotation 10, zoom 0.1, shift 0.05, horizontal_flip=True, val_split=0.2
  Deterministic dataframe shuffle via .sample(frac=1, random_state=SEED)
  val_gen shuffle=False (required for TF 2.19+ stability)
"""
import os, json, time, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 30
CLASSES    = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
N_CLASSES  = len(CLASSES)
SEED       = 42

AUDITED_BASE = '<original-wsl-directory>/data/brisc2025_clean/classification_task'
PROJECT_DIR  = '<original-project-directory>'
# Overwrite the canonical model.h5 so every workflow + the GUI share one
# artefact. Also emit a .keras twin (identical weights) for future-proofing.
OUT_H5       = f'{PROJECT_DIR}/modelo_brisc.h5'
OUT_KERAS    = f'{PROJECT_DIR}/modelo_brisc.keras'
OUT_META     = f'{PROJECT_DIR}/modelo_brisc_v3_meta.json'


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


def build_cnn():
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32,  (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),          # NOTE: the 128-d layer W11 will tap
        layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])
    m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


def make_generators(train_df, test_df):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10, zoom_range=0.1,
        width_shift_range=0.05, height_shift_range=0.05,
        horizontal_flip=True,
        validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)
    tr = train_datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES,
        subset='training', shuffle=True, seed=SEED)
    va = train_datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES,
        subset='validation', shuffle=False, seed=SEED)   # v3 fix
    te = test_datagen.flow_from_dataframe(
        test_df, x_col='path', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES,
        shuffle=False)
    return tr, va, te


def main():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix, precision_recall_fscore_support)

    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError: pass

    set_global_seed(SEED)
    print('=' * 60); print('  W2 CNN retrain -- v3 protocol'); print('=' * 60)

    train_df, test_df = build_paths_df(AUDITED_BASE)
    tr, va, te = make_generators(train_df, test_df)
    print(f'  train_df={len(train_df)}  test_df={len(test_df)}  '
          f'batches: train={len(tr)} val={len(va)} test={len(te)}')

    model = build_cnn()
    model.summary()
    es = EarlyStopping(monitor='val_loss', patience=5,
                        restore_best_weights=True, verbose=1)
    t0 = time.time()
    hist = model.fit(tr, validation_data=va, epochs=EPOCHS,
                      callbacks=[es], verbose=2)
    train_time = time.time() - t0

    te.reset()
    y_true  = np.asarray(te.classes)
    y_proba = model.predict(te, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_true, y_pred) * 100
    f1  = f1_score(y_true, y_pred, average='macro') * 100
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100

    print(f'\n  Test Accuracy : {acc:.2f}%')
    print(f'  Test F1 macro : {f1:.2f}%   AUC OvR: {auc:.2f}%')
    print(f'  Epochs run    : {len(hist.history["loss"])}  train time {train_time:.1f}s')

    # Save in BOTH formats. The scratch CNN has no LayerScale-type layers,
    # so .h5 works fine; we keep .keras as the future-proof canonical.
    model.save(OUT_KERAS)
    model.save(OUT_H5)
    print(f'  saved {OUT_KERAS}')
    print(f'  saved {OUT_H5}')

    # Metadata for tracing
    with open(OUT_META, 'w') as f:
        json.dump({
            'source_notebook': 'workflow2_leakage_v3.ipynb (audited protocol)',
            'seed': SEED, 'img_size': IMG_SIZE, 'batch_size': BATCH_SIZE,
            'epochs_max': EPOCHS, 'epochs_run': len(hist.history['loss']),
            'train_time_s': round(train_time, 1),
            'test_accuracy': round(acc, 2),
            'test_f1_macro': round(f1, 2),
            'test_auc_ovr':  round(auc, 2),
            'note': 'Corrected training: deterministic dataframe shuffle + '
                    'val_gen shuffle=False. Replaces modelo_brisc.h5 for W11.',
        }, f, indent=2)
    print(f'  meta -> {OUT_META}')


if __name__ == '__main__':
    main()
