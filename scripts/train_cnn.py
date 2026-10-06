#!/usr/bin/env python
"""Train the historical canonical three-block CNN protocol into a fresh directory.

Three conv/pooling blocks (32/64/128), Flatten, Dense(128), dropout 0.5 and
softmax. Seed 42, Adam 1e-3, up to 30 epochs, val_loss early stopping with
patience 5. Historical augmentation and 20% validation split are retained.
Evaluation adds identified probabilities and class-stratified bootstrap CIs.
This trains a new execution; the recovered checkpoint is never overwritten.
"""
import os, json, time, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import pandas as pd

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 30
CLASSES    = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
N_CLASSES  = len(CLASSES)
SEED       = 42

AUDITED_BASE = None
PROJECT_DIR = None
# Save this new execution independently; do not replace historical models.
# artefact. Also emit a .keras twin (identical weights) for future-proofing.
OUT_H5 = OUT_KERAS = OUT_META = None


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
    import argparse
    global AUDITED_BASE, PROJECT_DIR, OUT_H5, OUT_KERAS, OUT_META
    parser = argparse.ArgumentParser(description='Train the canonical three-block CNN protocol into a NEW output directory.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    AUDITED_BASE = args.dataset
    PROJECT_DIR = args.output
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=False)
    OUT_H5 = str(Path(PROJECT_DIR) / 'model.h5')
    OUT_KERAS = str(Path(PROJECT_DIR) / 'model.keras')
    OUT_META = str(Path(PROJECT_DIR) / 'metrics.json')
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

    from common import save_predictions, bootstrap_acc_ci
    save_predictions(Path(PROJECT_DIR) / 'predictions.csv', te.filenames, y_true, y_pred, y_proba)
    # Metadata for tracing
    with open(OUT_META, 'w') as f:
        json.dump({
            'source_notebook': 'workflow2_leakage_v3.ipynb (audited protocol)',
            'seed': SEED, 'img_size': IMG_SIZE, 'batch_size': BATCH_SIZE,
            'epochs_max': EPOCHS, 'epochs_run': len(hist.history['loss']),
            'train_time_s': round(train_time, 1),
            'test_accuracy': round(acc, 2),
            'ci95_stratified': bootstrap_acc_ci(y_true, y_pred),
            'test_f1_macro': round(f1, 2),
            'test_auc_ovr':  round(auc, 2),
            'note': 'Corrected training: deterministic dataframe shuffle + '
                    'val_gen shuffle=False. Replaces modelo_brisc.h5 for W11.',
        }, f, indent=2)
    print(f'  meta -> {OUT_META}')


if __name__ == '__main__':
    main()
