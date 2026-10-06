#!/usr/bin/env python
"""Train ONE (backbone x dataset) pair in an isolated process.

Why a standalone script instead of a notebook cell?
    Under `tf.config.experimental.set_memory_growth(True)`, TensorFlow never
    returns GPU memory to the OS within a running process. `clear_session()`
    only resets the graph. On an 8 GB laptop RTX 4060 shared with the Windows
    desktop, that means the 2nd..Nth backbone in a loop inherits a GPU that is
    already (nearly) full and OOMs. Running each backbone in its own process
    guarantees the OS reclaims all VRAM the moment the process exits, so every
    workflow starts with a clean GPU.

The notebook `workflow6_10_backbones_leakage.ipynb` orchestrates one call to
this script per (workflow, dataset), then aggregates the per-run metrics.json.

Usage:
    python train_one_backbone.py <workflow_name> <dataset_name> <base_dir> <out_root>

Writes:
    <out_root>/<workflow_name>/<dataset_name>/metrics.json   (always, even on error)
    ... plus model.h5, predictions.csv, confusion_matrix.csv, perclass.csv, history.json
"""
import os, sys, json, time, timeit, random, gc
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Hyper-parameters (identical across all backbones for a fair audit-vs-unaudit
# comparison -- only img_size and preprocessing_function vary per architecture).
# Single source of truth: mirrors W2-W5.
# ----------------------------------------------------------------------------
SEED         = 42
EPOCHS_HEAD  = 5
EPOCHS_FT    = 12
LR_HEAD      = 1e-3
LR_FT        = 1e-5
N_UNFREEZE   = 60
VAL_SPLIT    = 0.20
CLASSES      = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
N_CLASSES    = len(CLASSES)

# Per-backbone batch size. Even with process isolation, VGG16 / EfficientNetB2
# (260px) / ConvNeXt have far larger activation maps than ResNet50, so they get
# a smaller batch as a safety net. Dial these down further if a run still OOMs.
DEFAULT_BATCH = 16
BATCH_OVERRIDE = {
    'W7_VGG16_ImageNet':           8,   # 16x64x224x224 first activation ~= 820 MB
    'W8_EfficientNetB2_ImageNet':  8,   # 260px input
    'W9_ConvNeXtTiny_ImageNet':    8,
}


def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def build_paths_df(base_dir):
    """Build class-shuffled train/test dataframes."""
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


def make_generators(train_df, test_df, img_size, preprocessing_fn, batch_size):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocessing_fn,
        rotation_range=10, zoom_range=0.1,
        width_shift_range=0.05, height_shift_range=0.05,
        horizontal_flip=True, validation_split=VAL_SPLIT)
    test_pre = ImageDataGenerator(preprocessing_function=preprocessing_fn)
    tr = train_aug.flow_from_dataframe(
        train_df, x_col='path', y_col='label',
        target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', classes=CLASSES,
        subset='training', shuffle=True, seed=SEED)
    va = train_aug.flow_from_dataframe(
        train_df, x_col='path', y_col='label',
        target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', classes=CLASSES,
        subset='validation', shuffle=False, seed=SEED)
    te = test_pre.flow_from_dataframe(
        test_df, x_col='path', y_col='label',
        target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', classes=CLASSES, shuffle=False)
    return tr, va, te


def bootstrap_acc_ci(y_true, y_pred, n_iter=1000, alpha=0.05, seed=42):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed); n = len(y_true); accs = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        accs.append((y_true[idx] == y_pred[idx]).mean() * 100)
    return float(np.percentile(accs, 100*alpha/2)), float(np.percentile(accs, 100*(1-alpha/2)))


def evaluate(model, test_gen):
    test_gen.reset()
    y_true = np.asarray(test_gen.classes)
    y_proba = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    return y_true, y_pred, y_proba


def measure_inference_ms(model, img_size, n_warmup=10, n_iter=50):
    dummy = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
    for _ in range(n_warmup): model(dummy, training=False)
    t = timeit.timeit(lambda: model(dummy, training=False), number=n_iter)
    return (t / n_iter) * 1000.0


# ----------------------------------------------------------------------------
# Backbone factories -- each returns (keras_backbone, preprocessing_fn, img_size)
# ----------------------------------------------------------------------------
def build_resnet50_imagenet(img_size=224):
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input as pi
    return (ResNet50(weights='imagenet', include_top=False,
                     input_shape=(img_size, img_size, 3)), pi, img_size)


def build_vgg16_imagenet(img_size=224):
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input as pi
    return (VGG16(weights='imagenet', include_top=False,
                  input_shape=(img_size, img_size, 3)), pi, img_size)


def build_efficientnetb2_imagenet(img_size=260):
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.applications.efficientnet import preprocess_input as pi
    return (EfficientNetB2(weights='imagenet', include_top=False,
                           input_shape=(img_size, img_size, 3)), pi, img_size)


def build_efficientnetb0_imagenet(img_size=224):
    # Same-model comparator to the BRISC dataset paper's best baseline
    # (EfficientNetB0 = 99.20% on the raw/unaudited release, Fateh et al. 2026).
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input as pi
    return (EfficientNetB0(weights='imagenet', include_top=False,
                           input_shape=(img_size, img_size, 3)), pi, img_size)


def build_convnexttiny_imagenet(img_size=224):
    try:
        from tensorflow.keras.applications import ConvNeXtTiny
        from tensorflow.keras.applications.convnext import preprocess_input as pi
    except ImportError:
        raise RuntimeError('ConvNeXt not supported by this TF version')
    return (ConvNeXtTiny(weights='imagenet', include_top=False,
                         input_shape=(img_size, img_size, 3)), pi, img_size)


def build_radimagenet_resnet50(img_size=224):
    """RadImageNet ResNet50 with pretrained medical-image weights.

    These weights are NOT bundled with keras.applications -- they are the
    RadImageNet release (Mei et al. 2022) trained on 1.35M CT/MRI/US images.
    Download RadImageNet-ResNet50_notop.h5 (~100 MB) from
    https://github.com/BMEII-AI/RadImageNet and place it at the path below.
    If absent, this workflow is skipped cleanly (see notebook Section 5).
    """
    from tensorflow.keras.applications import ResNet50
    w = '<original-wsl-directory>/radimagenet/RadImageNet-ResNet50_notop.h5'
    if not os.path.exists(w):
        raise RuntimeError(f'RadImageNet weights not found at {w}. '
                           f'See notebook Section 5 for the download instructions.')
    base = ResNet50(weights=None, include_top=False,
                    input_shape=(img_size, img_size, 3))
    base.load_weights(w)
    def pi_radimagenet(x):
        return (x / 127.5) - 1.0   # RadImageNet uses [-1, 1] normalisation
    return base, pi_radimagenet, img_size


WORKFLOWS = {
    'W6_ResNet50_ImageNet':        build_resnet50_imagenet,
    'W7_VGG16_ImageNet':           build_vgg16_imagenet,
    'W8_EfficientNetB2_ImageNet':  build_efficientnetb2_imagenet,
    'W9_ConvNeXtTiny_ImageNet':    build_convnexttiny_imagenet,
    'W10_RadImageNet_ResNet50':    build_radimagenet_resnet50,
    'W14_EfficientNetB0_ImageNet': build_efficientnetb0_imagenet,
}


# ----------------------------------------------------------------------------
# Head + 2-stage fine-tuning
# ----------------------------------------------------------------------------
def build_top_model(backbone, img_size):
    from tensorflow.keras import layers, optimizers
    from tensorflow.keras.models import Model
    backbone.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    x = layers.Dense(128, activation='relu', name='feat_dense')(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    out = layers.Dense(N_CLASSES, activation='softmax')(x)
    m = Model(inputs, out)
    m.compile(optimizer=optimizers.Adam(learning_rate=LR_HEAD),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def fine_tune_top_layers(backbone, top_model, n_unfreeze=N_UNFREEZE):
    from tensorflow.keras import optimizers
    backbone.trainable = True
    for layer in backbone.layers[:-n_unfreeze]:
        layer.trainable = False
    top_model.compile(optimizer=optimizers.Adam(learning_rate=LR_FT),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    return top_model


def train_workflow(name, build_fn, dataset_name, base_dir, out_root):
    """Trains one backbone on one dataset. Returns metrics dict."""
    from tensorflow.keras import callbacks
    from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                                 roc_auc_score, precision_recall_fscore_support)

    batch_size = BATCH_OVERRIDE.get(name, DEFAULT_BATCH)
    print('=' * 60); print(f'  {name}  ({dataset_name})  batch={batch_size}'); print('=' * 60)
    set_global_seed(SEED)
    backbone, preprocessing_fn, img_size = build_fn()
    train_df, test_df = build_paths_df(base_dir)
    tr, va, te = make_generators(train_df, test_df, img_size, preprocessing_fn, batch_size)

    top = build_top_model(backbone, img_size)

    # Stage A: head only
    es_a = callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                   patience=3, restore_best_weights=True, verbose=1)
    t0 = time.time()
    h_a = top.fit(tr, validation_data=va, epochs=EPOCHS_HEAD, callbacks=[es_a], verbose=2)
    t_a = time.time() - t0

    # Stage B: fine-tune
    top = fine_tune_top_layers(backbone, top)
    es_b = callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                   patience=5, restore_best_weights=True, verbose=1)
    rl   = callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max',
                                       factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    t0 = time.time()
    h_b = top.fit(tr, validation_data=va, epochs=EPOCHS_FT, callbacks=[es_b, rl], verbose=2)
    t_b = time.time() - t0

    y_true, y_pred, y_proba = evaluate(top, te)
    acc = accuracy_score(y_true, y_pred) * 100
    f1  = f1_score(y_true, y_pred, average='macro') * 100
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100
    ci_lo, ci_hi = bootstrap_acc_ci(y_true, y_pred, seed=SEED)
    lat = measure_inference_ms(top, img_size)

    print(f"\n  Test Accuracy : {acc:.2f}%   CI95 [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Test F1 macro : {f1:.2f}%   AUC OvR: {auc:.2f}%")
    print(f"  Stage A       : {len(h_a.history['loss'])} epochs, {t_a:.1f}s")
    print(f"  Stage B       : {len(h_b.history['loss'])} epochs, {t_b:.1f}s")
    print(f"  Inference     : {lat:.2f} ms/image")

    # Persist artefacts (layout mirrors W2/W3/W4/W5)
    out_dir = f'{out_root}/{name}/{dataset_name}'
    os.makedirs(out_dir, exist_ok=True)
    top.save(f'{out_dir}/model.h5')
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}) \
      .to_csv(f'{out_dir}/predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred),
                 index=CLASSES, columns=CLASSES) \
      .to_csv(f'{out_dir}/confusion_matrix.csv')
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=range(N_CLASSES))
    pd.DataFrame({
        'class': CLASSES,
        'precision': (prec_c*100).round(2),
        'recall':    (rec_c*100).round(2),
        'f1':        (f1_c*100).round(2),
        'support':   sup_c.astype(int),
    }).to_csv(f'{out_dir}/perclass.csv', index=False)
    combined = {k: h_a.history.get(k, []) + h_b.history.get(k, [])
                for k in set(list(h_a.history) + list(h_b.history))}
    with open(f'{out_dir}/history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in combined.items()}, f, indent=2)

    return {
        'workflow': name, 'dataset': dataset_name, 'img_size': img_size,
        'batch_size': batch_size,
        'train_imgs': tr.samples, 'val_imgs': va.samples, 'test_imgs': te.samples,
        'accuracy': round(acc, 2), 'f1_macro': round(f1, 2), 'auc_ovr': round(auc, 2),
        'ci95_lo': round(ci_lo, 2), 'ci95_hi': round(ci_hi, 2),
        'epochs_head': len(h_a.history['loss']),
        'epochs_ft':   len(h_b.history['loss']),
        'train_time_s': round(t_a + t_b, 1),
        'inference_ms': round(lat, 2),
    }


def main():
    if len(sys.argv) != 5:
        print('Usage: python train_one_backbone.py <workflow_name> <dataset_name> '
              '<base_dir> <out_root>', file=sys.stderr)
        sys.exit(2)
    name, dataset_name, base_dir, out_root = sys.argv[1:5]

    if name not in WORKFLOWS:
        print(f'Unknown workflow {name!r}. Known: {list(WORKFLOWS)}', file=sys.stderr)
        sys.exit(2)

    out_dir = f'{out_root}/{name}/{dataset_name}'
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = f'{out_dir}/metrics.json'

    # GPU setup -- memory growth so this process only takes what it needs; the
    # OS reclaims everything when the process exits (that is the whole point of
    # running one backbone per process).
    import tensorflow as tf
    print('TF:', tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print('GPUs:', gpus)
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            print('memory_growth warning:', e)
    if not gpus:
        print('WARNING: no GPU detected -- training on CPU will be very slow.')

    try:
        row = train_workflow(name, WORKFLOWS[name], dataset_name, base_dir, out_root)
    except Exception as e:
        # Write a failure record so the orchestrator can report it and move on.
        row = {'workflow': name, 'dataset': dataset_name,
               'accuracy': None, 'error': str(e)[:200]}
        print(f'FAILED {name}/{dataset_name}: {e}', file=sys.stderr)

    with open(metrics_path, 'w') as f:
        json.dump(row, f, indent=2)
    print(f'Wrote {metrics_path}')

    gc.collect()
    sys.exit(0)


if __name__ == '__main__':
    main()
