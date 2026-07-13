#!/usr/bin/env python
"""W11 -- Hybrid: fine-tuned backbone (W6..W10) + W3 CNN as feature extractors,
7 classifiers on top. Persists trained classifiers + scaler so a downstream
SHAP notebook can reuse them without retraining.

Mirrors the W3 protocol (workflow3_hybrid_all_classifiers.ipynb) with identical
classifier hyperparameters, standardisation and random_state, so a direct
"plain-CNN vs transfer-learning-backbone" comparison is possible on the same
audited partition.

Usage:
    python train_w11_hybrid.py <backbone_name> <base_dir> <out_root>

Backbones:
    W3_CNN_scratch              (128px, /255, feat via layers[:-1])
    W6_ResNet50_ImageNet        (224px, resnet50.preprocess_input, feat_dense)
    W7_VGG16_ImageNet           (224px, vgg16.preprocess_input, feat_dense)
    W8_EfficientNetB2_ImageNet  (260px, efficientnet.preprocess_input, feat_dense)
    W9_ConvNeXtTiny_ImageNet    (224px, convnext.preprocess_input, feat_dense)
    W10_RadImageNet_ResNet50    (224px, [-1,1] via /127.5-1, feat_dense)

Outputs per backbone under <out_root>/<backbone>/:
    features.npz               cached 128-d features + labels
    scaler.pkl                 fitted StandardScaler
    classifiers/<name>.pkl     each of the 7 trained classifiers
    comparison.csv             accuracy/f1/auc/CI per classifier
    predictions_best.csv       y_true + y_pred for the best classifier
    confusion_matrix_best.csv  4x4 confusion matrix of the best classifier
    perclass_best.csv          per-class metrics of the best classifier
    best.json                  which classifier won + its accuracy
"""
import os, sys, json, time, pickle
import numpy as np
import pandas as pd

CLASSES  = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
N_CLASSES = len(CLASSES)
SEED = 42

# The pretrained-model directory (audited partition) and the W3 CNN artefact.
PROJECT_DIR  = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'
LEAKAGE_ROOT = f'{PROJECT_DIR}/leakage_backbones'
# W3 CNN feature extractor. `modelo_brisc.h5` is the single canonical scratch
# CNN, retrained with the v3 protocol (deterministic dataframe shuffle +
# val_gen shuffle=False + seed=42, ~93 % audited). All workflows and the GUI
# share this one artefact for consistency.
W3_MODEL     = f'{PROJECT_DIR}/modelo_brisc.h5'

# Per-backbone config: img_size, preprocessing choice, model path, feature-layer method.
# `feat_layer` = layer name to cut at (via get_layer). None => use layers[:-1] (W3 pattern).
BACKBONE_CONFIG = {
    'W3_CNN_scratch': {
        'img_size': 128, 'preprocess': 'rescale_255',
        'model_path': W3_MODEL, 'feat_layer': None,
    },
    'W6_ResNet50_ImageNet': {
        'img_size': 224, 'preprocess': 'resnet50',
        'model_path': f'{LEAKAGE_ROOT}/W6_ResNet50_ImageNet/audited/model.h5',
        'feat_layer': 'feat_dense',
    },
    'W7_VGG16_ImageNet': {
        'img_size': 224, 'preprocess': 'vgg16',
        'model_path': f'{LEAKAGE_ROOT}/W7_VGG16_ImageNet/audited/model.h5',
        'feat_layer': 'feat_dense',
    },
    'W8_EfficientNetB2_ImageNet': {
        'img_size': 260, 'preprocess': 'efficientnet',
        'model_path': f'{LEAKAGE_ROOT}/W8_EfficientNetB2_ImageNet/audited/model.h5',
        'feat_layer': 'feat_dense',
    },
    'W9_ConvNeXtTiny_ImageNet': {
        # Keras 3 cannot deserialise ConvNeXt from .h5 (LayerScale + weight
        # layout bugs). This entry points to the .keras retrain produced by
        # retrain_w9_keras.py; if it does not yet exist the worker will error
        # out with a clear message.
        'img_size': 224, 'preprocess': 'convnext',
        'model_path': f'{LEAKAGE_ROOT}/W9_ConvNeXtTiny_ImageNet/audited/model.keras',
        'feat_layer': 'feat_dense',
    },
    'W10_RadImageNet_ResNet50': {
        'img_size': 224, 'preprocess': 'radimagenet',
        'model_path': f'{LEAKAGE_ROOT}/W10_RadImageNet_ResNet50/audited/model.h5',
        'feat_layer': 'feat_dense',
    },
    # Headline single model: EfficientNetB1 with model-specific optimised
    # fine-tuning (full unfreeze, BatchNorm frozen), 98.82 % audited. Registered
    # here only so w11_gradcam.py can produce its Grad-CAM row; it is NOT part of
    # the hybrid feature-extractor sweep (the orchestrator iterates a fixed list).
    'W15_EffNetB1_optimised': {
        'img_size': 240, 'preprocess': 'efficientnet',
        'model_path': f'{PROJECT_DIR}/efficientnet_optimized/B1/audited/model.keras',
        'feat_layer': 'feat_dense',
    },
}


def get_preprocess_fn(name):
    if name == 'rescale_255':
        return lambda x: x / 255.0
    if name == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    if name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    if name == 'efficientnet':
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input
    if name == 'convnext':
        from tensorflow.keras.applications.convnext import preprocess_input
        return preprocess_input
    if name == 'radimagenet':
        return lambda x: (x / 127.5) - 1.0
    raise ValueError(f'Unknown preprocessing: {name}')


def collect_paths(base_dir, split):
    paths, labels = [], []
    for cls in CLASSES:
        d = f'{base_dir}/{split}/{cls}'
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                paths.append(f'{d}/{fn}')
                labels.append(cls)
    return paths, labels


def _get_custom_objects():
    # ConvNeXt uses a private LayerScale layer that is not auto-registered when
    # a saved .h5 is reloaded.
    try:
        from keras.src.applications.convnext import LayerScale
        return {'LayerScale': LayerScale}
    except ImportError:
        return {}


def build_feature_extractor(model_path, feat_layer, backbone_name=None):
    """Load CNN and cut at the requested feature layer.
    feat_layer=None => use Sequential(cnn.layers[:-1]) (W3 pattern).
    feat_layer='feat_dense' => use get_layer + functional Model (W6..W10 pattern)."""
    from tensorflow.keras.models import load_model, Model, Sequential
    cnn = load_model(model_path, compile=False,
                      custom_objects=_get_custom_objects())
    if feat_layer is None:
        # Everything except the last (softmax) layer.
        extractor = Sequential(cnn.layers[:-1])
        input_shape = cnn.input_shape
        extractor.build(input_shape)
    else:
        layer = cnn.get_layer(feat_layer)
        extractor = Model(cnn.input, layer.output)
    return extractor


def extract_features(backbone_name, base_dir, out_dir):
    """Extract 128-d features for train + test. Cache to features.npz."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    cache = f'{out_dir}/features.npz'
    if os.path.exists(cache):
        print(f'  [cache] loading {cache}')
        d = np.load(cache, allow_pickle=True)
        return d['X_train'], d['y_train'], d['X_test'], d['y_test']

    cfg = BACKBONE_CONFIG[backbone_name]
    img_size = cfg['img_size']
    pi = get_preprocess_fn(cfg['preprocess'])

    print(f'  loading {cfg["model_path"]}')
    extractor = build_feature_extractor(cfg['model_path'], cfg['feat_layer'],
                                         backbone_name=backbone_name)
    print(f'  extractor input {extractor.input_shape} -> output {extractor.output_shape}')

    def extract(paths, tag, batch=32):
        n = len(paths)
        feats = np.zeros((n, extractor.output_shape[-1]), dtype=np.float32)
        t0 = time.time()
        for i in range(0, n, batch):
            batch_paths = paths[i:i + batch]
            imgs = np.stack([
                pi(img_to_array(load_img(p, target_size=(img_size, img_size))))
                for p in batch_paths
            ]).astype(np.float32)
            feats[i:i + len(batch_paths)] = extractor.predict(imgs, verbose=0)
            if (i // batch) % 20 == 0:
                print(f'    [{tag}] {i + len(batch_paths)}/{n}  ({time.time() - t0:.1f}s)')
        return feats

    train_paths, train_labels = collect_paths(base_dir, 'train')
    test_paths,  test_labels  = collect_paths(base_dir, 'test')
    print(f'  Train: {len(train_paths)}   Test: {len(test_paths)}')

    X_train = extract(train_paths, 'train')
    X_test  = extract(test_paths,  'test')

    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_train = np.array([cls_to_idx[l] for l in train_labels], dtype=np.int64)
    y_test  = np.array([cls_to_idx[l] for l in test_labels],  dtype=np.int64)

    np.savez_compressed(cache,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test,   y_test=y_test)
    print(f'  saved features to {cache}   train={X_train.shape} test={X_test.shape}')
    return X_train, y_train, X_test, y_test


def build_classifiers():
    """Exact configuration from W3 (workflow3_hybrid_all_classifiers.ipynb, cell 12).
    Keeping the same hyperparameters is what makes W3 vs W11 directly comparable."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    return {
        'XGBoost':          XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                          random_state=SEED, verbosity=0,
                                          eval_metric='mlogloss'),
        'LightGBM':         LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                           random_state=SEED, verbose=-1),
        'RandomForest':     RandomForestClassifier(n_estimators=500, max_depth=None,
                                                   random_state=SEED, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                       max_depth=4, random_state=SEED),
        'CatBoost':         CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                               random_state=SEED, verbose=False),
        'SVM (RBF)':        SVC(C=10, gamma='scale', kernel='rbf',
                                probability=True, random_state=SEED),
        'MLP':              MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                          alpha=1e-4, max_iter=300, random_state=SEED,
                                          early_stopping=True, validation_fraction=0.15),
    }


def bootstrap_acc_ci(y_true, y_pred, n_iter=1000, alpha=0.05, seed=SEED):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    rng = np.random.RandomState(seed); n = len(y_true); accs = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        accs.append((y_true[idx] == y_pred[idx]).mean() * 100)
    return float(np.percentile(accs, 100 * alpha / 2)), \
           float(np.percentile(accs, 100 * (1 - alpha / 2)))


def train_and_evaluate(X_train, y_train, X_test, y_test, out_dir):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix,
                                 precision_recall_fscore_support)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    # Persist so the SHAP notebook can transform new inputs identically.
    with open(f'{out_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    clfs_dir = f'{out_dir}/classifiers'
    os.makedirs(clfs_dir, exist_ok=True)

    classifiers = build_classifiers()
    rows = []
    per_model_preds = {}
    per_model_perclass = {}

    for name, clf in classifiers.items():
        print(f'\n  === {name} ===', flush=True)
        t0 = time.time()
        clf.fit(Xtr, y_train)
        t_train = time.time() - t0

        # CatBoost (and some others) return 2-D predictions (n, 1); flatten to
        # 1-D so the bootstrap CI's y_true == y_pred does not broadcast into an
        # (n, n) matrix (which silently produced ~1/n_classes accuracy).
        y_pred  = np.asarray(clf.predict(Xte)).ravel()
        try:
            y_proba = clf.predict_proba(Xte)
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro') * 100
        except Exception:
            auc = np.nan

        acc  = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
        f1   = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100
        ci_lo, ci_hi = bootstrap_acc_ci(y_test, y_pred)

        print(f'    acc={acc:.2f}  prec={prec:.2f}  rec={rec:.2f}  f1={f1:.2f}  '
              f'auc={auc:.2f}  ci95=[{ci_lo:.2f}, {ci_hi:.2f}]  t={t_train:.1f}s')

        # Persist the trained classifier so the SHAP notebook loads it directly.
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        with open(f'{clfs_dir}/{safe_name}.pkl', 'wb') as f:
            pickle.dump(clf, f)

        rows.append({
            'classifier': name, 'accuracy': round(acc, 2),
            'precision': round(prec, 2), 'recall': round(rec, 2),
            'f1': round(f1, 2), 'auc': round(auc, 2),
            'ci95_lo': round(ci_lo, 2), 'ci95_hi': round(ci_hi, 2),
            'train_time_s': round(t_train, 2),
        })
        per_model_preds[name] = y_pred
        prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
            y_test, y_pred, labels=range(N_CLASSES), zero_division=0)
        per_model_perclass[name] = pd.DataFrame({
            'class': CLASSES,
            'precision': (prec_c * 100).round(2),
            'recall':    (rec_c * 100).round(2),
            'f1':        (f1_c * 100).round(2),
            'support':   sup_c.astype(int),
        })

    df = pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)
    df.to_csv(f'{out_dir}/comparison.csv', index=False)

    best_name = df.iloc[0]['classifier']
    y_pred_best = per_model_preds[best_name]
    pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_best}) \
      .to_csv(f'{out_dir}/predictions_best.csv', index=False)
    pd.DataFrame(confusion_matrix(y_test, y_pred_best),
                 index=CLASSES, columns=CLASSES) \
      .to_csv(f'{out_dir}/confusion_matrix_best.csv')
    per_model_perclass[best_name].to_csv(f'{out_dir}/perclass_best.csv', index=False)

    with open(f'{out_dir}/best.json', 'w') as f:
        json.dump({'best_classifier': best_name,
                   'best_accuracy':   float(df.iloc[0]['accuracy'])}, f, indent=2)
    print(f'\n  BEST: {best_name}  acc={df.iloc[0]["accuracy"]:.2f}%')
    return df, best_name


def main():
    if len(sys.argv) != 4:
        print('Usage: python train_w11_hybrid.py <backbone_name> <base_dir> <out_root>',
              file=sys.stderr)
        sys.exit(2)
    backbone, base_dir, out_root = sys.argv[1:4]
    if backbone not in BACKBONE_CONFIG:
        print(f'Unknown backbone {backbone!r}. Known: {list(BACKBONE_CONFIG)}',
              file=sys.stderr); sys.exit(2)

    model_path = BACKBONE_CONFIG[backbone]['model_path']
    if not os.path.exists(model_path):
        print(f'Missing pretrained model: {model_path}', file=sys.stderr); sys.exit(2)

    out_dir = f'{out_root}/{backbone}'
    os.makedirs(out_dir, exist_ok=True)

    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    print('=' * 70)
    print(f'W11 hybrid  |  backbone={backbone}  |  partition=audited')
    print('=' * 70)

    X_train, y_train, X_test, y_test = extract_features(backbone, base_dir, out_dir)
    df, best_name = train_and_evaluate(X_train, y_train, X_test, y_test, out_dir)
    print('\n' + df.to_string(index=False))
    sys.exit(0)


if __name__ == '__main__':
    main()
