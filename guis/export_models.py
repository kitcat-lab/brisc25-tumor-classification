# -*- coding: utf-8 -*-
"""Build the curated set of self-contained model pickles for the BRISC GUI.

Run this ONCE in the TensorFlow environment (the same kernel that ran the
workflows). It reads the trained artefacts, wraps each model in a SoftmaxModel /
HybridModel (see brisc_models.py) and writes:

    models/<slug>.pickle        one portable pickle per model (weights embedded)
    models_registry.json        lightweight index the GUI reads at startup

Curated set (edit CURATED below to change which models ship in the GUI). The
default four span every approach family in the article:

    CNN scratch              modelo_brisc.h5                 93.81%   baseline
    EfficientNetB1 optimised efficientnet_optimized/B1       98.82%   best single
    VGG16 + LightGBM hybrid  W7 backbone + W11 champion clf  98.67%   best hybrid
    RadImageNet ResNet50     W10 backbone                    72.12%   contrast

Usage:
    python export_models.py
"""
import os, io, json, pickle
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

HERE = os.path.dirname(os.path.abspath(__file__))
LEAKAGE = f'{HERE}/leakage_backbones'
HYBRID = f'{HERE}/w11_hybrid'
OPT = f'{HERE}/efficientnet_optimized'
OUT_DIR = f'{HERE}/models'

# --- curated model definitions -------------------------------------------------
# type 'softmax' : whole Keras model, outputs the 4-way softmax.
# type 'hybrid'  : Keras backbone (cut at feat_layer) + scaler + sklearn clf.
CURATED = [
    dict(slug='cnn_scratch', type='softmax',
         name='CNN from scratch', family='CNN scratch', accuracy=93.81,
         model_path=f'{HERE}/modelo_brisc.h5', img_size=128, preprocess='rescale_255'),

    dict(slug='effnetb1_optimised', type='softmax',
         name='EfficientNetB1 (optimised)', family='Transfer (softmax)', accuracy=98.82,
         model_path=f'{OPT}/B1/audited/model.keras', img_size=240, preprocess='efficientnet'),

    dict(slug='vgg16_lightgbm', type='hybrid',
         name='VGG16 + LightGBM', family='Hybrid', accuracy=98.67, clf_name='LightGBM',
         backbone_path=f'{LEAKAGE}/W7_VGG16_ImageNet/audited/model.h5', feat_layer='feat_dense',
         scaler_path=f'{HYBRID}/W7_VGG16_ImageNet/scaler.pkl',
         clf_path=f'{HYBRID}/W7_VGG16_ImageNet/classifiers/LightGBM.pkl',
         img_size=224, preprocess='vgg16'),

    dict(slug='radimagenet', type='softmax',
         name='RadImageNet ResNet50', family='Medical pretrain', accuracy=72.12,
         model_path=f'{LEAKAGE}/W10_RadImageNet_ResNet50/audited/model.h5',
         img_size=224, preprocess='radimagenet'),
]


def _missing(paths):
    return [p for p in paths if not os.path.exists(p)]


def build_softmax(spec):
    from tensorflow.keras.models import load_model
    from brisc_models import SoftmaxModel
    try:
        from train_w11_hybrid import _get_custom_objects
        custom = _get_custom_objects()
    except Exception:
        custom = {}
    model = load_model(spec['model_path'], compile=False, custom_objects=custom)
    return SoftmaxModel(model, spec['img_size'], spec['preprocess'],
                        spec['name'], spec['family'], spec['accuracy'])


def build_hybrid(spec):
    from brisc_models import HybridModel
    from train_w11_hybrid import build_feature_extractor
    extractor = build_feature_extractor(spec['backbone_path'], spec['feat_layer'])
    with open(spec['scaler_path'], 'rb') as f:
        scaler = pickle.load(f)
    with open(spec['clf_path'], 'rb') as f:
        clf = pickle.load(f)
    return HybridModel(extractor, scaler, clf, spec['img_size'], spec['preprocess'],
                       spec['name'], spec['clf_name'], spec['family'], spec['accuracy'])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    registry = []
    for spec in CURATED:
        print('=' * 64)
        print(f"  {spec['slug']:20s}  {spec['name']}  ({spec['accuracy']}%)")
        needed = ([spec['model_path']] if spec['type'] == 'softmax'
                  else [spec['backbone_path'], spec['scaler_path'], spec['clf_path']])
        miss = _missing(needed)
        if miss:
            print('  [skip] missing artefacts:')
            for m in miss:
                print('        ', m)
            continue
        obj = build_softmax(spec) if spec['type'] == 'softmax' else build_hybrid(spec)

        # sanity: interface returns a full 4-class distribution
        out_path = f"{OUT_DIR}/{spec['slug']}.pickle"
        with open(out_path, 'wb') as f:
            pickle.dump(obj, f)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f'  saved -> {out_path}  ({size_mb:.1f} MB)')
        registry.append({'slug': spec['slug'], 'name': spec['name'],
                         'family': spec['family'], 'accuracy': spec['accuracy'],
                         'clf_name': spec.get('clf_name'),
                         'file': f"models/{spec['slug']}.pickle"})

    with io.open(f'{HERE}/models_registry.json', 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print('=' * 64)
    print(f'wrote models_registry.json with {len(registry)} model(s).')
    if len(registry) < len(CURATED):
        print('NOTE: some models were skipped (missing artefacts above).')


if __name__ == '__main__':
    main()
