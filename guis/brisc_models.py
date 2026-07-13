# -*- coding: utf-8 -*-
"""Unified, self-contained model wrappers for the BRISC multi-model GUI.

Every model exposes the SAME interface so the GUI can treat them uniformly:

    m.name        -> short display name (e.g. 'EfficientNetB1 (optimised)')
    m.family      -> approach family (see FAMILIES) for grouping/colour
    m.accuracy    -> audited test accuracy (%) shown next to the name
    m.predict(path) -> {'glioma': p, 'meningioma': p, 'no_tumor': p, 'pituitary': p}

Two concrete wrappers cover every BRISC model:

    SoftmaxModel  - a Keras classifier that outputs the 4-way softmax directly.
                    Covers the from-scratch CNN and every transfer/optimised
                    backbone (per-model input size + preprocessing).
    HybridModel   - a Keras backbone used as a 128-d feature extractor, followed
                    by a fitted StandardScaler and an sklearn classifier
                    (the W11 champion is VGG16 + LightGBM = 98.67%).

Both pickle themselves fully (Keras weights embedded as bytes + sklearn objects),
so each model ships as one portable `.pickle` with no external files -- the same
pattern as the original `brisc_classifier.pickle`.

TensorFlow is imported lazily (inside predict / unpickle) so the GUI starts fast
and only pays the TF cost when a model is actually run.
"""
import os
import tempfile
import numpy as np

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Approach families (drives grouping and colour in the GUI). Kept in sync with
# the article's taxonomy.
FAMILIES = ['CNN scratch', 'Transfer (softmax)', 'Hybrid', 'Medical pretrain']


# ── image preprocessing ───────────────────────────────────────────────────────
# Stored as a *name* (portable string) and reconstructed at predict time, so a
# pickle never has to embed a lambda or a TF symbol. Names match
# train_w11_hybrid.get_preprocess_fn.
def preprocess_fn(name):
    """Return a function mapping a raw [0,255] float array to the model input."""
    if name == 'rescale_255':
        return lambda x: x / 255.0
    if name == 'radimagenet':
        return lambda x: (x / 127.5) - 1.0
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
    raise ValueError(f'Unknown preprocessing: {name!r}')


def _load_image(image_path, img_size, preprocess):
    """Load an MRI, resize to the model's native size and apply preprocessing.
    Returns a (1, H, W, 3) batch ready for the network."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path, target_size=(img_size, img_size))
    arr = img_to_array(img).astype('float32')          # [0, 255]
    arr = preprocess_fn(preprocess)(arr.copy())
    return np.expand_dims(arr, axis=0)


# ── Keras (de)serialisation to bytes ──────────────────────────────────────────
# Use the .keras format (Keras 3 native) rather than .h5: it round-trips every
# architecture in the study, including ConvNeXt/EfficientNet, whereas .h5 breaks
# on ConvNeXt's LayerScale. See [[brisc-artefact-consistency]].
def _keras_to_bytes(model):
    tmp = tempfile.NamedTemporaryFile(suffix='.keras', delete=False)
    tmp.close()
    model.save(tmp.name)
    with open(tmp.name, 'rb') as f:
        data = f.read()
    os.unlink(tmp.name)
    return data


def _bytes_to_keras(data):
    from tensorflow.keras.models import load_model
    tmp = tempfile.NamedTemporaryFile(suffix='.keras', delete=False)
    tmp.write(data)
    tmp.close()
    try:
        model = load_model(tmp.name, compile=False)
    finally:
        os.unlink(tmp.name)
    return model


# ── model wrappers ────────────────────────────────────────────────────────────
class SoftmaxModel:
    """A Keras model whose final layer already emits the 4-way softmax."""

    def __init__(self, model, img_size, preprocess, name,
                 family='Transfer (softmax)', accuracy=None):
        self.model = model
        self.img_size = int(img_size)
        self.preprocess = preprocess
        self.name = name
        self.family = family
        self.accuracy = accuracy

    def predict(self, image_path):
        batch = _load_image(image_path, self.img_size, self.preprocess)
        probs = self.model.predict(batch, verbose=0)[0]
        return {c: float(p) for c, p in zip(CLASSES, probs)}

    def __getstate__(self):
        return {'model_bytes': _keras_to_bytes(self.model),
                'img_size': self.img_size, 'preprocess': self.preprocess,
                'name': self.name, 'family': self.family, 'accuracy': self.accuracy}

    def __setstate__(self, s):
        self.model = _bytes_to_keras(s['model_bytes'])
        self.img_size = s['img_size']; self.preprocess = s['preprocess']
        self.name = s['name']; self.family = s['family']; self.accuracy = s['accuracy']


class HybridModel:
    """Keras backbone (128-d feature extractor) + StandardScaler + sklearn clf.

    Reproduces the W11 hybrid inference path exactly: preprocess -> extract 128-d
    feature -> scaler.transform -> classifier.predict_proba, then map the
    classifier's integer classes back to the BRISC label order."""

    def __init__(self, extractor, scaler, clf, img_size, preprocess, name,
                 clf_name, family='Hybrid', accuracy=None):
        self.extractor = extractor
        self.scaler = scaler
        self.clf = clf
        self.img_size = int(img_size)
        self.preprocess = preprocess
        self.name = name
        self.clf_name = clf_name
        self.family = family
        self.accuracy = accuracy

    def _proba(self, feat_scaled):
        """Per-class probabilities as a (n_classes,) vector in classifier order.
        Prefers predict_proba; falls back to a softmax over decision_function,
        then to a one-hot of the hard prediction."""
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(feat_scaled)[0]
        if hasattr(self.clf, 'decision_function'):
            d = np.atleast_2d(self.clf.decision_function(feat_scaled))[0]
            e = np.exp(d - d.max())
            return e / e.sum()
        pred = int(self.clf.predict(feat_scaled)[0])
        oh = np.zeros(len(self.clf.classes_))
        oh[list(self.clf.classes_).index(pred)] = 1.0
        return oh

    def predict(self, image_path):
        batch = _load_image(image_path, self.img_size, self.preprocess)
        feat = self.extractor.predict(batch, verbose=0)          # (1, 128)
        feat = self.scaler.transform(feat)
        proba = self._proba(feat)
        # classifier.classes_ holds the integer label ids (0..3) in CLASSES order.
        out = {c: 0.0 for c in CLASSES}
        for cls_id, p in zip(self.clf.classes_, proba):
            out[CLASSES[int(cls_id)]] = float(p)
        return out

    def __getstate__(self):
        import pickle
        return {'extractor_bytes': _keras_to_bytes(self.extractor),
                'scaler': pickle.dumps(self.scaler), 'clf': pickle.dumps(self.clf),
                'img_size': self.img_size, 'preprocess': self.preprocess,
                'name': self.name, 'clf_name': self.clf_name,
                'family': self.family, 'accuracy': self.accuracy}

    def __setstate__(self, s):
        import pickle
        self.extractor = _bytes_to_keras(s['extractor_bytes'])
        self.scaler = pickle.loads(s['scaler']); self.clf = pickle.loads(s['clf'])
        self.img_size = s['img_size']; self.preprocess = s['preprocess']
        self.name = s['name']; self.clf_name = s['clf_name']
        self.family = s['family']; self.accuracy = s['accuracy']


# ── consensus across selected models ──────────────────────────────────────────
def consensus(predictions):
    """Combine several models' {class: prob} dicts into one consensus dict.

    Default rule: mean of the per-class probabilities (soft voting). This treats
    every selected model equally; it is robust and needs no thresholds. Returns
    the same {class: prob} shape, re-normalised to sum to 1.

    Alternatives worth considering (easy to swap in): accuracy-weighted mean
    (weight each model by m.accuracy), or hard majority vote on argmax.
    """
    if not predictions:
        return {c: 0.0 for c in CLASSES}
    mean = {c: float(np.mean([p[c] for p in predictions])) for c in CLASSES}
    total = sum(mean.values()) or 1.0
    return {c: v / total for c, v in mean.items()}
