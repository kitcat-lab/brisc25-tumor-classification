#!/usr/bin/env python
"""W11 Grad-CAM++ worker -- one backbone per subprocess to keep GPU memory
release deterministic. Generates 4 heatmaps (one per class) using the SAME
example image indices for every backbone, so the resulting grid is a
side-by-side visual comparison of what each pretraining attends to on
identical inputs.

Usage:
    python w11_gradcam.py <backbone_name> <base_dir> <out_root>

Writes:
    <out_root>/<backbone>/gradcam/<class>.png       overlay (image + heatmap)
    <out_root>/<backbone>/gradcam/<class>_raw.npy   raw heatmap array

Grad-CAM++ formula: Chattopadhay et al. 2018. Uses only the tape-recorded
first-order gradient (dY_c/dA), the numerator/denominator weights are the
standard Grad-CAM++ closed-form on that gradient. Good enough for the
comparative-visualisation purpose here; if you need the double-derivative
variant, replace grads with tape.gradient(grads, conv_out).
"""
import os, sys, numpy as np

# Last-conv layer name for each backbone. Determined by inspecting each
# keras.applications architecture; verified with model.summary() during
# development. If a backbone version changes these names, update here.
LAST_CONV = {
    'W3_CNN_scratch':             None,   # detected at runtime (last Conv2D)
    'W6_ResNet50_ImageNet':       'conv5_block3_out',
    'W7_VGG16_ImageNet':          'block5_conv3',
    'W8_EfficientNetB2_ImageNet': 'top_conv',
    'W9_ConvNeXtTiny_ImageNet':   None,   # detected at runtime (name varies)
    'W10_RadImageNet_ResNet50':   'conv5_block3_out',
    'W15_EffNetB1_optimised':     'top_conv',
}

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Use the same 4 example images across all backbones (first .jpg alphabetically
# in each class' test folder). The consistency is what makes the resulting
# figure a fair side-by-side.
EXAMPLE_INDEX = 0


def pick_example(base_dir, cls):
    """First image (alphabetically) of the requested test class."""
    d = f'{base_dir}/test/{cls}'
    files = sorted(f for f in os.listdir(d)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if not files:
        raise RuntimeError(f'No test images in {d}')
    return f'{d}/{files[EXAMPLE_INDEX]}'


def find_last_conv(model):
    """Walk the model in reverse and return the first Conv2D layer name.
    Handles the case where the backbone is a nested Model."""
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.models import Model
    for L in reversed(model.layers):
        if isinstance(L, Conv2D):
            return L.name
        if isinstance(L, Model):
            for LL in reversed(L.layers):
                if isinstance(LL, Conv2D):
                    return LL.name
    raise RuntimeError('No Conv2D layer found')


def _build_grad_model(model, last_conv_name):
    """Return a functional model inp -> [conv_activation, predictions] that works
    for both (a) flat/Sequential models where the conv is a top-level layer, and
    (b) transfer-learning models where the conv is nested inside a backbone
    sub-model. Keras 3 cannot build Model(model.inputs, [nested.output, ...])
    because the nested layer's output lives in the sub-model's own graph, so we
    reconstruct the forward pass explicitly."""
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model

    layers = [L for L in model.layers if L.__class__.__name__ != 'InputLayer']

    # Is the conv a top-level layer, or nested in a sub-model?
    top_names = {L.name for L in layers}
    inp = Input(shape=model.input_shape[1:])

    if last_conv_name in top_names:
        # Case A: rebuild forward pass, capturing the conv output.
        x = inp; conv_out = None
        for L in layers:
            x = L(x)
            if L.name == last_conv_name:
                conv_out = x
        return Model(inp, [conv_out, x])

    # Case B: conv is inside a nested sub-model.
    base_idx = None
    for i, L in enumerate(layers):
        if hasattr(L, 'layers') and any(ll.name == last_conv_name for ll in L.layers):
            base_idx = i; break
    if base_idx is None:
        raise RuntimeError(f'Layer {last_conv_name!r} not found in model')
    base = layers[base_idx]
    base_grad = Model(base.input, [base.get_layer(last_conv_name).output, base.output])
    x = inp
    for L in layers[:base_idx]:
        x = L(x)
    conv_out, base_out = base_grad(x)
    x = base_out
    for L in layers[base_idx + 1:]:
        x = L(x)
    return Model(inp, [conv_out, x])


def gradcam_plus_plus(model, img_batch, class_idx, last_conv_name):
    """Compute Grad-CAM++ heatmap for the given class. Robust to Sequential and
    nested-backbone topologies (see _build_grad_model)."""
    import tensorflow as tf

    grad_model = _build_grad_model(model, last_conv_name)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)  # (1, H, W, C)
    # Grad-CAM++ weights (closed-form using only first-order gradients):
    grads_sq   = grads * grads
    grads_cube = grads_sq * grads
    sum_conv   = tf.reduce_sum(conv_out, axis=(0, 1, 2))          # (C,)
    alpha_num  = grads_sq
    alpha_den  = 2.0 * grads_sq + sum_conv[None, None, None, :] * grads_cube
    alpha_den  = tf.where(alpha_den != 0, alpha_den, tf.ones_like(alpha_den))
    alphas     = alpha_num / alpha_den                            # (1, H, W, C)
    weights    = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1, 2))  # (C,)

    cam = tf.reduce_sum(conv_out[0] * weights[None, None, :], axis=-1)   # (H, W)
    cam = tf.nn.relu(cam).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam


def get_preprocess_fn(name):
    if name == 'rescale_255':
        return lambda x: x / 255.0
    if name in ('resnet50', 'W6', 'W10'):
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


def main():
    if len(sys.argv) != 4:
        print('Usage: python w11_gradcam.py <backbone_name> <base_dir> <out_root>',
              file=sys.stderr)
        sys.exit(2)
    backbone, base_dir, out_root = sys.argv[1:4]

    # Import config (single source of truth) from the training script.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_w11_hybrid import BACKBONE_CONFIG
    if backbone not in BACKBONE_CONFIG:
        print(f'Unknown backbone {backbone!r}', file=sys.stderr); sys.exit(2)
    cfg = BACKBONE_CONFIG[backbone]

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import matplotlib.pyplot as plt
    from matplotlib import cm

    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    out_dir = f'{out_root}/{backbone}/gradcam'
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print(f'W11 Grad-CAM++  |  {backbone}')
    print('=' * 70)

    # ConvNeXt is unrecoverable from .h5 in current Keras 3 (LayerScale not
    # registered + positional-arg deserialisation bug). We rebuild the exact
    # W9 architecture and load only the weights, which sidesteps the whole
    # architecture-serialisation path.
    print(f'  loading {cfg["model_path"]}')
    if backbone == 'W9_ConvNeXtTiny_ImageNet':
        from tensorflow.keras import layers, Input
        from tensorflow.keras.models import Model as _KModel
        from tensorflow.keras.applications import ConvNeXtTiny
        SEED = 42; N_CLASSES = len(CLASSES)
        backbone_net = ConvNeXtTiny(weights=None, include_top=False,
                                     input_shape=(cfg['img_size'], cfg['img_size'], 3))
        inputs = Input(shape=(cfg['img_size'], cfg['img_size'], 3))
        x = backbone_net(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3, seed=SEED)(x)
        x = layers.Dense(128, activation='relu', name='feat_dense')(x)
        x = layers.Dropout(0.3, seed=SEED)(x)
        out = layers.Dense(N_CLASSES, activation='softmax')(x)
        model = _KModel(inputs, out)
        model.load_weights(cfg['model_path'])
    else:
        try:
            from keras.src.applications.convnext import LayerScale
            custom_objects = {'LayerScale': LayerScale}
        except ImportError:
            custom_objects = {}
        model = load_model(cfg['model_path'], compile=False,
                            custom_objects=custom_objects)

    last_conv = LAST_CONV.get(backbone) or find_last_conv(model)
    print(f'  last conv layer: {last_conv}')

    img_size = cfg['img_size']
    pi = get_preprocess_fn(cfg['preprocess'])

    for cls_idx, cls in enumerate(CLASSES):
        img_path = pick_example(base_dir, cls)
        pil = load_img(img_path, target_size=(img_size, img_size))
        raw = img_to_array(pil).astype(np.float32)       # [0, 255]
        arr = pi(raw.copy())                             # backbone-specific
        cam = gradcam_plus_plus(model, np.expand_dims(arr, 0),
                                cls_idx, last_conv)

        # Upsample cam to img_size (nearest-neighbour is fine for viz)
        import cv2  # opencv
        cam_up = cv2.resize(cam, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        # Overlay: raw grayscale image + jet heatmap
        disp = raw / 255.0
        if disp.ndim == 3 and disp.shape[-1] == 3:
            disp_gray = disp.mean(axis=-1)
        else:
            disp_gray = disp
        heatmap = cm.jet(cam_up)[..., :3]
        overlay = 0.55 * heatmap + 0.45 * np.stack([disp_gray] * 3, axis=-1)
        overlay = np.clip(overlay, 0, 1)

        # Save
        np.save(f'{out_dir}/{cls}_raw.npy', cam_up)
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        ax[0].imshow(disp_gray, cmap='gray'); ax[0].set_title(f'{cls}\n({os.path.basename(img_path)})',
                                                              fontsize=8); ax[0].axis('off')
        ax[1].imshow(cam_up, cmap='jet');    ax[1].set_title('Grad-CAM++', fontsize=8); ax[1].axis('off')
        ax[2].imshow(overlay);               ax[2].set_title('overlay',    fontsize=8); ax[2].axis('off')
        plt.suptitle(f'{backbone}  /  {cls}', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{cls}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f'  {cls:12s}  saved  (cam min={cam.min():.3f} max={cam.max():.3f})')

    print('Done.')
    sys.exit(0)


if __name__ == '__main__':
    main()
