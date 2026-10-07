"""Restore cached preprocessing state after loading nested legacy Keras H5 models."""
def restore_normalization_state(model):
    """Refresh Normalization tensors from loaded variables without changing weights.

    The legacy H5 loader assigns nested model variables but can leave a child
    Normalization layer's cached mean/variance at its build-time defaults.
    Calling finalize_state recursively restores the stored inference transform.
    This is distinct from BatchNormalization and does not fit or adapt a layer.
    """
    import tensorflow as tf
    restored = []
    def visit(layer, prefix):
        path = prefix + layer.name
        if isinstance(layer, tf.keras.layers.Normalization):
            layer.finalize_state()
            restored.append(path)
        for child in getattr(layer, 'layers', []):
            visit(child, path + '/')
    visit(model, '')
    return restored
