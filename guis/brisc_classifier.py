# -*- coding: utf-8 -*-
"""BriscClassifier - wraps a Keras CNN model in a single pickle."""
import os
import io
import tempfile
import numpy as np


class BriscClassifier:
    CLASSES  = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    IMG_SIZE = 128

    def __init__(self, model):
        self.model = model

    def predict(self, image_path):
        """Classify an MRI image and return {class: probability}."""
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        img = load_img(image_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))
        arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        probs = self.model.predict(arr, verbose=0)[0]
        return dict(zip(self.CLASSES, probs.tolist()))

    # -- Serialisation (Pickle) -------------------------------------------------
    def __getstate__(self):
        # Saves the Keras model as bytes inside the pickle
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()
        self.model.save(tmp.name)
        with open(tmp.name, 'rb') as f:
            model_bytes = f.read()
        os.unlink(tmp.name)
        return {'model_bytes': model_bytes}

    def __setstate__(self, state):
        from tensorflow.keras.models import load_model
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.write(state['model_bytes'])
        tmp.close()
        self.model = load_model(tmp.name)
        os.unlink(tmp.name)
