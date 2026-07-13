# -*- coding: utf-8 -*-
"""Debug script - checks if the model is working as in Colab.

Usage:  python debug_modelo.py <image_path>
"""
import os
import sys
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from brisc_classifier import BriscClassifier

if len(sys.argv) < 2:
    print('Usage:  python debug_modelo.py <image_path>')
    sys.exit(1)

img_path = sys.argv[1]

print('\n' + '='*60)
print('DEBUG: BRISC Classifier')
print('='*60)

# 1. Model info
with open('brisc_classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

print(f'\n[1] Model loaded')
print(f'    Input shape  : {clf.model.input_shape}')
print(f'    Output shape : {clf.model.output_shape}')
print(f'    Classes      : {clf.CLASSES}')
print(f'    IMG_SIZE     : {clf.IMG_SIZE}')

# 2. Load image
print(f'\n[2] Image: {img_path}')
img = load_img(img_path, target_size=(clf.IMG_SIZE, clf.IMG_SIZE))
arr = img_to_array(img) / 255.0
print(f'    Shape pre-batch : {arr.shape}')
print(f'    Values min/max  : {arr.min():.3f} / {arr.max():.3f}')
print(f'    Mean value      : {arr.mean():.3f}')

# 3. Raw prediction
arr_batch = np.expand_dims(arr, axis=0)
probs = clf.model.predict(arr_batch, verbose=0)[0]

print(f'\n[3] Raw probabilities (CNN output):')
for i, (cls, p) in enumerate(zip(clf.CLASSES, probs)):
    barra = '#' * int(p * 40)
    print(f'    [{i}] {cls:15s} {p*100:6.2f}%  {barra}')

best_idx = int(np.argmax(probs))
print(f'\n    --> Prediction: index {best_idx} = {clf.CLASSES[best_idx]}')
print(f'    --> Sum of probabilities = {probs.sum():.4f} (should be ~1.0)')

# 4. Prediction via pickle
print(f'\n[4] Via clf.predict():')
result = clf.predict(img_path)
for cls, p in result.items():
    print(f'    {cls:15s} {p*100:6.2f}%')

print('\n' + '='*60)
print('If "no_tumor" always appears with very high probability')
print('(>80%) even in tumor images, the problem is with the model.')
print('If the probabilities are ~25% each, it is uncertainty/preprocessing.')
print('='*60 + '\n')
