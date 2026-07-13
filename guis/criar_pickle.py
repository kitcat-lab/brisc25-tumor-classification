# -*- coding: utf-8 -*-
"""Script para criar o ficheiro pickle final a partir do modelo .h5.
Corre apenas uma vez, depois de descarregar o modelo do Colab.
"""
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from brisc_classifier import BriscClassifier

print('A carregar o modelo .h5...')
model = load_model('modelo_brisc.h5')

print('A empacotar no BriscClassifier...')
classifier = BriscClassifier(model)

print('A gravar pickle...')
with open('brisc_classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)

tamanho_mb = os.path.getsize('brisc_classifier.pickle') / (1024 * 1024)
print(f'Pronto. brisc_classifier.pickle criado ({tamanho_mb:.1f} MB).')
