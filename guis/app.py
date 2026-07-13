# -*- coding: utf-8 -*-
"""PyQt6 BRISC CNN brain MRI classifier.

Loads a pickled BriscClassifier and classifies MRI images into
4 classes: glioma, meningioma, no_tumor, pituitary.
"""
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PyQt6 import uic, QtWidgets
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from brisc_classifier import BriscClassifier  # needed for pickle


CLASSES_PT = {'glioma': 'Glioma', 'meningioma': 'Meningioma',
              'no_tumor': 'Sem Tumor', 'pituitary': 'Pituitary'}

PICKLE_FILE = 'brisc_classifier.pickle'

MODEL_NAME = 'CNN from scratch'
MODEL_ACC  = 93.81


# ── FUNCTIONS ────────────────────────────────────────────────────────────────

def funcao_carregar_modelo():
    """Carrega o classificador a partir do pickle."""
    if not os.path.exists(PICKLE_FILE):
        formulario.lStatus.setText(
            f'[!] Ficheiro {PICKLE_FILE} nao encontrado nesta pasta.')
        return None
    with open(PICKLE_FILE, 'rb') as f:
        clf = pickle.load(f)
    formulario.lStatus.setText('[ok] Modelo carregado. Pronto a classificar.')
    return clf


def funcao_carregar_imagem():
    """Open a file dialog to select an MRI image."""
    global imagem_path
    path, _ = QFileDialog.getOpenFileName(
        None, 'Selecionar imagem MRI', '',
        'Imagens (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
    if not path:
        return
    imagem_path = path
    pixmap = QPixmap(path).scaled(
        340, 320,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation)
    formulario.lImagem.setPixmap(pixmap)
    formulario.pbClassificar.setEnabled(classifier is not None)
    formulario.lStatus.setText(f'[+] Imagem carregada: {os.path.basename(path)}')
    funcao_resetar()


def funcao_classificar():
    """Aplica o modelo CNN e mostra o resultado."""
    if not imagem_path or classifier is None:
        return
    try:
        formulario.pbClassificar.setEnabled(False)
        formulario.lStatus.setText('[...] A classificar...')
        QApplication.processEvents()

        # returns {class: probability}
        resultado = classifier.predict(imagem_path)

        melhor_classe = max(resultado, key=resultado.get)
        melhor_prob   = resultado[melhor_classe]
        nome_pt = CLASSES_PT[melhor_classe]
        pct = melhor_prob * 100

        formulario.lDiagnostico.setText(nome_pt)
        formulario.lDiagnostico.setStyleSheet(
            'font-size: 22px; font-weight: bold; color: #f5f5f5; '
            'background: transparent; font-family: Segoe UI;')
        formulario.lConfianca.setText(f'{pct:.1f}%')
        formulario.lConfianca.setStyleSheet(
            'background-color: #2e2e3a; color: #e07090; border-radius: 8px; '
            'font-size: 20px; font-weight: bold; font-family: Segoe UI;')

        bars = {
            'glioma':     (formulario.barGlioma,     formulario.lGliomaPct),
            'meningioma': (formulario.barMeningioma, formulario.lMeningiomaPct),
            'no_tumor':   (formulario.barNoTumor,    formulario.lNoTumorPct),
            'pituitary':  (formulario.barPituitary,  formulario.lPituitaryPct),
        }
        for cls, prob in resultado.items():
            bar, lbl = bars[cls]
            bar.setValue(int(prob * 100))
            lbl.setText(f'{prob*100:.1f}%')

        formulario.lStatus.setText(
            f'[*] Resultado: {nome_pt}  -  Confianca: {pct:.1f}%')
    except Exception as e:
        QMessageBox.critical(None, 'Erro na classificacao', str(e))
        formulario.lStatus.setText(f'[!] Erro: {e}')
    finally:
        formulario.pbClassificar.setEnabled(True)


def funcao_limpar():
    """Limpa a imagem e os resultados."""
    global imagem_path
    imagem_path = None
    formulario.lImagem.clear()
    formulario.lImagem.setText('Nenhuma imagem carregada')
    formulario.pbClassificar.setEnabled(False)
    funcao_resetar()
    formulario.lStatus.setText(
        '[ ] Pronto  -  Carregue uma imagem MRI para comecar')


def funcao_resetar():
    """Repoe o painel de resultados para o estado inicial."""
    formulario.lDiagnostico.setText('Aguarda imagem')
    formulario.lDiagnostico.setStyleSheet(
        'color: #7070a0; font-size: 22px; font-weight: bold; '
        'font-family: Segoe UI; background: transparent;')
    formulario.lConfianca.setText('0%')
    formulario.lConfianca.setStyleSheet(
        'background-color: #2e2e3a; color: #7070a0; border-radius: 8px; '
        'font-size: 20px; font-weight: bold; font-family: Segoe UI;')
    for bar in [formulario.barGlioma, formulario.barMeningioma,
                formulario.barNoTumor, formulario.barPituitary]:
        bar.setValue(0)
    for lbl in [formulario.lGliomaPct, formulario.lMeningiomaPct,
                formulario.lNoTumorPct, formulario.lPituitaryPct]:
        lbl.setText('0%')


# ── ARRANQUE ──────────────────────────────────────────────────────────────────
imagem_path = None

app = QtWidgets.QApplication([])
app.setStyle('Fusion')

formulario = uic.loadUi('main_window.ui')

classifier = funcao_carregar_modelo()

formulario.setWindowTitle(f'BRISC - {MODEL_NAME} - Classificador de MRI Cerebral')
formulario.lBadge.setText(f'{MODEL_NAME}: Acc. {MODEL_ACC}%')

formulario.pbCarregar.clicked.connect(funcao_carregar_imagem)
formulario.pbLimpar.clicked.connect(funcao_limpar)
formulario.pbClassificar.clicked.connect(funcao_classificar)

formulario.show()
app.exec()
