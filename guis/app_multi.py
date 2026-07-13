# -*- coding: utf-8 -*-
"""BRISC - Multi-model MRI comparison GUI (PyQt6).

Load one brain-MRI, tick any subset of the curated models, and see a side-by-side
table of per-class probabilities for every selected model plus a soft-voting
consensus row. Models are described by `models_registry.json` and loaded lazily
from `models/<slug>.pickle` (built by export_models.py).

This is the multi-model successor to the single-model app.py; it does not modify
or replace that file. Run:  python app_multi.py
"""
import os, io, json, pickle

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QAbstractItemView)
from PyQt6.QtGui import QPixmap, QColor, QFont, QIcon
from PyQt6.QtCore import Qt, QSize

try:
    import qtawesome as qta          # vector Font-Awesome / Material icons
except ImportError:
    qta = None


def icon(name, color=None):
    """Themed vector icon; returns an empty icon if qtawesome is unavailable
    or the name is unknown, so the GUI still runs without icons."""
    if not qta:
        return QIcon()
    try:
        return qta.icon(name, color=color or INK)
    except Exception:
        return QIcon()

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, 'models_registry.json')

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_LABEL = {'glioma': 'Glioma', 'meningioma': 'Meningioma',
               'no_tumor': 'No tumour', 'pituitary': 'Pituitary'}

# modern grey + pink theme (professional)
INK      = '#20242E'   # primary text  (near-black)
MUTED    = '#5C606B'   # secondary text
BORDER   = '#9A9EAA'   # visible grey border
SURFACE  = '#FFFFFF'   # cards / inputs (float on the grey canvas)
CANVAS   = '#CDCFD6'   # window bg, a darker neutral grey with contrast
ACCENT   = '#C6497B'   # pink accent (primary button, consensus, highlight)
ACCENT_MD= '#D5709A'   # pink hover / mid
HILITE   = '#F3D8E4'   # soft pink for the argmax cell

# family -> muted accent colour (distinct but coordinated with grey + pink)
FAMILY_COLOR = {'CNN scratch': '#6B7280',        # grey
                'Transfer (softmax)': '#5566A0', # dusty indigo
                'Hybrid': '#B0446F',             # rose (champion)
                'Medical pretrain': '#B37A46'}   # muted amber

QSS = f"""
QWidget {{ background: {CANVAS}; color: {INK};
           font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; }}
QLabel#section {{ color: {MUTED}; font-weight: 700; font-size: 11px;
                  letter-spacing: 1px; margin-top: 6px; }}
QPushButton {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 6px 12px; color: {INK}; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: #ffffff; border: none;
                       font-weight: 700; padding: 10px; font-size: 13px; }}
QPushButton#primary:hover {{ background: {ACCENT_MD}; }}
QPushButton#primary:disabled {{ background: #D7B7C6; color: #F6EAF0; }}
QTableWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
                gridline-color: {BORDER}; }}
QHeaderView::section {{ background: {CANVAS}; color: {MUTED}; border: none;
                        border-bottom: 1px solid {BORDER}; padding: 7px; font-weight: 700; }}
QTableWidget::item {{ padding: 4px; }}
"""


class BriscMultiApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BRISC - Multi-Model Brain MRI Comparison')
        self.setWindowIcon(icon('fa5s.brain', ACCENT))
        self.resize(1080, 640)
        self.image_path = None
        self.registry = self._load_registry()
        self._cache = {}          # slug -> unpickled model (lazy)
        self._build_ui()

    # ── data ──────────────────────────────────────────────────────────────
    def _load_registry(self):
        if not os.path.exists(REGISTRY):
            return []
        with io.open(REGISTRY, encoding='utf-8') as f:
            return json.load(f)

    def _get_model(self, slug):
        """Lazily unpickle a model the first time it is needed (this triggers the
        TensorFlow load, so it is slow only once per model per session)."""
        if slug not in self._cache:
            entry = next(e for e in self.registry if e['slug'] == slug)
            with open(os.path.join(HERE, entry['file']), 'rb') as f:
                self._cache[slug] = pickle.load(f)
        return self._cache[slug]

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)

        # -- left column: image + model checklist + actions --
        left = QVBoxLayout()
        self.lImage = QLabel('No image loaded')
        self.lImage.setFixedSize(340, 320)
        self.lImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lImage.setStyleSheet(
            f'border: 1px solid {BORDER}; border-radius: 8px; color: {MUTED}; '
            f'background: {SURFACE};')
        left.addWidget(self.lImage)

        btnRow = QHBoxLayout()
        self.pbLoad = QPushButton('  Load image')
        self.pbLoad.setIcon(icon('fa5s.folder-open', MUTED))
        self.pbLoad.setIconSize(QSize(16, 16))
        self.pbLoad.clicked.connect(self.load_image)
        self.pbClear = QPushButton('  Clear')
        self.pbClear.setIcon(icon('fa5s.eraser', MUTED))
        self.pbClear.setIconSize(QSize(16, 16))
        self.pbClear.clicked.connect(self.clear_all)
        btnRow.addWidget(self.pbLoad); btnRow.addWidget(self.pbClear)
        left.addLayout(btnRow)

        left.addWidget(self._section_label('Models to compare'))
        # quick select-all / clear buttons
        selRow = QHBoxLayout()
        pbAll = QPushButton('  All'); pbAll.setIcon(icon('fa5s.check-double', MUTED))
        pbAll.clicked.connect(lambda: self._set_all(True))
        pbNone = QPushButton('  None'); pbNone.setIcon(icon('fa5s.square', MUTED))
        pbNone.clicked.connect(lambda: self._set_all(False))
        selRow.addWidget(pbAll); selRow.addWidget(pbNone); selRow.addStretch()
        left.addLayout(selRow)
        # one large, clearly clickable checkbox per model (multi-select)
        self.model_checks = {}                       # slug -> QCheckBox
        checksLayout = QVBoxLayout()
        checksLayout.setSpacing(2)
        self._populate_models(checksLayout)
        checksFrame = QFrame()
        checksFrame.setLayout(checksLayout)
        checksFrame.setStyleSheet(
            f'QFrame {{ border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE}; }}')
        left.addWidget(checksFrame, 1)

        self.pbClassify = QPushButton('  Classify')
        self.pbClassify.setObjectName('primary')
        self.pbClassify.setIcon(icon('fa5s.brain', 'white'))
        self.pbClassify.setIconSize(QSize(18, 18))
        self.pbClassify.setEnabled(False)
        self.pbClassify.clicked.connect(self.classify)
        left.addWidget(self.pbClassify)

        root.addLayout(left)

        # -- right column: comparison table + status --
        right = QVBoxLayout()
        right.addWidget(self._section_label('Per-class probability (%)'))
        self.table = QTableWidget()
        self._init_table()
        right.addWidget(self.table, 1)

        self.lStatus = QLabel(self._startup_status())
        self.lStatus.setStyleSheet(f'color: {MUTED}; padding: 4px;')
        right.addWidget(self.lStatus)
        root.addLayout(right, 1)

    def _section_label(self, text):
        lbl = QLabel(text.upper())
        lbl.setObjectName('section')
        return lbl

    def _startup_status(self):
        if not self.registry:
            return ('[!] models_registry.json not found. '
                    'Run export_models.py first.')
        return f'[ ] Ready - {len(self.registry)} models available. Load an image.'

    def _populate_models(self, layout):
        """One large checkbox per registered model, all ticked by default,
        coloured by approach family with its audited accuracy."""
        for e in self.registry:
            acc = f"   ·   {e['accuracy']:.2f}%" if e.get('accuracy') is not None else ''
            cb = QCheckBox(f"{e['name']}{acc}")
            cb.setChecked(True)
            cb.setToolTip(f"Family: {e['family']}")
            color = FAMILY_COLOR.get(e['family'], '#333')
            cb.setStyleSheet(
                f'QCheckBox {{ color: {color}; font-size: 13px; font-weight: 600;'
                f' padding: 7px; border: none; background: transparent; }}'
                'QCheckBox::indicator { width: 18px; height: 18px; }')
            self.model_checks[e['slug']] = cb
            layout.addWidget(cb)
        layout.addStretch()

    def _set_all(self, state):
        for cb in self.model_checks.values():
            cb.setChecked(state)

    def _init_table(self):
        headers = ['Model'] + [CLASS_LABEL[c] for c in CLASSES] + ['Prediction', 'Conf.']
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(headers)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

    # ── actions ───────────────────────────────────────────────────────────
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select MRI image', '',
            'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not path:
            return
        self.image_path = path
        pix = QPixmap(path).scaled(
            336, 316, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.lImage.setPixmap(pix)
        self.pbClassify.setEnabled(bool(self.registry))
        self.lStatus.setText(f'[+] Image: {os.path.basename(path)}')

    def _selected_slugs(self):
        return [slug for slug, cb in self.model_checks.items() if cb.isChecked()]

    def classify(self):
        if not self.image_path:
            return
        slugs = self._selected_slugs()
        if not slugs:
            QMessageBox.information(self, 'No models',
                                    'Select at least one model.')
            return
        self.pbClassify.setEnabled(False)
        self.table.setRowCount(0)
        try:
            from brisc_models import consensus
            preds = []
            for n, slug in enumerate(slugs, 1):
                entry = next(e for e in self.registry if e['slug'] == slug)
                self.lStatus.setText(
                    f'[...] ({n}/{len(slugs)}) {entry["name"]} - classifying...')
                QApplication.processEvents()
                model = self._get_model(slug)
                result = model.predict(self.image_path)
                self._add_row(entry['name'], entry['family'], result)
                preds.append(result)
            # consensus row (soft voting across the selected models)
            self._add_row('Consensus (mean)', None, consensus(preds), is_consensus=True)
            self.lStatus.setText(
                f'[*] {len(slugs)} models classified. '
                "Highlighted cell = each model's predicted class.")
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            self.lStatus.setText(f'[!] Error: {e}')
        finally:
            self.pbClassify.setEnabled(True)

    def _add_row(self, name, family, result, is_consensus=False):
        r = self.table.rowCount()
        self.table.insertRow(r)
        best = max(result, key=result.get)

        name_item = QTableWidgetItem(name)
        if family:
            name_item.setForeground(QColor(FAMILY_COLOR.get(family, '#333')))
        if is_consensus:
            f = QFont(); f.setBold(True); name_item.setFont(f)
        self.table.setItem(r, 0, name_item)

        for c_i, cls in enumerate(CLASSES, start=1):
            cell = QTableWidgetItem(f'{result[cls] * 100:.1f}')
            cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if cls == best:                       # highlight the argmax cell
                cell.setBackground(QColor(ACCENT if is_consensus else HILITE))
                cell.setForeground(QColor('#ffffff' if is_consensus else INK))
                bf = QFont(); bf.setBold(True); cell.setFont(bf)
            self.table.setItem(r, c_i, cell)

        pred = QTableWidgetItem(CLASS_LABEL[best])
        pred.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(r, len(CLASSES) + 1, pred)
        conf = QTableWidgetItem(f'{result[best] * 100:.1f}%')
        conf.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(r, len(CLASSES) + 2, conf)

    def clear_all(self):
        self.image_path = None
        self.lImage.clear()
        self.lImage.setText('No image loaded')
        self.table.setRowCount(0)
        self.pbClassify.setEnabled(False)
        self.lStatus.setText(self._startup_status())


def main():
    app = QApplication([])
    app.setStyle('Fusion')
    app.setStyleSheet(QSS)
    w = BriscMultiApp()
    w.show()
    app.exec()


if __name__ == '__main__':
    main()
