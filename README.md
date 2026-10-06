# BRISC 2025 — brain-tumor classification reproducibility package

Materials supporting **Benchmarking Brain-Tumor MRI Classification Methods Under Leakage-Controlled Conditions: A BRISC 2025 Study**.

This repository brings together the original results, predictions linked to image filenames, corrected statistics and scripts for checking the models and reviewing duplicate-image candidates. **The historical pHash exclusions remain under review.** Reproducing their image count does not establish that every excluded image was a true duplicate, or that patient-level independence holds.

## Verified results

Inference on the historical 678-image test set confirmed:

| Configuration | Accuracy | Evidence |
|---|---:|---|
| Canonical three-block CNN | 93.8053% | 636/678; checkpoint and probabilities verified |
| Earlier three-block CNN execution | 93.0678% | 631/678; separate checkpoint |
| VGG16 softmax | 97.9351% | 664/678 |
| VGG16 + LightGBM | 98.6726% | 669/678; post hoc test-selected classifier |
| RadImageNet ResNet50 | 72.1239% | 489/678; historical preprocessing |

The CNN confidence interval in the manuscript came from the earlier execution. The corrected class-stratified bootstrap interval for the canonical CNN is **91.89–95.43%** (1000 resamples, seed 42, class/filename order). VGG16 versus VGG16+LightGBM gives continuity-corrected McNemar **p = 0.182422439**; Holm across the six declared contrasts retains that value.

See [audit findings](docs/AUDIT_2026-10-05.md), [result interpretation](docs/RESULTS.md) and [reproducibility instructions](docs/REPRODUCIBILITY.md).

## Quick start: recompute statistics

Python 3.12 is the recorded runtime. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements/analysis.txt
python scripts/reproduce_results.py --output runs/recomputed
```

Activate the environment before installing/running, or invoke its Python directly. This command needs no GPU, raw images or model files. It validates identifiers/probabilities, recomputes accuracy/F1/AUC and class-stratified CIs, and runs the declared six paired contrasts with Holm adjustment. It refuses to overwrite an existing output directory.

## Package layout

```text
configs/                 portable model registry
data/manifests/          original image hashes, historical inclusion, audit log
results/verified/        inference predictions with filenames and probabilities
results/corrected/       regenerated statistics from verified predictions
results/historical/      upstream outputs retained as historical evidence
figures/audit/           examples requiring review
figures/historical/      upstream article figures; may contain old statistics
scripts/                 portable analysis, inference, curation and training entries
reference/               historical training/MATLAB source for provenance
provenance/              upstream commit, artifact/source hashes, execution metadata
requirements/            separate analysis and training environments
artifacts/               local checkpoints/features/PDFs, excluded from Git
runs/                    new executions, excluded from Git
```

## Data and local artifacts

Obtain BRISC 2025 from its original publisher: DOI [10.1038/s41597-026-06753-y](https://doi.org/10.1038/s41597-026-06753-y). Images are external inputs and are not bundled in Git. Supply a dataset directory containing `train/<class>/` and `test/<class>/` for the four declared classes.

The historical dataset is frozen by `data/manifests/dataset_manifest_verified.csv`: 6000 original images, 4363 retained train and 678 retained test. The normalized original log contains 50 MD5 exclusions and 909 pHash exclusions. Several pHash candidates are visibly distinct; do not treat this log as an approved future-curation rule.

Local checkpoint/feature copies are available in `artifacts/` in this working tree. Their SHA256 and sizes are tracked in `provenance/artifacts.json`. A Git clone alone will not contain those large files; the artifact inventory defines the files to supply separately. The checkpoints are currently available locally; a public download location has not been set up.

```bash
python scripts/verify_artifacts.py
python scripts/verify_bundle.py
python scripts/curation.py verify --dataset /path/to/historical/classification_task
python scripts/evaluate_models.py --dataset /path/to/historical/classification_task --models cnn_canonical vgg16 radimagenet --output runs/inference
```

Use `requirements/training.txt` for model inference. Legacy pickles are trusted local project artifacts and must match the recorded hashes/environment; analysis-only commands do not load them. The provenance inventory includes the private working PDFs as local artifacts; they are excluded from Git.

## Curation for a future benchmark

```bash
python scripts/curation.py candidates --dataset /path/to/original/classification_task --threshold 5 --output runs/review
```

This writes candidate pairs and hash-distance counts and removes **zero** images. Fill every row's `decision` (`distinct`, `keep_a`, `keep_b`), `reviewer` and `note` after reviewing the original images. Then:

```bash
python scripts/curation.py apply --dataset /path/to/original/classification_task --review runs/review/review.csv --output runs/curated_dataset
```

Application verifies reviewed image hashes, refuses unresolved/conflicting decisions and copies accepted images into a fresh dataset directory. A reviewed dataset is a new benchmark version; it must not silently replace the historical partition.

## Scientific scope and licensing

Hybrid classifier maxima were selected using test performance and remain exploratory. Holm and bootstrap do not remove that selection bias. Patient/study identifiers are absent; image deduplication cannot establish patient-level independence. Optimized EfficientNet changes several factors and is not a BatchNorm-only ablation. RadImageNet checkpoint preprocessing requires further validation before claims about pretraining superiority.

Project code retains the upstream MIT license. Dataset rights and third-party dependencies are separate. The external GLCM implementation is available only as a local artifact until its redistribution terms are documented; see [dependency provenance](third_party/README.md).
