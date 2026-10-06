# Reproduction workflow

All paths below are arguments or repository-relative paths. Run scripts with Python 3.12; the recorded environment is in `results/verified/runtime_versions.json`. Statistics do not require TensorFlow.

## 1. Analysis only

Install `requirements/analysis.txt`. Run `python scripts/reproduce_results.py --output runs/recomputed`. Compare with the tracked `results/corrected/` tables. Output ordering is fixed by class and filename. The seven verified configurations include an earlier CNN and a separate classical refit; the paired-test family contains only the four declared principal models.

## 2. Historical input verification

Use `python scripts/curation.py verify --dataset DATASET` on the retained historical directory. Add `--original` to verify the original 6000-image corpus. Verification checks SHA256 and file membership; it does not validate that exclusions were genuine duplicates. Never substitute `brisc2025_clean_v2`, whose 4886 images belong to a different audit.

## 3. Existing-checkpoint inference

Install `requirements/training.txt`; verify local artifacts with `python scripts/verify_artifacts.py`. Run:

```bash
python scripts/evaluate_models.py --dataset DATASET --models cnn_canonical vgg16 radimagenet --output runs/inference
```

The model registry includes the earlier CNN, the four-block CNN, the uniform transfer backbones and optimized EfficientNet B0–B2. Local LightGBM/scaler artifacts are supplied for the canonical CNN and VGG16. Each run checks the checkpoint and historical test-image hashes before inference. Other registered models can be requested with `--models`; they were not all rerun when assembling this package.

## 4. New training executions

These commands retain the historical numerical training protocols with portable paths and corrected evaluation reporting. Fresh executions are not guaranteed to reproduce bit-identical weights or accuracies on different runtimes/hardware. Validation augmentation in the canonical/common protocols is retained for fidelity; the four-block CNN uses its historical separate clean-validation generator.

```bash
python scripts/train_cnn.py --dataset DATASET --output runs/cnn-new
python scripts/train_one_backbone.py W7_VGG16_ImageNet audited DATASET runs/backbones-new
python scripts/train_efficientnet_optimized.py B1 audited DATASET runs/efficientnet-new
python scripts/train_scratch_cnn_optimized.py audited DATASET runs/cnn4-new
python scripts/train_hybrid.py W7_VGG16_ImageNet DATASET runs/hybrid-new
```

Training refuses completed output directories. New softmax prediction tables include filenames and class probabilities. Hybrid sweeps report all seven classifier predictions and explicitly label the test-ranked maximum as exploratory; they do not implement confirmatory validation-based head selection. `BRISC_CNN_MODEL` can override the scratch-CNN checkpoint. New transfer executions save `.keras`; hybrid loading accepts that format when the matching `.h5` is absent. `BRISC_MODEL_ROOT` can supply a model tree matching the documented `artifacts/models/` layout. `BRISC_RADIMAGENET_WEIGHTS` points to an external RadImageNet initialization checkpoint. No such initialization file is automatically fetched. The historical normalization is `/127.5 - 1`; checkpoint-specific preprocessing remains an unresolved research check.

## 5. Diagnostic classical refit

Install `requirements/classical.txt` and run:

```bash
python scripts/refit_classical.py --output runs/classical-refit
```

This is the separate figure-script refit with identified probabilities. It does not recreate the complete original 19-classifier factorial sweep or recover its original fitted model. Keep its execution ID and caveat.

## 6. MATLAB feature extraction

The four historical feature workbooks are supplied locally under `artifacts/features/`. Portable entry functions are under `scripts/matlab/`; historical source is under `reference/matlab/`. Dependencies include MATLAB Image Processing and Wavelet toolboxes and the external GLCM function. MATLAB execution has not been validated in this package because a MATLAB runtime was not invoked.

The historical MATLAB scripts expect a class-flattened input directory (`glioma/`, `meningioma/`, `no_tumor/`, `pituitary/`) with filenames encoding the official split. Prepare this directory from the manifest without changing filenames or mixing dataset versions. Example from MATLAB with the repository on its path:

```matlab
addpath('scripts/matlab');
with_db4('/path/to/class_flattened_images', '/path/to/new_feature_outputs');
```

The output directory receives the three NumLevels sheets. The db4 wrapper retains the original skull-stripping toggle in its source. Keep its value and resulting workbook name in execution metadata. Outputs from a revised curation must be treated as a new version.

## Large files and deployment state

`artifacts/`, raw images and `runs/` are ignored by Git. `provenance/artifacts.json` records the exact locally supplied checkpoints, two hybrid classifiers/scalers, workbooks and working PDFs. The third-party GLCM file is listed separately. A future public release should provide model/feature release assets or an artifact host with checksums, without embedding private reviewer correspondence. No remote or publication operation is part of the current package.

## Repeat the full pHash audit

Run the manifest check, candidate scan, decoded-RGB comparison and candidate-count sensitivity as one command:

```bash
python scripts/run_phash_audit.py --dataset /path/to/original/classification_task --output runs/new_phash_audit
```

The output directory must be new. The command records package versions, source hashes and stage exit codes. It writes diagnostic results only and makes no image-removal decisions. `results/curation/` contains the repeated 6 October execution with all review fields blank.

## Rebuild the coauthor document

```bash
python -m pip install -r requirements/documents.txt
python scripts/build_coauthor_report.py
```

The Markdown source under `docs/` is the editable master; Word and PDF are generated discussion copies. On Windows, `--font-dir C:/Windows/Fonts` embeds Arial in the PDF. Regenerating documents changes their file hashes; refresh the bundle inventory when publishing a new version.
