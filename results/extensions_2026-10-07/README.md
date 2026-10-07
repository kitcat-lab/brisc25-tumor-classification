# Follow-up experiments, 7 October 2026

These are exported evidence tables for workflows 17–19, separate from the historical benchmark.

- `resnet_comparison/`: 12 ResNet50 runs, two pretrained sources, two protocols, three training seeds.
- `inference/`: fresh inference on eight registered historical checkpoints, batch size 8. EfficientNetB0/B2 now reproduce all original class predictions after restoring nested Normalization state from H5. ResNet50 reload differs on one image; its original CSV supports the manuscript value. Superseded loading-bug outputs are explicitly archived.
- `phash_sensitivity/`: nine model evaluations: six new runs plus three explicitly reused runs. Common validation and test sets; one representative configuration.

Local path prefixes in exported JSON and CSV files were normalised. Original local execution files remain in `runs/`. Source hashes refer to the original execution files, not to rewritten portable exports. The training-seed metadata collision in workflow 17 has been corrected in this export from its frozen job plan; `bootstrap_seed` identifies the resampling seed. No predictions or scores were changed.

All seeds are reported. Standard deviations across seeds are not confidence intervals. Per-seed paired bootstrap intervals do not include training variability or patient clustering, and have no multiplicity adjustment. Equal means do not demonstrate equivalence. The test images excluded by historical curation were not restored in workflow 19.

The pretraining comparison changes multiple protocol components; it is not a BatchNormalization-only ablation. Historical RadImageNet preprocessing differs from the new comparison. No claim of general pretraining superiority or patient-level independence follows from these experiments.

The manifests describe local external datasets, not bundled image files. Checkpoints remain external artifacts. Notebook instructions and analysis scripts are in `notebooks/` and `scripts/`.
