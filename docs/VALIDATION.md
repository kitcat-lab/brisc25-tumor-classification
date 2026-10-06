# Package validation — 6 October 2026

Executed in the existing Python 3.12 WSL environment with the versions recorded under `results/verified/runtime_versions.json` and pinned requirements. No new neural training was run while assembling this package.

| Check | Outcome |
|---|---|
| Python entry scripts compile | Passed |
| Identified prediction tables → corrected statistics | Passed; canonical CNN 636/678, stratified CI 91.89–95.43% |
| SHA256/size inventory of 23 local artifacts | All matched; none missing |
| Historical retained image corpus → manifest | 5041 images matched; no membership or content discrepancies |
| Portable evaluation command with copied canonical checkpoint | 93.8053097345%, reproduces recovered execution |
| Portable classical refit | 614/678, 90.5604719764%; retained as separate diagnostic execution |
| Regression: canonical evidence and shuffled-order invariance | Passed |
| Regression: mismatched IDs/labels and duplicate IDs | Rejected as required |
| Regression: unresolved curation and existing output | Rejected; reviewed synthetic output preserves source images |
| Corrected comparison figure | Generated as PNG/PDF/SVG and visually inspected |
| Git-only export, excluding local artifacts | Package hashes verified; statistics reproduced from that export |

The curation regression uses synthetic images. It does not approve the historical pHash exclusions. The retained corpus hash check establishes that this is the historical corpus, not that its curation decisions were valid.

MATLAB execution and new neural training were not run. The portable training scripts are adapted historical protocols with syntax validation; future training may expose environment/hardware-specific differences. Original classical-model reproduction and full pHash manual review remain unresolved scientific work.

Local artifacts and validation run outputs are excluded from Git. The standalone package has no remote. This branch retains the original repository remote and Git history. No push or UI-open action was performed.

## Checks after integration into the original repository

On 6 October 2026, the package was copied into the local `article-clean` branch, based on commit `21a47328ec31c5f75a29c353d3f5b861c4fd02ee`. The original Git history and remote were retained. Package integrity, artifact hashes, regression tests and statistics were checked again from this checkout. The audit snapshot revision was recovered by fetching the updated main branch before merging. Both references are documented in `provenance/integration.json`.

## Repeated pHash audit and coauthor document

The 6 October run verified all 6000 original images against the SHA256 manifest and completed all four stages of `run_phash_audit.py`. The candidate scan reproduced 1349 pairs; 159 had identical decoded RGB pixels, including 110 across train and test. Candidate counts were compared at distances 0, 2, 4 and 5; this is not model-performance sensitivity. No dataset changes or new model training occurred.

Five regression tests passed, including identical pixels in files with different metadata, changed-source rejection, historical-pair recovery and existing curation safeguards. The six-page coauthor PDF was visually checked. The Word document is generated from the same Markdown source. Full human adjudication remains pending; all published reviewer/decision fields are blank.
