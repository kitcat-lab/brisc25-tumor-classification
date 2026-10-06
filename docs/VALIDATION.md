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

On 6 October 2026, the package was copied into the local `codex/article-clean` branch, based on commit `21a47328ec31c5f75a29c353d3f5b861c4fd02ee`. The original Git history and remote were retained. Package integrity, artifact hashes, regression tests and statistics were checked again from this checkout. The recorded audit snapshot revision is absent from this checkout history; both references are documented in `provenance/integration.json`.
