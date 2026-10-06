# pHash review — 6 October 2026

The historical filter removed genuinely repeated images, but it also linked visibly different images. Its threshold cannot be treated as proof of duplication. Keep the historical benchmark available for reproduction, and review the candidate pairs before creating a replacement dataset.

## Recomputed candidates

The scan used all 6000 original images, `imagehash.phash` on RGB images with its default 64-bit hash, and Hamming distance ≤5. Source SHA256 values were checked again during pixel comparison. No images were removed or restored during this review.

| Measurement | Count |
|---|---:|
| Candidate pairs | 1349 |
| Different images involved in candidates | 1834 |
| Pairs at distance 0 / 2 / 4 | 205 / 295 / 849 |
| Pairs across train and test | 440 |
| Pairs across class labels | 23 |
| Pairs across both class labels and splits | 3 |
| Pairs with identical file bytes, checked by SHA256 | 55 |
| Pairs with identical decoded RGB pixels and dimensions | 159 |
| Identical RGB pairs across train and test | 110 |
| Identical RGB pairs across class labels | 0 |

Pair counts differ from image-removal counts: an image can occur in several pairs. The 159 identical-RGB pairs form 134 groups containing 278 images. Retaining one image per such group would remove 144 redundant copies. That is an exact-copy reference scenario, not a complete leakage-control protocol; related slices and transformed copies may remain.

The broader pHash graph has 720 connected components, with at most 16 images in a component. These are review groups, not established duplicate groups. Similarity is not transitive: a path through several candidates does not establish that its endpoints are copies.

## Historical exclusions

All 909 logged pHash pairs were recovered in the new candidate scan. Their distances are 0 for 110 pairs, 2 for 193, and 4 for 606. There are 296 cross-split pairs and 10 cross-class pairs. The removed images comprise 599 train images and 310 test images.

Exact decoded-RGB equality confirms 83 of those 909 logged pHash pairs. The other 826 require a different assessment: different pixels can represent a recompressed copy, a transformed copy, a different slice, or an unrelated image. They must not all be counted as false positives. The 50 historical exact-file exclusions remain a separate category.

Visual inspection covered all 23 cross-class candidates and 18 same-class diagnostic candidates, selected at evenly spaced positions within each distance group, excluding exact-file pairs. This was a diagnostic inspection, not a random prevalence sample or a completed human adjudication of the dataset. It did not assess the correctness of diagnostic labels. Contact sheets remain local under `runs/phash_review_20261006/`.

Clear differences in internal image patterns and orientation occur in several cross-class pairs. Two historically excluded pairs even link axial images to sagittal images. A historically excluded pair with distance 2 is also visibly different. Within-class examples include apparently repeated images and images consistent with nearby slices; source patient/series information would be needed to establish that relationship.

Lowering the threshold to 2 therefore does not solve the problem. Identical pHash values alone are also insufficient evidence of pixel equality: only 159 of the 205 distance-zero pairs have identical decoded RGB arrays.

## What to do next

1. Keep the current 5041-image corpus and its results frozen as the historical version. Do not silently restore a few images to it.
2. Review the 23 cross-class pairs first, including the 10 pairs behind historical exclusions. Check all links involving each image before recommending restoration: it may also have a valid duplicate elsewhere.
3. Review the remaining train–test candidates, then within-split candidates. Separate exact copies, transformed copies, related but distinct slices, and unrelated images. Only confirmed copies justify duplicate removal. Related slices require patient/series grouping when that information is available.
4. For exact copies spanning train and test, choose and document a rule before assessing new results. One possible rule is to retain the train representative and remove its test copies; this changes the test population and must be reported. Resolve label conflicts separately rather than using the class name to pick a winner.
5. Record proposed decisions, reviewer and rationale. The published CSV leaves all decision fields blank; computational observations are not attributed to a human reviewer.
6. Create a new versioned dataset from those decisions. Compare it with the historical corpus and the exact-copy-only reference, reporting counts by class, split and exclusion reason. The latter reference does not establish independence by patient.
7. If training membership changes, retrain the compared methods and regenerate their features, predictions, intervals and paired tests on the revised benchmark. Existing checkpoints can support a diagnostic evaluation of restored test images, but cannot replace that retraining when training data changed. Select hybrid classifiers using training/validation data before the final test evaluation.

## What the manuscript can currently say

The current evidence supports describing the historical procedure as an automated image-similarity filter. It does not support calling every excluded image a verified duplicate, equating all exclusions with train–test leakage, or asserting patient-level independence.

A factual statement for the revision is:

> The historical filtering procedure excluded 50 exact-file duplicates and 909 images flagged by a pHash similarity rule. A subsequent audit found visually distinct images among the flagged pairs, so the pHash exclusions have not all been validated as duplicate removals. The historical results are retained for reproducibility; a reviewed dataset and a new benchmark are required to assess the effect of revised curation.

## Reproduce this review

Use a fresh output directory for each step:

```bash
python scripts/curation.py candidates --dataset /path/to/original/classification_task --threshold 5 --output runs/phash_candidates
python scripts/compare_candidate_pixels.py --dataset /path/to/original/classification_task --review runs/phash_candidates/review.csv --output runs/phash_pixels
python scripts/summarize_phash_candidates.py --review runs/phash_pixels/review.csv --output runs/phash_summary
```

The candidate list, pixel checks and priority order are published under `results/curation/`. Editing review fields is a separate step; these commands make no curation decisions.
