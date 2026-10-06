"""Describe pHash candidates and prioritize review without deciding exclusions."""
import argparse
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def summarize(rows, audit):
    lookup = {}
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row in rows:
        key = tuple(sorted((row['a'], row['b'])))
        if key in lookup:
            raise ValueError('Duplicate candidate pair: ' + repr(key))
        lookup[key] = row
        parent[find(row['a'])] = find(row['b'])

    groups = collections.defaultdict(list)
    for key in parent:
        groups[find(key)].append(key)
    component_ids = {}
    for i, group in enumerate(sorted(groups.values(), key=lambda g: min(g)), 1):
        for key in group:
            component_ids[key] = i

    historical = {}
    for row in audit:
        if row['reason'] == 'md5_exact':
            continue
        key = tuple(sorted((row['removed_path'], row['kept_path'])))
        if key not in lookup:
            raise ValueError('Historical pHash pair missing from candidates: ' + repr(key))
        historical[key] = row['removed_path']

    annotated = []
    for row in rows:
        key = tuple(sorted((row['a'], row['b'])))
        priority = ('1_cross_class' if row['cross_class'] == 'True' else
                    '2_cross_split' if row['cross_split'] == 'True' else
                    '3_within_split')
        annotated.append({**row, 'priority': priority,
                          'historical_removed': historical.get(key, ''),
                          'component_id': component_ids[row['a']]})
    annotated.sort(key=lambda r: (r['priority'], int(r['distance']), r['a'], r['b']))

    summary = {
        'candidate_pairs': len(rows),
        'unique_images_in_candidates': len(parent),
        'similarity_graph_components': len(groups),
        'largest_component_images': max((len(g) for g in groups.values()), default=0),
        'pair_counts_by_distance': dict(sorted(collections.Counter(
            int(r['distance']) for r in rows).items())),
        'cross_class_pairs': sum(r['cross_class'] == 'True' for r in rows),
        'cross_split_pairs': sum(r['cross_split'] == 'True' for r in rows),
        'cross_class_and_cross_split_pairs': sum(
            r['cross_class'] == r['cross_split'] == 'True' for r in rows),
        'exact_md5_pairs': sum(r['exact_md5'] == 'True' for r in rows),
        'exact_md5_cross_split_pairs': sum(
            r['exact_md5'] == r['cross_split'] == 'True' for r in rows),
        'historical_phash_exclusions': len(historical),
        'historical_phash_distance_counts': dict(sorted(collections.Counter(
            int(lookup[k]['distance']) for k in historical).items())),
        'historical_phash_cross_class_pairs': sum(
            lookup[k]['cross_class'] == 'True' for k in historical),
        'historical_phash_cross_split_pairs': sum(
            lookup[k]['cross_split'] == 'True' for k in historical),
        'historical_removed_by_split': dict(collections.Counter(
            p.split('/')[0] for p in historical.values())),
        'historical_removed_by_class': dict(collections.Counter(
            p.split('/')[1] for p in historical.values())),
        'automatic_removals': 0,
        'review_status': 'pending; priorities and graph components are not duplicate decisions',
        'component_note': 'Connected pHash candidates need not all be copies of one image.'
    }
    summary['threshold_sensitivity'] = []
    for threshold in (0, 2, 4, 5):
        selected = [r for r in rows if int(r['distance']) <= threshold]
        summary['threshold_sensitivity'].append({
            'max_hamming_distance': threshold,
            'candidate_pairs': len(selected),
            'cross_class_pairs': sum(r['cross_class'] == 'True' for r in selected),
            'cross_split_pairs': sum(r['cross_split'] == 'True' for r in selected),
            'identical_decoded_rgb_pairs': sum(
                r.get('identical_decoded_rgb') == 'True' for r in selected)})
    summary['threshold_sensitivity_note'] = (
        'Pair-count sensitivity only. No removals, new benchmark or model-performance sensitivity was computed.')
    if rows and 'identical_decoded_rgb' in rows[0]:
        summary['historical_phash_identical_decoded_rgb_pairs'] = sum(
            lookup[k]['identical_decoded_rgb'] == 'True' for k in historical)
    return annotated, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--audit-log', type=Path,
                        default=ROOT/'data/manifests/audit_log_959.csv')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    annotated, summary = summarize(read_rows(args.review), read_rows(args.audit_log))
    args.output.mkdir(parents=True)
    with (args.output/'review_prioritized.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotated[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(annotated)
    (args.output/'summary.json').write_text(
        json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
