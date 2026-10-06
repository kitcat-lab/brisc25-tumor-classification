"""Check exact decoded RGB equality; different pixels still need visual review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    with args.review.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    root = args.dataset.resolve()
    cache = {}
    for row in rows:
        for side in ('a', 'b'):
            key = row[side]
            if key not in cache:
                path = (root/key).resolve()
                if not path.is_relative_to(root):
                    raise ValueError('Candidate path outside dataset')
                with path.open('rb') as handle:
                    file_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                    cache[key] = file_sha, rgb.size, pixel_sha
            if cache[key][0] != row[side+'_sha256']:
                raise ValueError('Source differs from candidate inventory: ' + key)
        a, b = cache[row['a']], cache[row['b']]
        row['same_dimensions'] = a[1] == b[1]
        row['identical_decoded_rgb'] = a[1:] == b[1:]
    summary = {'pairs_checked': len(rows), 'images_checked': len(cache),
               'exact_file_sha256_pairs': sum(
                   cache[r['a']][0] == cache[r['b']][0] for r in rows),
               'identical_decoded_rgb_pairs': sum(r['identical_decoded_rgb'] for r in rows),
               'identical_decoded_rgb_cross_split_pairs': sum(
                   r['identical_decoded_rgb'] and r['cross_split'] == 'True' for r in rows),
               'identical_decoded_rgb_cross_class_pairs': sum(
                   r['identical_decoded_rgb'] and r['cross_class'] == 'True' for r in rows),
               'automatic_removals': 0,
               'method': 'Pillow RGB decode, original dimensions; no registration or resizing',
               'note': 'Different pixels are not proof that images are unrelated; review remains required.'}
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for row in rows:
        if row['identical_decoded_rgb']:
            parent[find(row['a'])] = find(row['b'])
    groups = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)
    summary.update({
        'identical_rgb_groups': len(groups),
        'images_in_identical_rgb_groups': len(parent),
        'redundant_images_if_one_per_identical_rgb_group': len(parent)-len(groups),
        'identical_rgb_groups_spanning_train_test': sum(
            len({key.split('/')[0] for key in group}) > 1 for group in groups.values())})
    args.output.mkdir(parents=True)
    with (args.output/'review.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
