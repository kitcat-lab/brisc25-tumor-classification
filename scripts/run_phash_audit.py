"""Verify the original corpus and rerun pHash/pixel checks without changing images."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    stages = [
        ('verify_original', 'curation.py', ['verify', '--original', '--dataset', str(args.dataset)]),
        ('candidates', 'curation.py', ['candidates', '--dataset', str(args.dataset), '--threshold', '5',
                                     '--output', str(args.output/'candidates')]),
        ('pixels', 'compare_candidate_pixels.py', ['--dataset', str(args.dataset),
                  '--review', str(args.output/'candidates/review.csv'), '--output', str(args.output/'pixels')]),
        ('summary', 'summarize_phash_candidates.py', ['--review', str(args.output/'pixels/review.csv'),
                   '--output', str(args.output/'summary')])]
    metadata = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'python': sys.version.split()[0],
                'versions': {name: importlib.metadata.version(name)
                             for name in ['numpy', 'Pillow', 'ImageHash']},
                'source_hashes': {}, 'stages': [], 'dataset_changes': 0,
                'manual_adjudication': 'not performed by this command'}
    for rel in ['data/manifests/dataset_manifest_verified.csv', 'data/manifests/audit_log_959.csv',
                'scripts/curation.py', 'scripts/compare_candidate_pixels.py',
                'scripts/summarize_phash_candidates.py', 'scripts/run_phash_audit.py']:
        with (ROOT/rel).open('rb') as handle:
            metadata['source_hashes'][rel] = hashlib.file_digest(handle, 'sha256').hexdigest()
    for stage, script, extra in stages:
        print('Running:', stage, flush=True)
        result = subprocess.run([sys.executable, str(ROOT/'scripts'/script), *extra],
                                capture_output=True, text=True)
        (args.output/(stage+'.stdout.txt')).write_text(result.stdout, encoding='utf-8')
        (args.output/(stage+'.stderr.txt')).write_text(result.stderr, encoding='utf-8')
        metadata['stages'].append({'name': stage, 'exit_code': result.returncode})
        metadata['status'] = 'running' if result.returncode == 0 else 'failed'
        (args.output/'execution.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        if result.returncode:
            print(result.stderr, flush=True)
            raise SystemExit(result.returncode)
        print(result.stdout.strip(), flush=True)
    metadata['status'] = 'completed'
    metadata['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.output/'execution.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Completed. No dataset changes.', flush=True)


if __name__ == '__main__':
    main()
