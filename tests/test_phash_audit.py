"""Check that pixel equality does not depend on file metadata or pHash labels."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from summarize_phash_candidates import summarize


class PixelAuditTests(unittest.TestCase):
    def test_same_pixels_different_bytes_and_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ['train/glioma/a.png', 'test/glioma/b.png', 'test/glioma/c.png']
            for rel in paths:
                (root/rel).parent.mkdir(parents=True, exist_ok=True)
            image = Image.new('RGB', (8, 8), 'white')
            image.save(root/paths[0])
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text('Note', 'Different file metadata, identical RGB')
            image.save(root/paths[1], pnginfo=metadata)
            image.putpixel((0, 0), (0, 0, 0))
            image.save(root/paths[2])
            rows = []
            for rel in paths[1:]:
                rows.append({'a': paths[0], 'b': rel,
                    'a_sha256': hashlib.sha256((root/paths[0]).read_bytes()).hexdigest(),
                    'b_sha256': hashlib.sha256((root/rel).read_bytes()).hexdigest(),
                    'distance': 0, 'exact_md5': False, 'cross_class': False,
                    'cross_split': True, 'decision': '', 'reviewer': '', 'note': ''})
            review = root/'review.csv'
            with review.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            def run(output):
                return subprocess.run([sys.executable, str(ROOT/'scripts/compare_candidate_pixels.py'),
                    '--dataset', str(root), '--review', str(review), '--output', str(output)],
                    capture_output=True, text=True)
            result = run(root/'checked')
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'checked/summary.json').read_text())
            self.assertEqual(summary['exact_file_sha256_pairs'], 0)
            self.assertEqual(summary['identical_decoded_rgb_pairs'], 1)
            self.assertEqual(summary['redundant_images_if_one_per_identical_rgb_group'], 1)
            (root/paths[1]).write_bytes((root/paths[2]).read_bytes())
            self.assertNotEqual(run(root/'changed').returncode, 0)
            self.assertFalse((root/'changed').exists())

    def test_historical_pairs_must_be_present_and_graph_is_not_identity(self):
        rows = [dict(a='train/glioma/a', b='test/glioma/b', distance='0',
                     cross_class='False', cross_split='True', exact_md5='False',
                     identical_decoded_rgb='True'),
                dict(a='test/glioma/b', b='train/meningioma/c', distance='2',
                     cross_class='True', cross_split='True', exact_md5='False',
                     identical_decoded_rgb='False')]
        audit = [dict(reason='phash<=5', removed_path='test/glioma/b', kept_path='train/glioma/a')]
        annotated, summary = summarize(rows, audit)
        self.assertEqual(summary['similarity_graph_components'], 1)
        self.assertEqual(summary['historical_phash_identical_decoded_rgb_pairs'], 1)
        self.assertEqual(summary['threshold_sensitivity'][0]['candidate_pairs'], 1)
        self.assertEqual(summary['threshold_sensitivity'][1]['cross_class_pairs'], 1)
        self.assertEqual(annotated[0]['priority'], '1_cross_class')
        with self.assertRaises(ValueError): summarize(rows[:1], [dict(
            reason='phash<=5', removed_path='unknown', kept_path='train/glioma/a')])


if __name__ == '__main__':
    unittest.main()
