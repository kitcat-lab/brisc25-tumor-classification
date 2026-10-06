"""Regression checks against the recovered evidence and curation safety contract."""
import sys, tempfile, subprocess, csv, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import metrics, align_frames, validate_predictions

class EvidenceTests(unittest.TestCase):
    def test_canonical_accuracy_and_corrected_ci(self):
        frame=pd.read_csv(ROOT/'results/verified/cnn_canonical_predictions.csv')
        result=metrics(frame)
        self.assertEqual(result['correct'],636)
        self.assertAlmostEqual(result['accuracy'],100*636/678)
        self.assertAlmostEqual(result['ci95_lo'],91.88790560471976)
        self.assertAlmostEqual(result['ci95_hi'],95.42772861356931)
        self.assertEqual(result,metrics(frame.sample(frac=1,random_state=17)))

    def test_pairing_rejects_cohort_or_label_changes(self):
        a=pd.read_csv(ROOT/'results/verified/vgg16_predictions.csv')
        b=pd.read_csv(ROOT/'results/verified/vgg16_lgbm_predictions.csv')
        aligned_a,aligned_b=align_frames(a,b.sample(frac=1,random_state=91))
        self.assertTrue(aligned_a.index.equals(aligned_b.index))
        with self.assertRaises(ValueError):align_frames(a,b.iloc[:-1])
        changed=b.copy();changed.loc[0,'y_true']=1
        with self.assertRaises(ValueError):align_frames(a,changed)
        duplicate=pd.concat([a,a.iloc[:1]])
        with self.assertRaises(ValueError):validate_predictions(duplicate)

    def test_curation_requires_review_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);raw=base/'raw'
            for split in ['train','test']:
                for label in ['glioma','meningioma','no_tumor','pituitary']:(raw/split/label).mkdir(parents=True)
            Image.fromarray(np.arange(256,dtype=np.uint8).reshape(16,16)).save(raw/'train/glioma/a.png')
            source=(raw/'train/glioma/a.png').read_bytes()
            (raw/'test/meningioma/b.png').write_bytes(source)
            def run(*args):
                return subprocess.run([sys.executable,str(ROOT/'scripts/curation.py'),*map(str,args)],capture_output=True,text=True)
            candidates=run('candidates','--dataset',raw,'--output',base/'review')
            self.assertEqual(candidates.returncode,0,candidates.stderr)
            review=base/'review/review.csv'
            with review.open() as handle: rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),1)
            failure=run('apply','--dataset',raw,'--review',review,'--output',base/'unresolved')
            self.assertNotEqual(failure.returncode,0)
            self.assertFalse((base/'unresolved').exists())
            rows[0].update(decision='keep_a',reviewer='synthetic regression test',note='identical synthetic bytes')
            with review.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            applied=run('apply','--dataset',raw,'--review',review,'--output',base/'curated')
            self.assertEqual(applied.returncode,0,applied.stderr)
            self.assertEqual(len(list((base/'curated').rglob('*.png'))),1)
            self.assertEqual((raw/'train/glioma/a.png').read_bytes(),source)
            self.assertEqual((raw/'test/meningioma/b.png').read_bytes(),source)
            self.assertNotEqual(run('apply','--dataset',raw,'--review',review,'--output',base/'curated').returncode,0)

if __name__=='__main__':unittest.main()
