"""Exploratory curation sensitivity, one backbone and fixed common evaluation sets."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import train_resnet_comparison as trainer
from common import ROOT, metrics, align_frames


def prepare(out):
    if out.exists():
        raise FileExistsError('Use a new output directory or resume the existing plan.')
    dataset = Path('/root/brisc/data_unaudited/classification_task')
    manifest = ROOT/'data/manifests/dataset_manifest_verified.csv'
    reference = ROOT/'runs/resnet_comparison_20261006_235702/split.csv'
    frame = pd.read_csv(manifest).sort_values('relative_path').reset_index(drop=True)
    rgb = []
    for row in frame.itertuples():
        path = dataset/row.relative_path
        if trainer.digest(path) != row.sha256:
            raise ValueError('Source image changed: '+str(path))
        with Image.open(path) as image:
            im = image.convert('RGB')
            rgb.append(hashlib.sha256(str(im.size).encode()+b'\0'+im.tobytes()).hexdigest())
    frame['rgb_sha256'] = rgb
    # Conflicting labels for identical pixels cannot be adjudicated automatically.
    conflicts = frame.groupby('rgb_sha256')['class'].nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    eligible = frame.loc[~frame.rgb_sha256.isin(conflicts)].copy()
    hist = eligible.loc[eligible.included.astype(str).str.lower().eq('true')]
    test = hist.loc[hist.split.eq('test')].drop_duplicates('rgb_sha256').copy()
    prior = pd.read_csv(reference)
    val_paths = set(prior.loc[prior.role.eq('validation'), 'relative_path'])
    val = hist.loc[hist.relative_path.isin(val_paths) & ~hist.rgb_sha256.isin(test.rgb_sha256)].drop_duplicates('rgb_sha256').copy()
    protected = set(test.rgb_sha256) | set(val.rgb_sha256)
    pool = eligible.loc[eligible.split.eq('train') & ~eligible.rgb_sha256.isin(protected)].copy()
    arms = {
        'restored_guarded': pool,
        'exact_only': pool.drop_duplicates('rgb_sha256'),
        'historical_guarded': pool.loc[pool.included.astype(str).str.lower().eq('true')],
    }
    cfg = json.loads((ROOT/'configs/resnet_comparison.json').read_text())
    cfg['protocols'] = {'adapted': cfg['protocols']['adapted']}
    cfg['weights'] = {'imagenet': cfg['weights']['imagenet']}
    weights = Path('/root/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if trainer.digest(weights) != cfg['weights']['imagenet']['sha256']:
        raise ValueError('Pretrained weights changed')
    sources = [Path(__file__), ROOT/'scripts/train_resnet_comparison.py', ROOT/'scripts/common.py', ROOT/'configs/resnet_comparison.json', manifest, reference]
    hashes = {str(p.resolve()): trainer.digest(p) for p in sources}
    out.mkdir(parents=True)
    frame.to_csv(out/'pixel_audit.csv', index=False)
    counts = []
    for arm, train in arms.items():
        folder = out/arm; folder.mkdir()
        split = pd.concat([train.assign(role='train'), val.assign(role='validation'), test.assign(role='test')]).sort_values(['role','class','filename'])
        groups = {role: set(part.rgb_sha256) for role, part in split.groupby('role')}
        assert not (groups['train'] & groups['test'] or groups['train'] & groups['validation'] or groups['validation'] & groups['test'])
        assert split.filename.is_unique
        assert all(part['class'].nunique() == 4 for _, part in split.groupby('role'))
        split.to_csv(folder/'split.csv', index=False)
        plan = dict(config=cfg, dataset=str(dataset), weights={'imagenet':str(weights)}, source_hashes=hashes,
            split_sha256=trainer.digest(folder/'split.csv'),
            jobs=[dict(id=f'imagenet_adapted_seed{s}', origin='imagenet', protocol='adapted', seed=s) for s in cfg['seeds']])
        (folder/'plan.json').write_text(json.dumps(plan, indent=2))
        if arm == 'historical_guarded':
            # Reuse only when image identities, roles, recipe, weights and runner match.
            old_plan = json.loads((reference.parent/'plan.json').read_text())
            cols = ['relative_path','role','sha256','class']
            left = split[cols].sort_values('relative_path').reset_index(drop=True)
            right = prior[cols].sort_values('relative_path').reset_index(drop=True)
            same_recipe = all(cfg[k] == old_plan['config'][k] for k in ['image_size','batch_size','head_epochs','head_learning_rate'])
            same_recipe &= cfg['protocols']['adapted'] == old_plan['config']['protocols']['adapted']
            same_recipe &= cfg['weights']['imagenet'] == old_plan['config']['weights']['imagenet']
            same_sources = all(old_plan['source_hashes'].get(str(p.resolve())) == trainer.digest(p) for p in [ROOT/'scripts/train_resnet_comparison.py', ROOT/'scripts/common.py'])
            if left.equals(right) and same_recipe and same_sources:
                for job in plan['jobs']:
                    source = reference.parent/job['id']
                    result = json.loads((source/'metrics.json').read_text())
                    checked = metrics(pd.read_csv(source/'predictions.csv'))
                    assert abs(checked['accuracy']-result['accuracy']) < 1e-9
                    assert result['split_sha256'] == old_plan['split_sha256']
                    target = folder/job['id']; target.mkdir()
                    for name in ['predictions.csv','history.json']:
                        shutil.copy2(source/name,target/name)
                    result['bootstrap_seed'] = result['seed']
                    result['seed'] = job['seed']
                    result['reused_from'] = str(source)
                    result['evaluation_split_sha256'] = plan['split_sha256']
                    (target/'metrics.json').write_text(json.dumps(result,indent=2))
                    (target/'reuse.json').write_text(json.dumps({'source':str(source),'identical_roles_images_recipe_and_runner':True,'source_file_hashes':{name:trainer.digest(source/name) for name in ['metrics.json','predictions.csv','history.json']},'model_path':str(source/'model.keras')},indent=2))
        for (role, label), part in split.groupby(['role','class']):
            counts.append(dict(arm=arm, role=role, label=label, n=len(part)))
    pd.DataFrame(counts).to_csv(out/'counts.csv', index=False)
    (out/'design.json').write_text(json.dumps(dict(arms=list(arms), seeds=cfg['seeds'], conflicting_rgb_groups=len(conflicts),
        common_test=len(test), common_validation=len(val), scope='Exploratory training-curation sensitivity on a common historical test subset. Restored and historical arms have identical-pixel safeguards; neither is a literal reproduction. No patient-independence claim; no manual pHash adjudication; no causal attribution of filtering versus training-set size.'), indent=2))
    print(pd.DataFrame(counts).groupby(['arm','role']).n.sum().to_string())


def run(out, arm, job):
    # Preserve the old runner and its historical source hashes. Separate metadata here.
    def labelled_metrics(frame):
        result = metrics(frame)
        result['bootstrap_seed'] = result.pop('seed')
        return result
    trainer.metrics = labelled_metrics
    trainer.run_job(SimpleNamespace(output=out/arm, job=job))


def summarize(out):
    design = json.loads((out/'design.json').read_text())
    rows = []; predictions = {}
    for arm in design['arms']:
        for seed in design['seeds']:
            path = out/arm/f'imagenet_adapted_seed{seed}'
            if not (path/'metrics.json').exists():
                raise ValueError('Incomplete experiment: '+str(path))
            result = json.loads((path/'metrics.json').read_text())
            pred = pd.read_csv(path/'predictions.csv')
            checked = metrics(pred)
            assert abs(checked['accuracy']-result['accuracy']) < 1e-9
            assert result['seed'] == seed
            rows.append(dict(arm=arm, **result)); predictions[arm,seed] = pred
    table = pd.DataFrame(rows)
    table.to_csv(out/'all_results.csv', index=False)
    summary = table.groupby('arm').agg(n_seeds=('seed','nunique'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),f1_mean=('f1_macro','mean'))
    summary.to_csv(out/'summary.csv')
    differences = []
    for arm in ['restored_guarded','exact_only']:
        for seed in design['seeds']:
            a,b = align_frames(predictions[arm,seed], predictions['historical_guarded',seed])
            delta = (a.y_pred.eq(a.y_true).astype(int)-b.y_pred.eq(b.y_true).astype(int)).to_numpy()
            y = a.y_true.to_numpy(); rng = np.random.RandomState(42)
            groups = [np.flatnonzero(y == c) for c in np.unique(y)]
            boot = [100*delta[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean() for _ in range(2000)]
            lo,hi = np.percentile(boot,[2.5,97.5])
            differences.append(dict(arm=arm,seed=seed,delta_accuracy_pp=100*delta.mean(),ci95_lo=lo,ci95_hi=hi))
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    report = '# Curation sensitivity: exploratory results\n\nResNet50–ImageNet, adapted protocol; all three seeds reported.\n\n'+summary.to_string()+'\n\nPaired differences versus historical_guarded (percentage points):\n\n'+pd.DataFrame(differences).to_string(index=False)+'\n\nPer-seed intervals resample test images within classes. They do not account for patient dependence or training variability and are not adjusted for multiple comparisons. No equivalence claim is supported. The common test excludes some historical cases; these scores must not replace the main manuscript table. Differences include training-set composition and size. This experiment does not adjudicate pHash pairs or establish patient independence.\n'
    (out/'REPORT_FOR_COAUTHORS.md').write_text(report)
    print(report)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','run','summarize']); p.add_argument('--output',type=Path,required=True); p.add_argument('--arm'); p.add_argument('--job'); a=p.parse_args()
    if a.action=='prepare': prepare(a.output)
    elif a.action=='run': run(a.output,a.arm,a.job)
    else: summarize(a.output)
