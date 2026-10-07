"""Prepare a frozen plan or train one ResNet50 job; notebook owns orchestration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common import ROOT, CLASSES, metrics, save_predictions


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(args):
    cfg = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; an existing plan can be resumed with --job')
    frame = pd.read_csv(ROOT/'data/manifests/dataset_manifest_verified.csv')
    frame = frame.loc[frame.included.astype(str).str.lower().eq('true')].copy()
    expected = set(frame.relative_path)
    observed = {p.relative_to(args.dataset).as_posix() for p in args.dataset.rglob('*')
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    if expected != observed:
        raise ValueError('Dataset membership differs from the frozen historical partition')
    for row in frame.itertuples():
        if digest(args.dataset/row.relative_path) != row.sha256:
            raise ValueError('Changed source image: '+row.relative_path)
    train = frame.loc[frame.split.eq('train')]
    fit, val = train_test_split(train, test_size=cfg['validation_fraction'],
                               stratify=train['class'], random_state=cfg['split_seed'])
    frame['role'] = 'test'
    frame.loc[fit.index, 'role'] = 'train'
    frame.loc[val.index, 'role'] = 'validation'
    weight_paths = {'imagenet': args.imagenet_weights.resolve(), 'radimagenet': args.radimagenet_weights.resolve()}
    for name, path in weight_paths.items():
        if digest(path) != cfg['weights'][name]['sha256']:
            raise ValueError('Weight hash mismatch: '+name)
    sources = [Path(__file__), ROOT/'scripts/common.py', args.config]
    plan = {'config': cfg, 'dataset': str(args.dataset.resolve()),
            'weights': {k: str(v) for k, v in weight_paths.items()},
            'source_hashes': {str(p.resolve()): digest(p) for p in sources},
            'jobs': [{'id': f'{origin}_{protocol}_seed{seed}', 'origin': origin,
                      'protocol': protocol, 'seed': seed}
                     for seed in cfg['seeds'] for protocol in cfg['protocols']
                     for origin in cfg['weights']]}
    args.output.mkdir(parents=True)
    frame.sort_values(['role','class','filename']).to_csv(args.output/'split.csv', index=False)
    plan['split_sha256'] = digest(args.output/'split.csv')
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    print(frame.groupby(['role','class']).size().to_string())
    print('Prepared', len(plan['jobs']), 'jobs in', args.output, flush=True)


def run_job(args):
    plan = json.loads((args.output/'plan.json').read_text())
    if digest(args.output/'split.csv') != plan['split_sha256']:
        raise ValueError('Split changed after preparation')
    for path, sha in plan['source_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Source changed after preparation: '+path)
    job = next(j for j in plan['jobs'] if j['id'] == args.job)
    cfg = plan['config']; recipe = cfg['protocols'][job['protocol']]
    output = args.output/job['id']
    if output.exists():
        raise FileExistsError('Job output already exists; inspect it rather than overwriting')
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    gpu = tf.config.list_physical_devices('GPU')
    if not gpu:
        raise RuntimeError('No GPU detected. Use the Linux/WSL kernel.')
    for device in gpu:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(job['seed'])
    tf.config.experimental.enable_op_determinism()
    frame = pd.read_csv(args.output/'split.csv')
    frame['path'] = [str(Path(plan['dataset'])/p) for p in frame.relative_path]
    for row in frame.itertuples():
        if digest(row.path) != row.sha256:
            raise ValueError('Dataset changed: '+row.relative_path)
    weights = Path(plan['weights'][job['origin']])
    if digest(weights) != cfg['weights'][job['origin']]['sha256']:
        raise ValueError('Weights changed after preparation')
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.resnet50 import preprocess_input
    def preprocess(x):
        x = preprocess_input(x.copy())
        return x / 255.0 if job['origin'] == 'radimagenet' else x
    augmented = ImageDataGenerator(preprocessing_function=preprocess, rotation_range=10,
        zoom_range=0.1, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True)
    plain = ImageDataGenerator(preprocessing_function=preprocess)
    generators = {}
    for role in ['train','validation','test']:
        subset = frame.loc[frame.role.eq(role)].sort_values(['class','filename'])
        generators[role] = (augmented if role == 'train' else plain).flow_from_dataframe(
            subset, x_col='path', y_col='class', classes=CLASSES, class_mode='categorical',
            target_size=(cfg['image_size'],cfg['image_size']), batch_size=cfg['batch_size'],
            shuffle=role == 'train', seed=job['seed'], interpolation='nearest')
    backbone = tf.keras.applications.ResNet50(weights=None, include_top=False,
        input_shape=(cfg['image_size'],cfg['image_size'],3))
    backbone.load_weights(weights)
    backbone.trainable = False
    inputs = tf.keras.Input((cfg['image_size'],cfg['image_size'],3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3, seed=job['seed'])(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='feat_dense')(x)
    x = tf.keras.layers.Dropout(0.3, seed=job['seed'])(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(4, activation='softmax')(x))
    output.mkdir()
    def compile_model(rate):
        model.compile(optimizer=tf.keras.optimizers.Adam(rate),
                      loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=False)
    def early(patience):
        return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
            patience=patience, restore_best_weights=True)
    class Progress(tf.keras.callbacks.Callback):
        def __init__(self, stage):
            super().__init__(); self.stage=stage
        def on_epoch_end(self, epoch, logs=None):
            (output/'progress.json').write_text(json.dumps({'stage':self.stage,
                'epoch':epoch+1,'logs':{k:float(v) for k,v in (logs or {}).items()}}))
    start=time.time()
    compile_model(cfg['head_learning_rate'])
    warm=model.fit(generators['train'], validation_data=generators['validation'],
        epochs=cfg['head_epochs'], callbacks=[early(3), Progress('head')], verbose=2)
    # Keep a candidate from warm-up too: test is never used for checkpoint selection.
    checkpoint=output/'best.weights.h5'
    model.save_weights(checkpoint)
    best_warm=max(warm.history['val_accuracy'])
    backbone.trainable=True
    if recipe['unfreeze']=='top60':
        for layer in backbone.layers[:-60]: layer.trainable=False
    else:
        for layer in backbone.layers:
            if isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=False
    compile_model(recipe['learning_rate'])
    fine=model.fit(generators['train'], validation_data=generators['validation'],
        epochs=recipe['epochs'], callbacks=[early(recipe['patience']), Progress('fine_tune'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=0.5,
            patience=3,min_lr=1e-7)], verbose=2)
    if max(fine.history['val_accuracy']) <= best_warm:
        model.load_weights(checkpoint)
        selected_stage='head'
    else:
        selected_stage='fine_tune'
    model.save(output/'model.keras')
    test=generators['test']; test.reset()
    probabilities=model.predict(test,verbose=0)
    save_predictions(output/'predictions.csv', test.filenames, test.classes,
                     probabilities.argmax(axis=1), probabilities)
    result={**job, **metrics(pd.read_csv(output/'predictions.csv')),
        'selected_stage':selected_stage,'selection':'validation accuracy; best across both stages; ties prefer head',
        'best_validation_accuracy':max(best_warm,max(fine.history['val_accuracy'])),
        'training_seconds':time.time()-start,'tensorflow':tf.__version__,
        'pretrained_sha256':digest(weights),'split_sha256':plan['split_sha256'],
        'preprocessing':cfg['weights'][job['origin']]['preprocessing']}
    (output/'history.json').write_text(json.dumps({'head':warm.history,'fine_tune':fine.history},indent=2))
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/resnet_comparison.json')
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--imagenet-weights',type=Path)
    parser.add_argument('--radimagenet-weights',type=Path)
    args=parser.parse_args()
    if args.job: run_job(args)
    else:
        if any(v is None for v in [args.dataset,args.imagenet_weights,args.radimagenet_weights]):
            parser.error('Preparation requires dataset and both pretrained weights')
        prepare(args)
