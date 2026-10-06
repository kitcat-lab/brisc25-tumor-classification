"""Evaluate registered checkpoints on a dataset and save identified predictions."""
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, CLASSES, save_predictions, metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True,help='Directory containing train/ and test/')
    parser.add_argument('--models',nargs='+',default=['cnn_canonical','vgg16','radimagenet'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registry',type=Path,default=ROOT/'configs/models.json')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Use a fresh output directory')
    registry=json.loads(args.registry.read_text())
    if not set(args.models)<=set(registry):parser.error('Unknown model name')
    manifest=pd.read_csv(ROOT/'data/manifests/test_manifest.csv').set_index('filename')
    paths=[p for c in CLASSES for p in sorted((args.dataset/'test'/c).iterdir()) if p.suffix.lower() in ['.jpg','.jpeg','.png']]
    if set(p.name for p in paths)!=set(manifest.index):parser.error('Test filenames differ from the frozen historical benchmark')
    y=np.array([CLASSES.index(p.parent.name) for p in paths])
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest.loc[p.name,'sha256'] or p.parent.name!=manifest.loc[p.name,'class']:
            parser.error('Image content or label differs from manifest: '+p.name)
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(gpu,True)
    def preprocess(name):
        if name=='rescale_255':return lambda x:x/255
        if name=='radimagenet_historical':return lambda x:x/127.5-1
        return getattr(tf.keras.applications,name).preprocess_input
    args.output.mkdir(parents=True);results={}
    for name in args.models:
        cfg=registry[name];checkpoint=ROOT/cfg['path']
        with checkpoint.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=cfg['sha256']:raise ValueError('Checkpoint hash mismatch: '+name)
        custom={}
        if name=='convnext':
            try:
                from keras.src.applications.convnext import LayerScale
                custom={'LayerScale':LayerScale}
            except ImportError:pass
        model=tf.keras.models.load_model(checkpoint,compile=False,custom_objects=custom)
        pre=preprocess(cfg['preprocessing']);size=cfg['image_size']
        ext=None
        if cfg.get('hybrid'):
            if cfg['feature_layer']:ext=tf.keras.Model(model.inputs,model.get_layer(cfg['feature_layer']).output)
            else:ext=tf.keras.Sequential(model.layers[:-1]);ext.build(model.input_shape)
        probs=[];features=[]
        for i in range(0,len(paths),32):
            batch=pre(np.stack([tf.keras.utils.img_to_array(tf.keras.utils.load_img(p,target_size=(size,size))) for p in paths[i:i+32]]))
            probs.append(model(batch,training=False).numpy())
            if ext is not None:features.append(ext(batch,training=False).numpy())
        def emit(label,prob):
            path=args.output/f'{label}_predictions.csv'
            save_predictions(path,[p.name for p in paths],y,prob.argmax(axis=1),prob)
            results[label]={**metrics(pd.read_csv(path)),'model_sha256':sha,'historical_test_set':True}
        emit(name,np.concatenate(probs))
        if ext is not None:
            with (ROOT/cfg['hybrid']['scaler']).open('rb') as f:scaler=pickle.load(f)
            with (ROOT/cfg['hybrid']['classifier']).open('rb') as f:classifier=pickle.load(f)
            emit(name+'_lgbm',classifier.predict_proba(scaler.transform(np.concatenate(features))))
            results[name+'_lgbm']['exploratory_test_selection']=True
        (args.output/'metrics.json').write_text(json.dumps(results,indent=2))
        del model,ext;tf.keras.backend.clear_session()
        print(name,results[name]['accuracy'],flush=True)

if __name__=='__main__':main()
