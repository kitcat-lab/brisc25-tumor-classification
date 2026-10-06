"""Diagnostic LightGBM refit matching the historical figure script, not the original sweep."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,openpyxl
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from common import ROOT,CLASSES,save_predictions,metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=ROOT/'artifacts/features/features_BRISC_db4_preprocessed.xlsx')
    parser.add_argument('--sheet',default='NL16')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output directory')
    workbook=openpyxl.load_workbook(args.features,read_only=True,data_only=True)
    it=workbook[args.sheet].iter_rows(values_only=True);header=list(next(it));columns={name:i for i,name in enumerate(header)}
    features=[i for i,name in enumerate(header) if name not in ['filename','split','class','view','index']]
    train=[];labels=[];test={}
    def number(x):
        try:return float(x)
        except (TypeError,ValueError):return 0.
    for row in it:
        if row[columns['filename']] is None:continue
        values=[number(row[i]) for i in features];label=CLASSES.index(row[columns['class']])
        if row[columns['split']]=='train':train.append(values);labels.append(label)
        else:
            name=Path(str(row[columns['filename']])).name
            if name in test:raise ValueError('Duplicate feature-table identifier')
            test[name]=(values,label)
    workbook.close()
    manifest=pd.read_csv(ROOT/'data/manifests/test_manifest.csv').sort_values(['class','filename'])
    names=manifest.filename.tolist()
    if set(names)!=set(test):raise ValueError('Feature test membership differs from historical manifest')
    y=np.array([test[n][1] for n in names])
    if not np.array_equal(y,np.array([CLASSES.index(c) for c in manifest['class']])):raise ValueError('Feature labels differ from image labels')
    scaler=StandardScaler().fit(train)
    classifier=LGBMClassifier(n_estimators=500,learning_rate=.05,num_leaves=63,random_state=42,verbose=-1)
    classifier.fit(scaler.transform(train),labels)
    probabilities=classifier.predict_proba(scaler.transform([test[n][0] for n in names]))
    args.output.mkdir(parents=True)
    path=args.output/'classical_refit_predictions.csv'
    save_predictions(path,names,y,probabilities.argmax(axis=1),probabilities)
    result={**metrics(pd.read_csv(path)),'execution':'diagnostic_python_refit','original_classical_execution_recovered':False,
            'feature_sheet':args.sheet,'n_features':len(features),'n_train':len(train)}
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
