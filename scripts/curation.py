"""Generate similarity candidates for human review; apply only explicit decisions."""
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash
from common import CLASSES, ROOT

def inventory(root):
    return {f'{split}/{label}/{p.name}': p for split in ['train','test'] for label in CLASSES
            for p in sorted((root/split/label).iterdir()) if p.suffix.lower() in ['.jpg','.jpeg','.png']}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(args):
    files=inventory(args.dataset); manifest=list(csv.DictReader(args.manifest.open()))
    expected={r['relative_path']:r for r in manifest if args.original or r['included'].lower()=='true'}
    mismatches=[]
    for key in set(files)&set(expected):
        if sha(files[key])!=expected[key]['sha256']:mismatches.append(key)
    result={'expected':len(expected),'found':len(files),'missing':sorted(set(expected)-set(files)),
            'extra':sorted(set(files)-set(expected)),'content_mismatches':mismatches,
            'status':'historical_manifest_check; does not validate duplicate-removal decisions'}
    print(json.dumps(result,indent=2))
    if result['missing'] or result['extra'] or mismatches:raise SystemExit(1)

def candidates(args):
    if not 0<=args.threshold<=64:raise ValueError('Threshold must be between 0 and 64')
    if args.output.exists():raise ValueError('Output must be a fresh directory')
    files=inventory(args.dataset);keys=sorted(files);info={};hashes=[]
    for key in keys:
        p=files[key];data=p.read_bytes()
        with Image.open(p) as im:h=int(str(imagehash.phash(im.convert('RGB'))),16)
        info[key]={'sha256':hashlib.sha256(data).hexdigest(),'md5':hashlib.md5(data).hexdigest()};hashes.append(h)
    hashes=np.asarray(hashes,dtype=np.uint64);rows=[];distribution={i:0 for i in range(args.threshold+1)}
    for i,key in enumerate(keys):
        distances=np.bitwise_count(np.bitwise_xor(hashes[i+1:],hashes[i]))
        for j in np.flatnonzero(distances<=args.threshold)+i+1:
            other=keys[j];distance=int(distances[j-i-1]);distribution[distance]+=1
            rows.append({'a':key,'b':other,'a_sha256':info[key]['sha256'],'b_sha256':info[other]['sha256'],
                         'distance':distance,'exact_md5':info[key]['md5']==info[other]['md5'],
                         'cross_class':key.split('/')[1]!=other.split('/')[1],
                         'cross_split':key.split('/')[0]!=other.split('/')[0],
                         'decision':'','reviewer':'','note':''})
    args.output.mkdir(parents=True)
    columns=['a','b','a_sha256','b_sha256','distance','exact_md5','cross_class','cross_split','decision','reviewer','note']
    with (args.output/'review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    (args.output/'candidate_metadata.json').write_text(json.dumps({'image_count':len(keys),'hash':'imagehash.phash RGB, default 64 bits',
        'max_distance':args.threshold,'pair_counts_by_distance':distribution,'automatic_removals':0},indent=2))
    print(f'{len(rows)} candidate pairs written; no images removed')

def apply(args):
    if args.output.exists():raise ValueError('Output must be a new directory')
    files=inventory(args.dataset);rows=list(csv.DictReader(args.review.open()));drops=set();keeps=set();checked={}
    for row in rows:
        if row['decision'] not in ['distinct','keep_a','keep_b'] or not row['reviewer'].strip() or not row['note'].strip():
            raise ValueError('Every candidate requires decision, reviewer and note')
        for column in ['a','b']:
            key=row[column]
            if key not in files:raise ValueError('Unknown source image '+key)
            if key not in checked:checked[key]=sha(files[key])
            if checked[key]!=row[column+'_sha256']:raise ValueError('Source changed since review: '+key)
        if row['decision']=='keep_a':keeps.add(row['a']);drops.add(row['b'])
        if row['decision']=='keep_b':keeps.add(row['b']);drops.add(row['a'])
    if keeps&drops:raise ValueError('Conflicting keep/remove decisions; resolve the review before applying')
    args.output.mkdir(parents=True);manifest=[]
    for key,p in files.items():
        if key in drops:continue
        target=args.output/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        manifest.append({'relative_path':key,'sha256':sha(target)})
    with (args.output/'curated_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','sha256']);w.writeheader();w.writerows(manifest)
    shutil.copy2(args.review,args.output/'review_decisions.csv')
    print(f'Copied {len(manifest)} reviewed images into new dataset; omitted {len(drops)} confirmed duplicates')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('verify');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--original',action='store_true')
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifests/dataset_manifest_verified.csv');p.set_defaults(func=verify)
    p=sub.add_parser('candidates');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--threshold',type=int,default=5)
    p.add_argument('--output',type=Path,required=True);p.set_defaults(func=candidates)
    p=sub.add_parser('apply');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.set_defaults(func=apply)
    args=parser.parse_args();args.func(args)

if __name__=='__main__':main()
