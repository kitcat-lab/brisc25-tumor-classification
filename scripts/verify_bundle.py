"""Check tracked package files without models or third-party libraries."""
import argparse, hashlib, json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=root)
    args=parser.parse_args();missing=[];changed=[];checked=0
    for entry in json.loads((args.root/'provenance/bundle_files.json').read_text(encoding='utf-8')):
        p=args.root/entry['path']
        if not p.is_file():missing.append(entry['path']);continue
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        if h!=entry['sha256']:changed.append(entry['path'])
        checked+=1
    print(json.dumps({'checked':checked,'missing':missing,'changed':changed},indent=2))
    if missing or changed:raise SystemExit(1)

if __name__=='__main__':main()
