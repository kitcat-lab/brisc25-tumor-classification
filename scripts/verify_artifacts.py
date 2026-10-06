"""Verify local untracked artifacts against the tracked SHA256 inventory."""
import argparse, hashlib, json
from pathlib import Path
from common import ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args(); failures=[]; missing=[]; checked=0
    for item in json.loads((args.root/'provenance/artifacts.json').read_text()):
        p=args.root/item['path']
        if not p.is_file(): missing.append(item['path']); continue
        with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=item['sha256'] or p.stat().st_size!=item['size']: failures.append(item['path'])
        checked+=1
    print(json.dumps({'checked':checked,'missing':missing,'mismatched':failures},indent=2))
    if failures or (missing and not args.allow_missing): raise SystemExit(1)

if __name__=='__main__': main()
