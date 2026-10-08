"""Download only preregistered first-night records; verify official SHA256."""
from pathlib import Path
import json, hashlib, urllib.request, concurrent.futures, time
ROOT=Path(__file__).resolve().parents[1]
sums={}
for line in (ROOT/'references/SHA256SUMS.txt').read_text().splitlines():
    h,n=line.split(maxsplit=1); sums[n.lstrip('*').removeprefix('./')]=h
cfg=json.loads((ROOT/'configs/pilot.json').read_text())
files=[]
for sid in cfg['subject_ids']:
    ps=sorted(n for n in sums if n.startswith(f'sleep-cassette/SC4{sid:02d}') and n.endswith('-PSG.edf'))
    if not ps: raise RuntimeError(f'Missing subject {sid}')
    p=ps[0]; prefix=Path(p).name[:6]
    hy=[n for n in sums if Path(n).name.startswith(prefix) and n.endswith('-Hypnogram.edf')]
    assert len(hy)==1,(p,hy)
    files += [{'subject':sid,'path':n,'sha256':sums[n]} for n in [p,hy[0]]]
(ROOT/'data/manifest.json').write_text(json.dumps(files,indent=2))
def download(rec):
    n=rec['path']; out=ROOT/'data/raw'/Path(n).name
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()==rec['sha256']:return f'cached {out.name}'
    for attempt in range(4):
        try:
            with urllib.request.urlopen('https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/'+n,timeout=90) as r, open(str(out)+'.part','wb') as f:
                while b:=r.read(1024*1024): f.write(b)
            part=Path(str(out)+'.part')
            assert hashlib.sha256(part.read_bytes()).hexdigest()==rec['sha256'],n
            part.replace(out);return f'verified {out.name} {out.stat().st_size}'
        except Exception as e:
            print('retry',n,str(e),flush=True);time.sleep(2)
    raise RuntimeError(n)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for r in ex.map(download,files):print(r,flush=True)
print('DOWNLOAD COMPLETE',len(files),flush=True)
