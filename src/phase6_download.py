from pathlib import Path
import json,hashlib,urllib.request,concurrent.futures,time,os
os.sched_setaffinity(0,set(range(8,24)))
r=Path(__file__).resolve().parents[1];out=r/"phase6";out.mkdir(exist_ok=True);raw=out/"raw";raw.mkdir(exist_ok=True)
sums={n.lstrip("*").removeprefix("./"):h for h,n in (x.split(maxsplit=1) for x in (r/"references/SHA256SUMS.txt").read_text().splitlines())}
files=[]
for sid in range(40,59):
 ps=sorted(n for n in sums if n.startswith(f"sleep-cassette/SC4{sid:02d}") and n.endswith("-PSG.edf"));p=ps[0];prefix=Path(p).name[:6]
 hy=[n for n in sums if Path(n).name.startswith(prefix) and n.endswith("-Hypnogram.edf")];assert len(hy)==1
 files.extend(dict(subject=sid,path=n,sha256=sums[n]) for n in [p,hy[0]])
protocol=dict(purpose="Fresh-subject evaluation reserved until methods and hyperparameters are frozen",subjects=list(range(40,59)),record_rule="First available night for consecutive subjects40-58, selected before reading their outcomes; ages66-74 from public metadata",labels_status="Downloaded as opaque files; do not decode until evaluation lock exists",development_subjects=list(range(39)),allowed_cpu_ids=list(range(8,24)),gpu_allowed=False,files=files)
(out/"cohort_manifest.json").write_text(json.dumps(protocol,indent=2))
def get(rec):
 path=raw/Path(rec["path"]).name
 if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==rec["sha256"]:return path.name+" cached"
 for trial in range(4):
  try:
   with urllib.request.urlopen("https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/"+rec["path"],timeout=90) as response,open(str(path)+".part","wb") as f:
    while chunk:=response.read(1024*1024):f.write(chunk)
   part=Path(str(path)+".part");assert hashlib.sha256(part.read_bytes()).hexdigest()==rec["sha256"];part.replace(path);return path.name+" verified"
  except Exception as e:print("RETRY",path.name,str(e),flush=True);time.sleep(2)
 raise RuntimeError(path)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for msg in ex.map(get,files):print(msg,flush=True)
print("VERIFIED",len(files),"OPAQUE FILES",flush=True)
