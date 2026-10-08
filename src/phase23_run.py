from pathlib import Path
import subprocess,os,time,json,itertools,hashlib,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
r=Path(__file__).resolve().parents[1];out=r/"phase23";env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while len([f for f in (r/"phase22").glob("*/complete.json") if not f.parent.name.startswith("benchmark")])<20:time.sleep(30)
with (r/"phase22/summary.log").open("w") as f:subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase22_summary.py")],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
selection=json.loads((r/"phase22/source_selection.json").read_text());protocol=json.loads((out/"protocol.json").read_text());checkpoints={}
for source in protocol["sources"]:
 d=json.loads((r/"phase21/evaluation"/f"{source}_pretrained/selection.json").read_text());path=r/"phase21"/d["selected_directory"]/"model.pt";checkpoints[source]=dict(path=str(path.relative_to(r)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
for f in (r/"phase22").glob("*/complete.json"):
 if f.parent.name.startswith("benchmark"):continue
 d=json.loads(f.read_text());assert all(Path(v["checkpoint"])==r/checkpoints[d["source"]]["path"] for v in d["results"])
lock=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selections=selection["selections"],checkpoints=checkpoints,source_selection_sha256=hashlib.sha256((r/"phase22/source_selection.json").read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((out/"protocol.json").read_bytes()).hexdigest(),exploratory=True)
if (out/"selection_lock.json").exists():
 old=json.loads((out/"selection_lock.json").read_text());assert old["selections"]==lock["selections"] and old["checkpoints"]==lock["checkpoints"]
else:(out/"selection_lock.json").write_text(json.dumps(lock,indent=2))
def job(case):
 source,objective,sid=case;name=f"{source}_{objective}_s{sid:02d}"
 if (out/f"{source}_{objective}"/f"s{sid:02d}.json").exists():return name,0
 with (out/f"{name}.log").open("a") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase23_evaluate.py"),"--source",source,"--objective",objective,"--subject",str(sid)],env=env,stdout=f,stderr=subprocess.STDOUT)
 return name,result.returncode
cases=list(itertools.product(protocol["sources"],protocol["objectives"],protocol["evaluation"]));failed=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for fut in as_completed([pool.submit(job,c) for c in cases]):
  name,code=fut.result();print(name,code,flush=True)
  if code:failed.append(name)
if failed:raise SystemExit(str(failed))
