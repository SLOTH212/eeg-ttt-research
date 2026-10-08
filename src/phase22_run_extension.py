from pathlib import Path
import subprocess,os,time,json,itertools
from concurrent.futures import ThreadPoolExecutor,as_completed
r=Path(__file__).resolve().parents[1];protocol=json.loads((r/"phase22/extension_protocol.json").read_text())
while len([f for f in (r/"phase22").glob("*/complete.json") if not f.parent.name.startswith("benchmark")])<12:time.sleep(30)
assert json.loads((r/"phase22/graph_benchmark.json").read_text())["status"]=="PASS"
env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
def job(case):
 source,objective,lr=case;name=f"{source}_{objective}_{lr:g}"
 if (r/"phase22"/name/"complete.json").exists():return name,0
 with (r/"phase22"/f"{name}.log").open("a") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase22_graph_screen.py"),"--source",source,"--objective",objective,"--lr",str(lr)],env=env,stdout=f,stderr=subprocess.STDOUT)
 return name,result.returncode
cases=list(itertools.product(protocol["sources"],protocol["objectives"],protocol["learning_rates"]))
failed=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for fut in as_completed([pool.submit(job,c) for c in cases]):
  name,code=fut.result();print(name,code,flush=True)
  if code:failed.append(name)
if failed:raise SystemExit(str(failed))
