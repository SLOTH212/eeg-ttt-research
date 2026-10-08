from pathlib import Path
import subprocess,os,time,json,itertools
from concurrent.futures import ThreadPoolExecutor,as_completed
r=Path(__file__).resolve().parents[1];out=r/"phase24";protocol=json.loads((out/"protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while len(list((r/"phase23").glob("*/s*.json")))<120:time.sleep(30)
def job(case):
 source,sid,trajectory=case;name=f"{source}_{trajectory}_s{sid:02d}"
 if (out/name/"complete.json").exists():return name,0
 with (out/f"{name}.log").open("a") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase24_pulses.py"),"--source",source,"--subject",str(sid),"--trajectory",trajectory],env=env,stdout=f,stderr=subprocess.STDOUT)
 return name,result.returncode
cases=list(itertools.product(protocol["sources"],protocol["controller_subjects"],protocol["trajectories"]));failed=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for fut in as_completed([pool.submit(job,c) for c in cases]):
  name,code=fut.result();print(name,code,flush=True)
  if code:failed.append(name)
if failed:raise SystemExit(str(failed))
