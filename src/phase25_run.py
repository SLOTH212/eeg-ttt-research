from pathlib import Path
import subprocess,os,time,json,itertools
from concurrent.futures import ThreadPoolExecutor,as_completed
r=Path(__file__).resolve().parents[1];out=r/"phase25";protocol=json.loads((out/"protocol.json").read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while len(list((out/"cache").glob("*.npz")))<8:time.sleep(10)
def job(case):
 source,policy,sid=case;name=f"{source}_{policy}"+(f"_s{sid:02d}" if sid is not None else "")
 if policy=="static_matched":cmd=[str(r/".venv/bin/python"),str(r/"src/phase25_static_matched.py"),"--source",source]
 else:cmd=[str(r/".venv/bin/python"),str(r/"src/phase25_gate.py"),"--source",source,"--policy",policy,"--subject",str(sid)]
 with (out/f"{name}.log").open("a") as f:result=subprocess.run(["taskset","-c","8-23"]+cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 return name,result.returncode
cases=list(itertools.product(protocol["sources"],protocol["learned_policies"]+["cadence2","cadence4"],protocol["validation_subjects"]))+[(source,"static_matched",None) for source in protocol["sources"]];failed=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for fut in as_completed([pool.submit(job,c) for c in cases]):
  name,code=fut.result();print(name,code,flush=True)
  if code:failed.append(name)
if failed:raise SystemExit(str(failed))
