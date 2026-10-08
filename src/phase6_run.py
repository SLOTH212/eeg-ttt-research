from pathlib import Path
import os,json,subprocess,concurrent.futures,time,hashlib
r=Path(__file__).resolve().parents[1];out=r/"phase6";lock=json.loads((out/"evaluation_lock.json").read_text())
while not (out/"data_audit.json").exists():time.sleep(3)
for name,h in lock["sha256"].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
for mode,names in [("baselines",list(lock["models"])),("controllers",list(lock["controllers"]))]:
 def run(name):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
  with open(out/f"{mode}_{name}.log","a") as f:p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase6_eval.py"),"--name",name,"--mode",mode],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(mode,name,p.returncode,flush=True)
  if p.returncode:raise RuntimeError((mode,name))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,names))
print("PHASE6 COMPLETE",flush=True)
