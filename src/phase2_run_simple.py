from pathlib import Path
import os,json,subprocess,concurrent.futures
r=Path(__file__).resolve().parents[1];names=list(json.loads((r/"phase2/evaluation_lock.json").read_text())["models"])
def run(name):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
 with open(r/"phase2"/f"simple_{name}.log","a") as f:p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase2_simple.py"),"--name",name],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(name)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,names))
