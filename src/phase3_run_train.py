from pathlib import Path
import os,subprocess,itertools,concurrent.futures
r=Path(__file__).resolve().parents[1]
def run(job):
 kind,fold,seed=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES="",EEG_THREADS="4" if kind=="eegnet" else "1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="4" if kind=="eegnet" else "1")
 with open(r/"phase3"/f"{kind}_{fold}_{seed}.log","a") as f:
  p=subprocess.run([str(r/".venv/bin/python"),str(r/"src/phase3_train.py"),"--kind",kind,"--fold",fold,"--seed",str(seed),"--epochs","25" if kind=="eegnet" else "30"],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(job,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(job)
jobs=list(itertools.product(["spectral","eegnet"],["young","middle","mixed"],[17,29,43]))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,jobs))
