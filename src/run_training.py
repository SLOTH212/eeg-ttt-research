import os,subprocess,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[1]
jobs=[('spectral',f,s,30,2) for f in ['A','B'] for s in [17,29,43]]+[('eegnet',f,17,25,4) for f in ['A','B']]
def run(job):
 k,f,s,e,t=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS=str(t),OMP_NUM_THREADS=str(t),OPENBLAS_NUM_THREADS=str(t))
 with (R/f'logs/train_{k}_{f}_{s}.log').open('w') as log:
  r=subprocess.run([str(R/'.venv/bin/python'),str(R/'src/train.py'),'--kind',k,'--fold',f,'--seed',str(s),'--epochs',str(e)],env=env,stdout=log,stderr=subprocess.STDOUT)
 print(job,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(str(job))
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(run,jobs))
