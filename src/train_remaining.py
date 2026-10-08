import os,subprocess,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def run(job):
 f,s=job;name=f'eegnet_{f}_{s}';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with (R/f'logs/train_{name}.log').open('a') as log:
  r=subprocess.run([str(R/'.venv/bin/python'),str(R/'src/train.py'),'--kind','eegnet','--fold',f,'--seed',str(s),'--epochs','25'],env=env,stdout=log,stderr=subprocess.STDOUT)
 print(job,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name)
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(run,[(f,s) for s in [29,43] for f in ['A','B']]))
