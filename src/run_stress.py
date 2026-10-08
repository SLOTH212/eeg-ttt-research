import os,subprocess,time,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def run(job):
 k,f=job;name=f'{k}_{f}_17';out=R/f'results/{name}'
 while not (out/'supplement.json').exists():time.sleep(5)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (R/f'logs/stress_{name}.log').open('w') as log:
  r=subprocess.run([str(R/'.venv/bin/python'),str(R/'src/stress.py'),'--model',str(out/'model.pt')],env=env,stdout=log,stderr=subprocess.STDOUT)
 print(job,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name)
with concurrent.futures.ThreadPoolExecutor(1) as pool:list(pool.map(run,[(k,f) for k in ['spectral','eegnet'] for f in ['A','B']]))
