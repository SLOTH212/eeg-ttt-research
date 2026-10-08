import os,subprocess,time,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def run(job):
 k,f,s=job;name=f'{k}_{f}_{s}';out=R/f'results/{name}'
 while not (out/'supplement.json').exists():time.sleep(5)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (R/f'logs/identity_{name}.log').open('w') as log:
  r=subprocess.run([str(R/'.venv/bin/python'),str(R/'src/identity_control.py'),'--model',str(out/'model.pt')],env=env,stdout=log,stderr=subprocess.STDOUT)
 print(job,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(name)
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(run,[(k,f,s) for k in ['spectral','spectralhead','eegnet','eegnethead'] for s in [17,29,43] for f in ['A','B']]))
