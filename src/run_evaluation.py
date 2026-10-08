import os,subprocess,concurrent.futures,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
jobs=[(k,f,s) for k in ['spectral','eegnet'] for f in ['A','B'] for s in ([17,29,43] if k=='spectral' else [17])]
def run(job):
 k,f,s=job;name=f'{k}_{f}_{s}';path=R/f'results/{name}/model.pt'
 while not (R/f'logs/train_{name}.log').exists() or 'TRAIN COMPLETE' not in (R/f'logs/train_{name}.log').read_text():time.sleep(5)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (R/f'logs/eval_{name}.log').open('w') as log:
  r=subprocess.run([str(R/'.venv/bin/python'),str(R/'src/evaluate.py'),'--model',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
 print(job,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(str(job))
with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(run,jobs))
