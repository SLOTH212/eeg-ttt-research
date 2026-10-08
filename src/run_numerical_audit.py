import os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for k in ['spectral','spectralhead','eegnet']:
 for s in [17,29,43]:
  for f in ['A','B']:
   name=f'{k}_{f}_{s}';out=R/f'results/{name}'
   while not (out/'lr_sensitivity.json').exists():time.sleep(5)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   with (R/f'logs/numerical_{name}.log').open('w') as log:
    r=subprocess.run([str(R/'.venv/bin/python'),str(R/'src/numerical_audit.py'),'--model',str(out/'model.pt')],env=env,stdout=log,stderr=subprocess.STDOUT)
   print(name,r.returncode,flush=True)
   if r.returncode:raise RuntimeError(name)
