from common import *
import subprocess
for fold in ['A','B']:
 for seed in [17,29,43]:
  src=ROOT/f'results/spectral_{fold}_{seed}/model.pt';out=ROOT/f'results/spectralhead_{fold}_{seed}';out.mkdir(exist_ok=True)
  ck=torch.load(src,map_location='cpu',weights_only=False);ck['writable_scope']='head';ck['source_checkpoint']=str(src);torch.save(ck,out/'model.pt')
  for script,label in [('evaluate.py','eval'),('supplement.py','supplement'),('policy_diagnostics.py','policy')]:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   with (ROOT/f'logs/{label}_spectralhead_{fold}_{seed}.log').open('w') as log:
    r=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'src'/script),'--model',str(out/'model.pt')],env=env,stdout=log,stderr=subprocess.STDOUT)
   print(fold,seed,label,r.returncode,flush=True)
   if r.returncode:raise RuntimeError(str((fold,seed,label)))
print('HEAD PROBE COMPLETE',flush=True)
