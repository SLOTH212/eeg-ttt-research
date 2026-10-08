from common import *
import subprocess,time,concurrent.futures,fcntl

def run(job):
 fold,seed=job;source=ROOT/f'results/eegnet_{fold}_{seed}';out=ROOT/f'results/eegnethead_{fold}_{seed}'
 while not (source/'summary.json').exists() and not (source/'training_complete.json').exists():time.sleep(5)
 out.mkdir(exist_ok=True);lock=(out/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX);ck=torch.load(source/'model.pt',map_location='cpu',weights_only=False);ck['writable_scope']='head';ck['source_checkpoint']=str(source/'model.pt');torch.save(ck,out/'model.pt')
 for script,label in [('evaluate.py','eval'),('supplement.py','supplement'),('lr_sensitivity.py','lr'),('policy_diagnostics.py','policy'),('numerical_audit.py','numerical')]:
  marker={'eval':'summary.json','supplement':'supplement.json','lr':'lr_sensitivity.json','policy':'policy_diagnostics.json','numerical':'numerical_audit.json'}[label]
  if (out/marker).exists():continue
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  with (ROOT/f'logs/{label}_eegnethead_{fold}_{seed}.log').open('a') as log:
   r=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'src'/script),'--model',str(out/'model.pt')],env=env,stdout=log,stderr=subprocess.STDOUT)
  print(fold,seed,label,r.returncode,flush=True)
  if r.returncode:raise RuntimeError(str((fold,seed,label)))
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(run,[(f,s) for s in [17,29,43] for f in ['A','B']]))
print('CNN HEAD PROBE COMPLETE',flush=True)
