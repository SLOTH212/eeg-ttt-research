import os,subprocess,concurrent.futures,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def call(name,script,args):
 threads='4' if script=='train.py' else '1'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EEG_THREADS=threads,OMP_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,MKL_NUM_THREADS=threads)
 with (R/f'logs/{name}.log').open('a') as log:
  r=subprocess.run([str(R/'.venv/bin/python'),str(R/'src'/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' return '+str(r.returncode))
 print('DONE',name,flush=True)
def run(job):
 k,f,s=job;name=f'{k}_{f}_{s}';out=R/f'results/{name}';out.mkdir(exist_ok=True)
 trainlog=R/f'logs/train_{name}.log'
 if not trainlog.exists() or 'TRAIN COMPLETE' not in trainlog.read_text():call('train_'+name,'train.py',['--kind',k,'--fold',f,'--seed',str(s),'--epochs',str(25 if k=='eegnet' else 30)])
 if not (out/'summary.json').exists():call('eval_'+name,'evaluate.py',['--model',str(out/'model.pt')])
 if not (out/'supplement.json').exists():call('supplement_'+name,'supplement.py',['--model',str(out/'model.pt')])
# At most two experiment workers, one compute thread each, all pinned to CPU 8-23.
jobs=[('spectral',f,s) for s in [17,29,43] for f in ['A','B']]+[('eegnet',f,s) for s in [17,29,43] for f in ['A','B']]
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(run,jobs))
print('ALL CORE EXPERIMENTS COMPLETE',flush=True)
