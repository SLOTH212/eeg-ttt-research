from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase71"
for seed in [29,43]:
 env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2",CBRAMOD_CLASSIFIER_SEED=str(seed))
 for lr in [2e-5,.0001]:
  with (o/f"seed{seed}"/f"pretrained_{lr:g}.log").open("a") as f:
   code=subprocess.call([str(r/".venv/bin/python"),"src/phase71_train.py","--kind","pretrained","--lr",str(lr)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(seed,lr,code,flush=True)
  if code:raise SystemExit(code)
(o/"training_complete.json").write_text(json.dumps(dict(complete=True,new_configs=4,seeds=[29,43])))
