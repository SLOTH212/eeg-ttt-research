from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase61"
for seed in [29,43]:
 env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2",EEG_META_SEED=str(seed))
 for variant in ["current","future"]:
  with (o/f"seed{seed}"/f"train_{variant}.log").open("a") as f:
   code=subprocess.call([str(r/".venv/bin/python"),"src/phase61_meta_head.py","--target",variant],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(seed,variant,code,flush=True)
  if code:raise SystemExit(code)
(o/"training_complete.json").write_text(json.dumps(dict(complete=True,seeds=[29,43])))
