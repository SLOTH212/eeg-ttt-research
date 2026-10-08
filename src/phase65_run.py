from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase65"
for seed in [29,43]:
 for target in ["current","future"]:
  env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2",STATIC_SEED=str(seed),META_LABEL_TARGET=target)
  with (o/f"seed{seed}"/f"static_{target}.log").open("a") as f:
   code=subprocess.call([str(r/".venv/bin/python"),"src/phase65_static.py","--source","mixed"],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(seed,target,code,flush=True)
  if code:raise SystemExit(code)
(o/"training_complete.json").write_text(json.dumps(dict(complete=True,seeds=[29,43],targets=["current","future"])))
