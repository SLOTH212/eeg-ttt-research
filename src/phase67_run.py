from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase67"
for seed,target in [(17,"future"),(29,"future"),(43,"current"),(43,"future")]:
 env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2",STATIC_SEED=str(seed),META_LABEL_TARGET=target)
 with (o/f"seed{seed}"/f"static_{target}.log").open("a") as f:
  code=subprocess.call([str(r/".venv/bin/python"),"src/phase67_static.py","--source","mixed"],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(seed,target,code,flush=True)
 if code:raise SystemExit(code)
(o/"training_complete.json").write_text(json.dumps(dict(complete=True,new_cases=4,reused_epoch5_cases=2)))
