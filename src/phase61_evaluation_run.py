from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase61";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2")
assert (o/"training_complete.json").exists()
for seed in [29,43]:
 with (o/f"seed{seed}"/"evaluate.log").open("a") as f:
  code=subprocess.call([str(r/".venv/bin/python"),"src/phase61_evaluate.py","--seed",str(seed)],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(seed,code,flush=True)
 if code:raise SystemExit(code)
(o/"evaluation_complete.json").write_text(json.dumps(dict(complete=True,seeds=[29,43])))
