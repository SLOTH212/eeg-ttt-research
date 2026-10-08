from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase57";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2");assert (o/"state_pool_complete.json").exists()
jobs=[("implementation_audit",["src/phase57_factorial_meta.py","--audit"])]+[(v,["src/phase57_factorial_meta.py","--variant",v]) for v in ["fresh_ordinary","fresh_balanced","states_ordinary","states_balanced"]]
for name,args in jobs:
 with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
(o/"training_complete.json").write_text(json.dumps(dict(complete=True)))
