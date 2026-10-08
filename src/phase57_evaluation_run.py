from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase57";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2");assert (o/"training_complete.json").exists()
jobs=[("reconstruction_validity",["src/phase57_evaluate.py","--validity"])]+[("held_"+v,["src/phase57_evaluate.py","--variant",v]) for v in ["original","fresh_ordinary","fresh_balanced","states_ordinary","states_balanced"]]
for name,args in jobs:
 with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
(o/"evaluation_complete.json").write_text(json.dumps(dict(complete=True)))
