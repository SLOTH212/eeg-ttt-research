from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase52";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2");assert (o/"training_complete.json").exists()
jobs=[("reconstruction_validity",["src/phase52_evaluate.py","--validate"])]+[("held_"+k,["src/phase52_evaluate.py","--variant",k]) for k in ["initial","future","current","reconstruction","initial_normmatched"]]+[("summary",["src/phase52_summary.py"])]
for name,args in jobs:
 with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
