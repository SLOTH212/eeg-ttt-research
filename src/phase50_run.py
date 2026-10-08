import subprocess,os,json,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];o=r/"phase50";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2");jobs=[("static",["src/phase50_static_matched.py","--source","mixed"])]+[(s,["src/phase50_scope_online.py","--scope",s]) for s in ["future_groups","allLN","current_groups","future_normmatched","current_normmatched"]]
for name,args in jobs:
 with (o/(name+".log")).open("a") as f:
  code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
(o/"execution_complete.json").write_text(json.dumps({"jobs":[n for n,a in jobs],"time":time.time()}))
