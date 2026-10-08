from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase55";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2");jobs=[("graph_audit",["src/phase55_small_lr.py","--audit"])]+[(v+"_"+str(lr),["src/phase55_small_lr.py","--variant",v,"--lr",str(lr)]) for v in ["current","future","initial","reconstruction"] for lr in [1e-5,1e-4,.001]]
for name,args in jobs:
 with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
(o/"execution_complete.json").write_text(json.dumps(dict(complete=True)))
