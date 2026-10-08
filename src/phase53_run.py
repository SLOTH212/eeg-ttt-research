from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase53";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2")
assert (o/"static_current/complete.json").exists()
jobs=[("static_future",["src/phase53_static.py","--source","mixed"],dict(env,META_LABEL_TARGET="future")),("graph_audit",["src/phase53_online.py","--audit"],env)]+[(v+"_"+t,["src/phase53_online.py","--variant",v,"--trajectory",t],env) for v in ["current","future","reconstruction","initial"] for t in ["continuous","reset64"] if not(v=="initial" and t=="continuous")]
for name,args,ev in jobs:
 with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=ev,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
(o/"execution_complete.json").write_text(json.dumps(dict(complete=True,jobs=[n for n,a,e in jobs])))
