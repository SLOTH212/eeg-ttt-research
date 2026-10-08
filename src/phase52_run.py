from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase52";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2")
for name,args in [("task_only_audit",["src/phase52_task_gradient_audit.py"])]+[(k,["src/phase52_meta_head.py","--target",k]) for k in ["future","current","reconstruction"]]:
 with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
(o/"training_complete.json").write_text(json.dumps(dict(complete=True,variants=["future","current","reconstruction"])))
