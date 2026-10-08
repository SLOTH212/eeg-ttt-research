from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase60";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2");jobs=[("graph_audit",["src/phase60_memory_controls.py","--audit"])]+[(v+"_"+m,["src/phase60_memory_controls.py","--variant",v,"--mode",m]) for v in ["current","future"] for m in ["reset64","warmup64","warmup256","decay0.99","decay0.999"] if not(v=="future" and m=="reset64")]
for name,args in jobs:
 with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python")]+args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,code,flush=True)
 if code:raise SystemExit(code)
(o/"execution_complete.json").write_text(json.dumps(dict(complete=True)))
