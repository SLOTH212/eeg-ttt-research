from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase59";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2")
assert json.loads((o/"graph_audit.json").read_text())["status"]=="PASS"
for trajectory in ["continuous","reset64"]:
 for sid in [7,17,27,37]:
  name=f"lr0.0001_{trajectory}_s{sid:02d}"
  with (o/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python"),"src/phase59_pulses.py","--subject",str(sid),"--lr",".0001","--trajectory",trajectory],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,code,flush=True)
  if code:raise SystemExit(code)
(o/"pulses_complete.json").write_text(json.dumps(dict(cases=8,complete=True)))
