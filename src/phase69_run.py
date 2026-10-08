from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];o=r/"phase69";env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2")
for kind in ["pretrained","random"]:
 with (o/(kind+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python"),"src/phase69_pulses.py","--kind",kind],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(kind,code,flush=True)
 if code:raise SystemExit(code)
(o/"execution_complete.json").write_text(json.dumps(dict(complete=True,carriers=2)))
