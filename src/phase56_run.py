from pathlib import Path
import subprocess,os
r=Path(__file__).resolve().parents[1];env=dict(os.environ,OPENBLAS_NUM_THREADS="1",EEG_THREADS="2",OMP_NUM_THREADS="2")
for folder,name,script in [("phase55","full_zero_forward","phase55_full_zero_audit.py"),("phase55","summary","phase55_summary.py"),("phase56","matched_states","phase56_matched_states.py"),("phase56","summary","phase56_summary.py")]:
 with (r/folder/(name+".log")).open("a") as f:code=subprocess.call([str(r/".venv/bin/python"),"src/"+script],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(folder,name,code,flush=True)
 if code:raise SystemExit(code)
