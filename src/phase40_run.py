from pathlib import Path
import os,subprocess,time
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while len(list((r/"phase39").glob("*/complete.json")))<3:time.sleep(15)
for script in ["phase39_select.py","phase40_evaluate.py"]:
 with (r/"phase40"/(script+".log")).open("a") as f:
  p=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src"/script)],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(script,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
