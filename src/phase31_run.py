from pathlib import Path
import subprocess,os,time,json
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while len(list((r/"phase30").glob("young_w*/complete.json")))<16:time.sleep(20)
with (r/"phase30/summary.log").open("w") as f:
 result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase30_summary.py")],env=env,stdout=f,stderr=subprocess.STDOUT)
if result.returncode:raise SystemExit(result.returncode)
for weight in [0.,.01]:
 with (r/"phase31"/f"weight{weight:g}.log").open("a") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase31_evaluate.py"),"--weight",str(weight)],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(weight,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
