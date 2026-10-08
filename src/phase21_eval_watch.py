from pathlib import Path
import subprocess,os,time
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="4",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
while True:
 with (r/"phase21/evaluation.log").open("a") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase21_evaluate.py")],env=env,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(result.returncode)
 n=len(list((r/"phase21/evaluation").glob("*/evaluation.json")))
 if n:
  with (r/"phase21/summary.log").open("w") as f:
   result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase21_summary.py")],env=env,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise SystemExit(result.returncode)
 print("evaluated carriers",n,flush=True)
 if n==4:break
 time.sleep(30)
