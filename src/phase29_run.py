from pathlib import Path
import subprocess,os,time
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while True:
 for source in ["young","mixed"]:
  for weight in [0.,.01,.1]:
   name=f"{source}_joint_{weight:g}"
   if not (r/"phase28"/name/"validation_audit.json").exists() or (r/"phase29"/name/"complete.json").exists():continue
   with (r/"phase29"/f"{name}.log").open("a") as f:
    result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase29_pulses.py"),"--source",source,"--weight",str(weight)],env=env,stdout=f,stderr=subprocess.STDOUT)
   print(name,result.returncode,flush=True)
   if result.returncode:raise SystemExit(result.returncode)
 if len(list((r/"phase29").glob("*/complete.json")))==6:break
 time.sleep(20)
