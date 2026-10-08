from pathlib import Path
import subprocess,os
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
for source in ["young","mixed"]:
 for variant in ["unfitted","fitted"]:
  name=f"{source}_{variant}"
  if (r/"phase36"/name/"complete.json").exists():continue
  with (r/"phase36"/f"{name}.log").open("a") as f:
   result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase36_pulses.py"),"--source",source,"--variant",variant],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(name,result.returncode,flush=True)
  if result.returncode:raise SystemExit(result.returncode)
