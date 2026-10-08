from pathlib import Path
import os,subprocess
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="4",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
for source in ["young","mixed"]:
 if (r/"phase35"/source/"complete.json").exists():continue
 with (r/"phase35"/f"{source}.log").open("a") as f:
  result=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase35_head_train.py"),"--source",source],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(source,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
