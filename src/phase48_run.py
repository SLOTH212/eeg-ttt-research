from pathlib import Path
import subprocess,os
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
for mode in ["masked","unmasked"]:
 with (r/"phase48"/f"{mode}.log").open("a") as f:
  p=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase48_gradient_groups.py"),"--mode",mode],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(mode,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
