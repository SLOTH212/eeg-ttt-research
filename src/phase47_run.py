from pathlib import Path
import subprocess,os
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
for args in [["--audit"],["--mode","masked"],["--mode","unmasked"]]:
 name="_".join(args).replace("--","")
 with (r/"phase47"/(name+".log")).open("a") as f:
  p=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase47_pulses.py")]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
