from pathlib import Path
import os,json,time,subprocess
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while not (r/"phase45/mixed_static_matched/complete.json").exists():time.sleep(15)
commands=[["--prepare"]]+[["--policy",policy,"--subject",str(sid)] for policy in ["future36","current36","low_entropy_write","anchor_guard","cadence2","cadence4"] for sid in [5,15,25,35]]
for args in commands:
 name="_".join(args).replace("--","")
 with (r/"phase45"/(name+".log")).open("a") as f:
  p=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase45_gate.py")]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
(r/"phase45/run_complete.json").write_text(json.dumps(dict(complete=True),indent=2))
