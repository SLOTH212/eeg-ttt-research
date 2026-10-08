from pathlib import Path
import os,subprocess,time
r=Path(__file__).resolve().parents[1];env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",EEG_THREADS="2",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
while not (r/"phase38/run_complete.json").exists():time.sleep(15)
for lr in [.0001,.001,.01]:
 with (r/"phase39"/f"entropy_{lr:g}.log").open("a") as f:
  p=subprocess.run(["taskset","-c","8-23",str(r/".venv/bin/python"),str(r/"src/phase39_entropy_screen.py"),"--lr",str(lr)],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(lr,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
